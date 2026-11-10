# midtown 样条性能优化计划（CubeS₂ / QuadS）

范围：**lr-pt 侧图内算子** `latent_charge_sum` 的 midtown 后端
（`source/op/pt/long_range/midtown/`），即 `sog_gpu.cu` / `sog_spline.h` / `sog_green.h`。
LAMMPS 树（`deepmd-kit-LR`）的 `sog.cpp` 不在本文范围内。

配套阅读：

| 文档 | 关系 |
|---|---|
| [SPLINE_MATRIX_BENCH.md](SPLINE_MATRIX_BENCH.md) | 十配置 × 两尺寸的实测主表与 `A + B·N` 分解（**注意：本文 §2 用设备级计时修正了它的归因**） |
| [MIDTOWN_AD_DESIGN.md](MIDTOWN_AD_DESIGN.md) | AD gather 无分支化的完整设计。**本文 M0 的实测把它的收益估计压到止损线以下**，见 §4 P1 |
| [AD_EVERYWHERE_PLAN.md](AD_EVERYWHERE_PLAN.md) | 全线切 AD、退役 ik 的正确性理由 |
| [IK_MODE_ACCURACY.md](IK_MODE_ACCURACY.md) | AD / ik 的误差律实测（ik 恰好高一阶） |
| [MESH_ERROR_MODEL_AND_FORM_A.md](MESH_ERROR_MODEL_AND_FORM_A.md) | 网格规则与 Form A 的来源 |

---

## 0. TL;DR（已含 M0 实测结论）

**M0 已完成，它的结果推翻了本计划初稿的核心假设。** 结论：

| 结论 | 依据 |
|---|---|
| **AD 路径上已经没有值得做的 mesh 优化** | CubeS₂-6 ad 的设备侧 solve 只占整步 **20%**，其中 gather 只占 **4.6%**。gather 改 −20~40% ⇒ 整步 **−0.9~−1.8%**，**低于 [MIDTOWN_AD_DESIGN.md §7](MIDTOWN_AD_DESIGN.md) 自己定的 3% 止损线**。**四个尺寸（5 184→526 848）复测后结论不变**，见 §1.3 |
| **整套 mesh 里唯一的大头是 ik 的 gather** | 18.6 ms = 整步 33%，且占它自己 solve 的 **72–93%**；但它属于 P2 建议**退役**的路径 |
| **真正剩下的机会不在 mesh 里，而在 solve 之外的 ~9 ms** | LR 相对 SR-only 多花 17.8 ms，设备 solve 只解释其中 8.84 ms（50%），另一半在电荷网络/图节点 |

因此本文的行动项被重新排序：

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| **M0** | 修三处计时缺陷 | ✅ **已完成** | 下面 §1 的数字就是它产出的 |
| **P0** | AD gather/spread 的"逐位一致"小修 | 建议做 | 正确性收益，不图性能（[MIDTOWN_AD_DESIGN.md §4.6](MIDTOWN_AD_DESIGN.md)） |
| **P1** | AD gather 无分支化 | ❌ **止损，不做** | 实测收益 < 2% |
| **P2** | ik 退役 | 建议做 | 决策 + 文档 |
| **P3** | ik gather 的 float4 合并 | ⏸ 仅当 P2 决定保留 ik | 唯一的大头（−16~−25%） |
| **P4** | 查 solve 之外的 ~9 ms | **新，优先级最高** | 这是 mesh 之外的真瓶颈 |

---

## 1. M0 实测：设备侧五段拆分（**本文的权威数据**）

条件：526 848 原子、`np=2`、replicated、`run 150` + `run 150`（第二段稳态）、
`SOG_GPU_TIMING=1`。数值在多次打印间稳定在 ±0.5%。

### CubeS₂-6 AD（384³ = 56.6 M 点），整步 43.5 ms

| 阶段 | ms/step | 占 solve | 占整步 |
|---|---|---|---|
| spread（含 `k_zero_fx`、`k_midtown_frac`、`k_fx_to_float`） | 2.79 | 32% | 6.4% |
| fft_fwd（R2C） | 1.60 | 18% | 3.7% |
| kspace（`k_kspace_h`） | 0.87 | 10% | 2.0% |
| fft_inv（C2R，1 次） | 1.57 | 18% | 3.6% |
| **gather** | **2.01** | **23%** | **4.6%** |
| **设备侧 solve 合计** | **8.84** | 100% | **20.3%** |

### CubeS₂-6 ik（256³ = 16.8 M 点），整步 56.9 ms

| 阶段 | ms/step | 占 solve | 占整步 |
|---|---|---|---|
| spread | 1.08 | 5% | 1.9% |
| fft_fwd | 0.33 | 2% | 0.6% |
| kspace | 0.23 | 1% | 0.4% |
| fft_inv（C2R，**4 次**，且含 3 次 `k_mul_ik` 的排队） | 1.69 | 8% | 3.0% |
| **gather** | **18.59** | **85%** | **32.7%** |
| **设备侧 solve 合计** | **21.9** | 100% | **38.5%** |

三个直接结论：

1. **ik 的 gather 是 AD 的 9.3×**（18.59 vs 2.01）。它每节点读 4 个独立
   `cudaMalloc` 的 float 数组（742 MB 散乱 4 B 读 vs AD 的 185 MB）——
   §3.2 的机制得到定量确认。
2. **spread 依赖网格大小**：同样 526 848 原子、同样 88 次定点原子加/原子，
   384³ 上 2.79 ms，256³ 上 1.08 ms（2.6×）。原子加在更大的地址空间上散得更开。
   这意味着"缩网格"同时便宜了 spread——但 ik 的 gather 把这份收益全吃掉还有余。
3. **stage 之和（8.84 / 21.9）远小于 [SPLINE_MATRIX_BENCH.md §4.5](SPLINE_MATRIX_BENCH.md)
   的 k-space 归因（16.8 / 29.8）**。差的那 ~8 ms 两种模式都一样多，说明它在
   **solve 之外**，不在 mesh 里（见 §2）。

### 1.3 尺寸阶梯：gather 的占比随尺寸怎么变

四个尺寸、同一个产物（`lr_mid6` / `lr_mid6ik`），`np=2`、`run 150`+`run 150`。
**绝对时间不可用**（见下面的污染声明），但 `solve` 与 `gather` 都由同一窗口内的
GPU event 测得，**两者的比例是干净的**。

| N | 网格 | solve (ms) | **gather (ms)** | gather / solve | 若 P1 过线，整步须 < |
|---|---|---|---|---|---|
| 5 184 | 81³ | 0.106 | 0.032 | **30.2%** | **0.43 ms** |
| 12 288 | 108³ | 0.169 | 0.055 | **32.5%** | **0.73 ms** |
| 98 304 | 216³ | 1.178 | 0.368 | **31.2%** | **4.89 ms** |
| 526 848 | 384³ | 8.840 | 2.010 | **22.7%** | 26.73 ms（实测 43.5） |

`gather / solve` 在小体系上**确实更高**（30–33% vs 526 k 的 23%）——
[MIDTOWN_AD_DESIGN.md](MIDTOWN_AD_DESIGN.md) 的"gather 占 23%"是针对**算子**说的，
在那个口径下它是对的。

**但它救不了 P1**：P1 只改 gather，整步收益 = `gather × f`（f ≤ 0.40），
要过 3% 就必须 `step < 13.3 × gather`。上表最后一列就是这个门槛：
12 288 上要 **0.73 ms**，5 184 上要 **0.43 ms**——**低于任何一个 DPA4 前向 +
MD 记账的合理下限**（同窗口的 SR-only 基线在严重争用下也只有 11.9 ms，
即便按 526 k 实测的约 2.3× 争用因子折算回去仍有 ~5 ms）。
**所以 P1 在全部四个尺寸上都是 STOP，小体系更差。**

同一阶梯对 ik 给出相反结论：**ik 的 gather 占它自己 solve 的 72–93%**
（5 184 / 12 288 / 98 304 / 526 848 → 72.1% / 78.5% / 93.1% / 84.8%）。
**ik 本质就是"gather + 一点别的"**，P3 的 float4 合并对它是通杀，
且在哪个尺寸都成立。

> **污染声明**：这一轮阶梯（2026-11-11 00:43 起）与 4 个 `deepmd --pt-expt train`
> 训练任务（00:42:45 启动）以及一个第三方 NEP-SOG 任务共用 4 张卡（93–98% 利用率）。
> 所以 `Loop time` 被抬高（SR-only 基线 11.9–20.6 ms 明显偏高），
> **绝对 ms 一律不引用**；上表的 `solve` / `gather` 之比仍是 GPU event 口径，可用。
> §1.1 / §1.2 的 526 k 数据测于 **23:45–23:50，训练任务启动之前**，是干净的。


---

## 2. 这些数字推翻了什么

[SPLINE_MATRIX_BENCH.md §4.5](SPLINE_MATRIX_BENCH.md) 用
`k-space = A + B·N` 的双点拟合，把 CubeS₂-6 ad 的成本拆成 A = 9.7 ms（每原子）
和 B·N = 7.1 ms（每格点）。设备级计时给出的真实值是：

| | 双点拟合 | **设备级实测** | 偏差 |
|---|---|---|---|
| 每原子部分 | A = 9.7 ms | spread + gather = **4.80 ms** | 拟合**高估 2×** |
| 每格点部分 | B·N = 7.1 ms | fft_fwd + kspace + fft_inv = **4.04 ms** | 拟合高估 1.8× |
| 合计 | 16.8 ms | solve = **8.84 ms** | 差 8 ms（在 solve 之外） |

**教训（写进文档以免再犯）**：双点拟合把"整步时间随网格的变化"全部归给 k-space，
但网格变大同时改变了展开方式、缓存行为、以及图里的其它节点。**要归因必须用
设备级 event 计时，不能用整步时间的差分。** 这也是 M0 必须先做的根本理由。

同时它也说明：**LR 相对 SR-only 的 +17.8 ms 里，只有一半是 mesh 计算**。
另一半（~9 ms，两种力模式一样多）在电荷网络的额外图节点、隐电荷广播、
AOTI 图调度里——那不是 midtown 样条的问题，本文 P4 只是把它指出来。

---

## 3. 三个已核实的结构性事实

### 3.1 AD gather 每节点重算一次带 5 路分支的权重梯度

- **散布**有逐原子一维表（`Cs2Tab4` / `Cs2Tab`，`cs2_fill_tab4:359`、
  `cs2_fill_tab_par:442`），逐节点只查表 + 乘加。
- **AD 收集没有表**：`k_gather_cubes2_warp_r:811` 在节点循环里直接调
  `cs2_wg6:323`，每节点一次 `if (cls==0) … else if (cls==1||cls==2) … else if (cls==3) … else …`。
  一条 warp 的 32 lane 持 32 个不同 `cls` → **整条 warp 把 5 个分支全走一遍**。
- 对照：B-spline 的 `interpolate_owners`（`mesh_sum.cu:731`）用每轴 6 个
  `value`/`slope` 支撑 216 个节点，内层无分支。

[MIDTOWN_AD_DESIGN.md §3](MIDTOWN_AD_DESIGN.md) 已证明每类权重都是
**≤4 项"一维函数之积"之和**（Python 参考验证到 1.1e-16）。
**机制成立，但见 §4 P1 的收益止损。**

### 3.2 ik 把收集变成 4 路读 + 4 次逆变换 + 3 次谱乘

`sog_gpu.cu:3128-3146`：

| | AD | ik |
|---|---|---|
| `k_mul_ik`（半谱乘） | — | **3 次** |
| `cufftExecC2R` | 1 次（势场） | **4 次**（3 梯度 + 势场） |
| gather 每节点读的数组 | `pot` × 1 | `pot` + `gradx` + `grady` + `gradz` × **4** |

`gradx/grady/gradz` 是**三块独立 `cudaMalloc`**（`ensure_ik_grads`），
与 `pot` 共四处不同内存区，无法合并事务。

量级（256³、526 848 原子）：一次 `k_mul_ik` 的半谱元素数 = 8 454 144 = **原子的 16.05×**；
三次 ≈ **406 MB** 流量；三个梯度网格 ≈ **269 MB** 额外显存；
gather 每节点 load 指令 AD 88 次 → ik **352 次**。

**反直觉但已确认（实测支持）**：ik 的 gather **算术比 AD 更少**
（1 个值权重 vs 值 + 3 个导数权重；fp32 乘 vs fp64 乘），却慢 9.3×。
**所以它纯粹是访存问题**——4 路散乱 4 B 读的扇出代价。这也说明
**P3（float4 合并）会把大部分收益拿回来**。

### 3.3 势场逆变换**不能**因为"能量改走谱归约"而省掉

`k_midtown_finish2:273` 里

```
grad[i] = phi - m*mean,   phi = (vpot[i] - prefactor*kernel_zero*q[i])*m
```

**`charge_gradient` 依赖 `vpot`**，而它**不是**只有 `fix sog/response` 才用——
本仓库模型自己的前向就消费它（零值线性项，其 autograd 反向把隐电荷响应送进原子力；
`dpa4c` 路径把它加到描述符余切上）。**普通 MD 里也必须物理正确。**

再加一层：**AD 模式下 gather 对同一个势场网格求导**——
`k_gather_cubes2_warp_r:842` 的 `u = pot[...]` 同时喂 `dtx/dty/dtz` 和 `gp`。
所以在 AD 路径上，即使丢掉 per-atom 能量，势场的 C2R 和那一路读**一次都省不掉**，
只省掉四路 FMA 里的一路。

（"看着像机会其实不是"：`k_kspace_h` 的 `red[0]` 是完整的非零模能量，
**已经在算却被丢弃**（只读 `red[1..6]`）。但 `atom_energy` 是**逐原子**输出、
`red[0]` 是每帧一个标量，替换它会破坏算子契约。**不可利用。**）

---

## 4. 优化项（按 M0 实测重排）

### P0（建议做）：AD gather 与 spread 的权重逐位一致

[MIDTOWN_AD_DESIGN.md §4.6](MIDTOWN_AD_DESIGN.md)：`cs2_w6_tab`（散布）与
`cs2_wg6`（收集）是两条不同代数路线，同一节点算出的 `w` 差 ~1e-7。
统一成"四项一维之积"后**逐位一致**，算子的伴随性从"近似"变"精确"，
对 NHC 能量守恒与响应力都有好处。

**这一步不依赖性能收益**，是正确性改进；工作量约 1 d。
若要做 P3，它也是 P3 的前置（P3 的收集段就是本设计去掉 `d` 项）。

### P1（❌ 止损，不做）：AD gather 无分支化

**实测否决**：gather 在 CubeS₂-6 ad 上是 2.01 ms = 整步 **4.6%**。
即使拿到设计文档上限（gather −40%），整步也只有 **−1.8%**。
[MIDTOWN_AD_DESIGN.md §7](MIDTOWN_AD_DESIGN.md) 自己写的止损线是"端到端 < 3% 则不做"，
**实测落在止损线以下 2 倍**。

它的旧测点（196608 原子/250³ 上 gather 占 23%）为什么误导：那个尺寸下
`B·N` 更小、gather 占比更高；到 526 k、384³，FFT 三段（4.04 ms）反过来是 gather 的 2×。

**保留 P0 的逐位一致小修；撤回无分支重写。**

### P2（建议做）：ik 退役

三条独立证据：

1. **它慢**：整步 56.9 vs 43.5 ms（+31%）。
2. **它的收益只有显存**：等精度 256³/3878 MiB vs 384³/4836 MiB，省 958 MiB 换 +13.4 ms。
3. **它有正确性代价**：ik 的力不是 `E_mesh` 的梯度（非变分），力/能量/维里不自洽，
   NVE 有系统性漂移（[AD_EVERYWHERE_PLAN.md §1.1](AD_EVERYWHERE_PLAN.md)）。
   而 lr-pt 的网格规则**本就按 AD 标定**，用 ik 属于过度分辨。

两个已核实、影响操作方式的事实：

1. **`force_mode` 默认值本来就是 `"ad"`**（`argcheck.py:3966-3983`、模块签名、
   `config.get` 回落、反序列化回落、kernel 签名五处一致；算子整数默认 `0` = AD）。
   矩阵里的 ik 行都是**用户显式要求**的结果。
2. **切换 `force_mode` 会连带改变网格**：`mesh_spacing()` 把 `force_mode` 传进
   `grid_rule.spacing()`，`IK_FORCE_RULE` 的常数比 `FORCE_RULE` 小 5–10 倍。
   **"把 ik 改成 ad"不能只改那一个键**——网格会按 AD 规则重新变细。
3. **不做"按尺寸自动选 ad/ik"**。硬约束：kernel 包装层
   （`deepmd/pt_expt/kernels/long_range.py:601-610`）为保持 AOTI 冻结图的
   **参数个数**不变，只在非默认值时才把两个整数追加进算子参数表。
   "每次调用按原子数决定 force_mode"会与固定 arity 冲突；而能看见原子数的调用点
   看不见配置，能看见配置的模块层（`dpmodel`）看不见原子数。**当前架构下不可行。**

**建议**：把 ik 标为"实验性、非默认、不保证性能"，在 `PAIR_DEEPMD_KK_SOG.md`
选型表写明 **ad 是唯一推荐值**。是否物理删除单独决策（会破坏已冻结的 `.pt2`）。

### P3（仅当 P2 决定保留 ik）：ik gather 的 4 路读合并

**这是整套 mesh 里唯一的大头**：18.59 ms = 整步 32.7%。

- 把 `pot` / `gradx` / `grady` / `gradz` 交错成一个 `float4` 数组，
  每节点 4 次散乱 4 B 读 → 1 次 16 B 读（sector 数 ÷4）。
- 预期 gather 18.6 → **4~6 ms**，整步 56.9 → **~43 ms（−24%）**，回到 AD 的量级。
- **顺带修一个缺陷**：`k_gather_cubes2_warp_r_ik:903` 的 `want_pot` 参数
  **完全未被引用**——即使 `want_pot=0` 也照读 `pot[m]` 并写 `vpot[i]`。
- **验收**：与现 ik **逐位一致**（只改内存布局，不改算术）。

### P4（新，优先级最高）：solve 之外的 ~9 ms

LR 相对 SR-only 多花 17.8 ms，设备 solve 只解释 8.84 ms。剩下的 ~9 ms
**两种力模式一样多** ⇒ 与样条无关。候选：电荷网络的额外前向、隐电荷广播、
AOTI 图节点调度、`pack` 阶段的设备同步等待
（`SOG_MIDTOWN_CPU` 在 np=1 上显示 `pack` = 15.2 ms，而那一步只读 12 个系数
——**那是同步等待，不是计算**）。

**这已经超出"midtown 样条优化"的范围**，但它是这个尺度上的真瓶颈，
所以在本文里点名，交回模型侧处理。

---

## 5. 验收

| 项 | 门槛 |
|---|---|
| **计时（M0）** | 五段之和与独立测得的整步时间、以及 `Loop time` 互相对得上；多次打印间 < ±1% |
| **数值（P0/P3）** | 同一网格上 gather 前后权重 `w` **逐位一致**；端到端力/能量相对差 < 1e-6 |
| **物理** | 水 / NaCl / 界面三体系上 E、F、P 与改动前 RMS ≤ 1e-6 |
| **性能（P3）** | `docs/bench_spline_matrix.py`，CubeS₂-6 ik 526 848 / 1 119 744 两档 |
| **回归** | `test_long_range_sum.py`、`test_mesh_backend.py`、`common/dpmodel/test_long_range.py` 全过 |
| **图捕获** | 新表是 `__constant__`/`__shared__`，无新分配，AOTI 捕获不受影响 |

**度量纪律**（已踩过的坑，全部写进代码注释）：

1. `run 6` 会把首次邻居表/图捕获算进 Loop time → 用 `run N` + `run M` 取第二段。
2. `SOG_GPU_TIMING` 每 100 次调用才打印一次 → 至少要跑过 100 步，且要跳过前 100 次。
3. 归因**必须**用设备级 event，**不能**用整步时间的双点差分（§2 的 2× 高估就是这么来的）。

---

## 6. 里程碑

| 步 | 内容 | 状态 |
|---|---|---|
| **M0** | 补齐 15 个 `cudaEventRecord` + 检查返回码并清粘滞错误 + 修 `CpuTimer` 的 flush 门 + `cudaGetLastError` 兜底 | ✅ **已完成**（见 §8 的三处缺陷） |
| **M1** | P0：统一 AD gather/spread 的权重代数，逐位一致 | 待做，~1 d |
| **M2** | P2：ik 退役文档化 + `PAIR_DEEPMD_KK_SOG.md` 选型表 | 待做，~0.5 d |
| **M3** | P4：定位 solve 之外的 ~9 ms（本文范围外，交模型侧） | 待评估 |
| **M4** | P3（条件） | 仅当 P2 决定保留 ik，~1 d |

---

## 7. 与既有计划的关系

- [MIDTOWN_AD_DESIGN.md](MIDTOWN_AD_DESIGN.md) 是 P1 的详细设计。
  **本文用实测把它的收益估计压到它自己的止损线以下**，因此 P1 不做；
  但它的 §4.6（逐位一致）升格为 P0，它的 §3（≤4 项一维之积）成为 P3 的前置。
- [AD_EVERYWHERE_PLAN.md](AD_EVERYWHERE_PLAN.md) 处理**正确性统一**
  （LR 树 B-spline 从 ik 换 AD）。本文 P2 引用其结论支撑"ik 退役"，但**不执行**它的 LR 侧改动。
- 两者交集在该文 §3.7（先正确性、再性能），本文与之不冲突：P0 是正确性，P1 已撤回。

---

## 8. M0 修掉的三个计时缺陷（含根因）

| # | 缺陷 | 后果 |
|---|---|---|
| 1 | **10 个 event 里 9 个只 `cudaEventCreate`、从不 `cudaEventRecord`**（全文件唯一一个 record 是 `evt_gather_stop`） | 对未记录的 event 调 `cudaEventElapsedTime` 返回 `cudaErrorInvalidResourceHandle(400)` 且**不写输出**；`float ms;` 未初始化 → 五个累加器全是垃圾，打印 `nan`/巨数 |
| 2 | 该错误**粘滞**在 thread error state，随后的 `cudaGetLastError()` 仍返回它（C10 的 launch 检查就读这个） | 下一次 torch 算子抛 **"CUDA error: invalid resource handle"** ——这就是 `SOG_GPU_TIMING` 会崩的机制。修法是失败后 `cudaGetLastError()` 清掉，不让诊断开关毒化整跑 |
| 3 | `CpuTimer::flush()` 被写在 `if (owner_begin==0 && owner_count==n)` 里 | np>1 时每个 rank 只拥有一个切片，**计时器永远不打印**——恰好在最需要它的地方沉默。已移到所有权判断之外（`flush()` 自身已按 50 次节流） |

外加一个未修的已知小问题：十个 event 从不 `cudaEventDestroy`，
设了 `SOG_GPU_TIMING` 的进程泄漏一组（单例，有界）。
slab 路径在 `sog_gpu.cu` 里仍**没有**设备侧分段，只有 `midtown_sum.cu` 的 host 侧
`SOG_SLAB_TIME`；若要在 slab 上做同类归因需另加。

---

## 9. 诚实的不确定性

1. **本文全部实测都在 replicated、np=2、526 848 原子、正交盒上。** slab 路径的
   spread/gather 与 replicated 共用同一批 kernel，但 transpose/halo 会改变重叠，
   占比可能不同。**要下 slab 的结论必须先测 slab。**
2. ~~**只看了一个尺寸**~~ —— **已补测（§1.3）**：5 184 / 12 288 / 98 304 / 526 848
   四个尺寸上 `gather/solve` 是 30.2% / 32.5% / 31.2% / 22.7%，
   **小体系的份额确实更高，但 P1 的整步门槛（0.43 / 0.73 / 4.89 ms）低到不可能达到**。
   **P1 在全部尺寸上 STOP。** 这条不确定性已关闭，且关闭方向与初稿的猜测相反。
3. **P4 的 ~9 ms 只是减法得到的余量**，还没有直接测量。定位它需要
   在模型前向里加分段（`sync 计时` 或 nsys），**不在本计划的执行范围内**。
4. **QuadS 未被本文测量。** 它的 A 项在矩阵里更差（42–49 ns/atom），
   且 216 节点是**严格可分离**的，理论上比 CubeS₂ 更容易无分支化；
   但 P1 在 CubeS₂ 上已被否决，QuadS 同理大概率不在止损线之上。
   若 P3 落地（ik 保留），QuadS 的 ik 会共享同一套 float4 改造。
