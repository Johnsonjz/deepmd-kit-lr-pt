# midtown 网格引擎优化方案：让 CubeS₂ / QuadS 真正兑现"网格更少"的优势

> 范围：`source/op/pt/long_range/` 下的图内算子 `latent_charge_sum`，即
> `mesh_sum.cu`（B-spline 基准）与 `midtown_sum.cu` + `midtown/sog_gpu.cu`
> （CubeS₂ / QuadS）。本文只讨论**实现结构**；网格规则与误差律是另一层，
> 只在 §2.7 点名。
>
> 配套脚本（本次新增，均不依赖 GPU）：
> - [`verify_cs2_terms.py`](repro/verify_cs2_terms.py)：验证"≤4 项一维之积"的
>   值与三个导数，逐节点、两阶、随机亚格点。
> - [`mesh_memory_ledger.py`](repro/mesh_memory_ledger.py)：按源码布局计算
>   B-spline / midtown 当前 / midtown 目标的设备 buffer 账本。
>
> 环境：`PY=/data/home/jiangzhen/.conda/envs/lr212/bin/python`。

---

## 0. 结论（TL;DR）

**"网格更少但不快、不省显存"不是单一问题，而是三件独立的事被同一个数字掩盖了。**

| # | 现象 | 真正原因 | 结论 |
|---|---|---|---|
| 1 | 等精度网格更少，CubeS₂-6 时间只持平 | **修复前**：每原子赋值不随网格缩小；CubeS₂-6 每节点成本 2.6× B-spline，吃掉 88 vs 216 的节点优势。**修复后**（§0.1）：mesh 本身 B-spline 0.514 ms vs CubeS₂ 0.385 ms（同网格 −25%）、LR −5%、整步 −0.7%——网格优势是真的，但 mesh 只占每步 0.4/9.3 ms。4 项收缩是**第二阶**：gather −25~40%，预计 LR 再 −1~3% | 第一阶（stream/capture）已完成；第二阶按需做，不是翻倍项 |
| 2 | 显存反而更大 | **修复前**：graph 捕获失败 → eager 回退，AOTI graph pool / 每步分配把峰值抬高到 +724/+2298 MiB；**不是 mesh buffer**（静态账本 midtown 是 B-spline 的 0.62–0.63×）。修复后 +312/+1102 MiB，方向与账本一致。仍真实存在的浪费：full-`nx` green 2×、double 力/势、cuFFT plan 自动分配顺序 | 先修 capture/stream；再做 compact green 等真实浪费 |
| 3 | QuadS 网格最小却最慢 | pre-fix 的 49 vs 17 ns 是 eager/默认 stream 下测的。修复后 operator graph replay：QuadS@81³ **0.405 ms**、CubeS₂@96³ **0.377**、B-spline@108³ **0.517**。QuadS 仍慢于 CubeS₂，**"旗舰"判断撤回**；表填两次是可做的微优化，预期 5–15% | 不再作为旗舰；CubeS₂ 已是三者中 mesh 最优 |

**目标（等精度、同原子数；按 CUDA 审查后的保守口径）：**

> 下表是**修复前**的规划口径。§0.1 的实跑已经先拿到 CubeS₂ 的 LR −5%；
> P1 与 P3/P4 是在 mesh/显存内部继续优化，不要把这部分和"修复 bug 带来的 −5%"
> 重复计算。QuadS 的 49 ns/27.1 ms 已撤回，不再作为目标。

| 指标 | 修复前口径 | 目标 | 依据 |
|---|---|---|---|
| CubeS₂-6 AD operator @96³ | 0.377 ms（已测） | **0.30–0.33 ms** | gather −25~40%，spread 不变；不是 2× |
| QuadS-6 AD operator（81³/108³） | pre-fix 49 ns | **0.405 / 0.413 ms（已测）** | 修复后仍慢于 CubeS₂；表填一次只作微优化 |
| LR 时间（526 848，dpa4 campaign） | B-spline 8.84 / CubeS₂ 8.46 ms（已测） | **8.10–8.25 ms（P1 后）** | 不再引用 SPLINE_MATRIX 的 11.1/27.1（不同网格） |
| 设备 buffer（331 776 原子） | midtown 96 MiB | **80 MiB**（B-spline 152 MiB 的 0.52×） | `mesh_memory_ledger.py` |
| 首次调用瞬态 | host 0.1–1.9 GiB、device 4 B/pt、0.6–12 s | **host < 1 MiB；首个 shape < 0.5 s，之后 cell 重建 < 50 ms** | 每轴表 + Form A 按 shape 缓存 |

**一句话：CubeS₂ 的"网格更少"在修复 stream/capture 后已经兑现为 mesh −25%、
LR −5%、显存 −4~5%；整步只有 −0.7%，因为 mesh 只占 0.4/9.3 ms。
QuadS 修复后仍慢于 CubeS₂（0.405 vs 0.377 ms），旧的"49 ns/旗舰"判断
不成立；后续优先级是 CubeS₂ 的 P1 + 显存 P3/P4，而不是 P2。**

> **QuadS 的现状**：修复后它仍慢于 CubeS₂（0.405 vs 0.377 ms），已降级为
> 微优化候选。若还要评估它，先做 P7 的力精度审计；旧的 42 ns/27.1 ms
> 是 pre-fix，不要引用。**CubeS₂ 是当前的最优 mesh。**

---

## 0.1 复核结果（2026-11-13，RTX 4090 实跑）

**结论先行：之前"速度/显存不占优"的结论是在两个 bug 下测的。修复后，CubeS₂-6 AD 在等精度下已经略快、略省显存；网格优势是真的，但整步只被它拉动了不到 1%。**

### 两个 bug

1. **CUDA graph 捕获失败。** `midtown_sum.cu` 把 CUDA 上的 `coefficient` 用
   `.to(at::kCPU)` 拉回 host；AOTI 在 capture 中禁止这种未 pin 的 CPU/CUDA
   copy，异常被 proxy executor 吞成"输出未定义"，整个模型**回退 eager**。
   修复前日志：
   ```
   Exception in aoti_torch: Cannot copy between CPU and CUDA tensors during
   CUDA graph capture unless the CPU tensor is pinned.
   WARNING: deepmd-kit could not capture the evaluation into a CUDA graph ...
   ```
2. **mesh kernel 不在当前 stream 上。** 即使捕获"成功"，`sog_gpu.cu` 的
   `<<<...>>>`、`cudaMemset`、`cufftExec` 全部走 legacy default stream；
   AOTI 在 side stream 上 capture，**整个 mesh 根本没进 graph**。
   直接 capture `latent_charge_sum` 时，mid6ad 的 replay 只有 **5.5 µs**
   （空图），bs6 是 **0.51 ms**。

**修复**（本次已落地并安装）：

- `midtown_sum.cu`：CUDA 系数改为 **pinned host mirror**，按 tensor 身份缓存；
  首次 eager 调用填充，后续（包括 capture）命中缓存，不再做未 pin 的 D2H。
- `sog_gpu.cu`：`sog_gpu_compute_core` 与 `midtown_sum` 的 kernel、memset、
  `cufftSetStream`、event 全部改用 `at::cuda::getCurrentCUDAStream()`；
  `cudaMemsetAsync(d_red, ..., stream)`。

### 复测：等精度（A 链，replicated，AD，GPU 3 空闲）

| N | 配置 | 网格 | 修复前 LR / 显存 | **修复后 LR / 显存** |
|---:|---|---:|---:|---:|
| 98 304 | B-spline-6 | 108³ | 1.30 ms / +326 MiB | 1.30 ms / +328 MiB |
| 98 304 | CubeS₂-6 AD | 96³ | 1.40 / **+724** | **1.23 / +312** |
| 331 776 | B-spline-6 | 160³ | 5.30 / +1152 | 5.21 / +1152 |
| 331 776 | CubeS₂-6 AD | 144³ | 4.99 / **+2298** | **4.94 / +1102** |
| 526 848 | B-spline-6 | 180³ | — | 8.84 / +1834 |
| 526 848 | CubeS₂-6 AD | 162³ | — | **8.46 / +1736** |

（修复前 98k 的 +724 是 `backend_9_98304.csv` 的原始值，本次在 GPU 3 上也复现为
+724；修复后同一脚本同一张卡为 +312。331k 同理：+2298 → +1102。**这不是采样噪声**，
而是 eager 回退时 AOTI graph pool / 每步分配的真实开销。）

**结果**：CubeS₂-6 AD 在等精度下 **LR −4~5%、峰值显存 −4~5%**（98k/331k/526k 三点一致），整步 −0.7~0.9%。
静态账本的方向（midtown 更小）是对的；修复前测到的是 bug，不是引擎。

**全配置复核**：B-spline 4/6、CubeS₂ 4/6（AD/ik）、QuadS 4/6（AD/ik）十种
配置的 98k/331k 等精度表、逐配置结论和"试过但回退"的负结果，见
[`MIDTOWN_ALL_CONFIGS.md`](MIDTOWN_ALL_CONFIGS.md)。要点：
- 4 阶：CubeS₂-4 / QuadS-4 AD 是两强（1.18/1.16 ms @98k；4.79/4.87 @331k），
  比 B-spline-4 快 29–39%，显存低 7–32%。
- 6 阶：CubeS₂-6 / QuadS-6 AD 约 1.23 ms，比 B-spline-6 快 5%、省 4–5%。
- ik：QuadS-6 ik 1.27/5.23 ms，接近 AD；CubeS₂-6 ik 1.70/6.48 ms，是唯一
  明确不推荐的配置（value-only gather 读 4 张网格）。
- QuadS-4/6 AD 的等精度 artifact 是本轮新冻结的
  `dpa4c-sog-quads4ad.pt2` / `dpa4c-sog-quads6ad.pt2`。
- **ik 口径更正**：本 campaign 的能量规则下 ik 与 AD **同网格**
  （4 阶 125³、6 阶 96³），ik 只买力的阶数；CubeS₂-6 ik 的 gather
  实测量 0.620 ms（profiler），比 AD 慢 0.47 ms 且无网格收益，退役；
  6 阶 ik 用 QuadS-6 ik（gather 0.166 ms，1.27 ms），4 阶 ik 用
  QuadS-4/CubeS₂-4。独立微基准显示 raw kernel 只需 0.216 ms，
  op 上下文的 ~0.4 ms 未能用 ncu 归因（`ERR_NVGPUCTRPERM`），
  详见 [`MIDTOWN_ALL_CONFIGS.md`](MIDTOWN_ALL_CONFIGS.md) §2。

### 复测：同网格算子（`torch.cuda.CUDAGraph` 直接 capture op）

| 配置 | 网格 | graph replay |
|---|---:|---:|
| B-spline-6 | 108³ | 0.514 ms |
| CubeS₂-6 AD | 96³（等精度） | **0.410 ms** |
| CubeS₂-6 AD | 108³（同网格） | **0.385 ms** |
| QuadS-6 AD | 81³（等精度） | **0.405 ms** |
| QuadS-6 AD | 108³（同网格） | 0.413 ms |

> **QuadS 的更正**：修复后 81³ 只比 108³ 快 2%（0.405 vs 0.413），
> 说明它的成本仍由 per-atom 赋值主导，网格缩小几乎不兑现；它比
> CubeS₂ 的 0.377–0.385 慢。**旧文档的 49 ns/atom 与
> "QuadS 是旗舰、LR 27.1→10–12 ms"是 pre-fix 数据，撤回。**
> QuadS 的"表填两次"仍是一个可做的微优化，但预期只有 5–15%，
> 不是 2.5–2.9×。

同网格下 mesh 本身 **−25%**；等精度下 **−20%**。整步只 −0.7%
（98k：9.333 → 9.268 ms），因为 mesh 只占每步 ~0.4/9.3 ms。

### 正确性

- 直接 capture 后**修改坐标再 replay**：bs6 与 mid6ad 的
  `max|ΔF|(graph − eager) = 0`（逐位），`|ΔF(new−old)| > 0`，说明 mesh 确实在
  replay 里执行，而不是冻结在 capture 时刻。
- `test_mesh_backend.py`：**20 passed**。
- `test_long_range_sum.py`：**50 passed**。

### 对前文的更正

- §1.1 的表格是**修复前**的旧 CSV，保留用于对照；不要再用它判断引擎优劣。
- §1.5 的"+724/+2298 是采样/瞬态污染"要修正为：**主因是 eager 回退**；
  run-to-run 散布的 caveat 仍成立，但不是这次系统性差异的来源。
- §1.3 的"18 vs 17 ns/atom"来自修复前的整步差分，已失效；同网格 graph replay
  显示 mesh 是 0.514 vs 0.385 ms（见上）。kernel 级优化（P1/P2）仍然有价值，
  但它们是**第二阶**的；第一阶是 stream/capture。

---

## 1. 现状证据：时间到底花在哪

### 1.1 等精度网格与实测

数据来自 `sog_work/bench/matrix_midtown/backend_9_{98304,331776}.csv`（A 链，
`pme_accuracy=1e-4`，LR = SOG − SR）。**下表是修复前的旧 build**，保留用于
对照；修复后的数字见 §0.1。

| N | 后端 | 网格 | LR ms | LR MiB | 备注 |
|---:|---|---:|---:|---:|---|
| 98 304 | B-spline-6 AD | 108³ | 1.401 | +326 | 基准 |
| 98 304 | CubeS₂-6 AD | **96³** | 1.397 | +724 | 网格 −30%，时间持平，显存 +122% |
| 98 304 | CubeS₂-4 AD | **125³** | 1.305 | +748 | 对 bs4 216³：网格 −81%，时间 −27%，显存 +53% |
| 98 304 | CubeS₂-4 ik | 125³ | 1.358 | +780 | |
| 331 776 | B-spline-6 AD | 160³ | 5.236 | +1152 | 基准 |
| 331 776 | CubeS₂-6 AD | **144³** | 4.991 | +2298 | 网格 −27%，时间 −5%，显存 +99% |
| 331 776 | CubeS₂-4 AD | **192³** | 4.905 | +2390 | 对 bs4 320³：网格 −78%，时间 −37%，显存 +85% |
| 331 776 | CubeS₂-4 ik | 192³ | 5.411 | +2502 | |

**读法：** 网格确实更少。CubeS₂-6 的时间只持平，CubeS₂-4 的时间赢 27–37%，
但两者的显存都输；显存差距远大于网格差距。
后一条本身就是一个矛盾信号——如果网格更少、每格点 buffer 又相近，
显存不可能翻倍。§1.5 给出账本。

### 1.2 每步成本由"每原子赋值"决定，不由网格决定

> **2026-11-13 更正**：下面的 A/B 拟合与 17/18/49 ns 是 **pre-fix**
> （graph 捕获失败、mesh 走默认 stream）数据，绝对值已失效；保留它是因为
> 它正确揭示了"per-atom 占比高、缩网格收益小"这一结构性事实。
> 修复后的 operator graph replay 见 §0.1。

`SPLINE_MATRIX_BENCH.md` §4.5 用 `k-space = A + B·N_百万点` 拟合 526 848 原子：

| 配置 | A（ms/step，固定+每原子） | B（ms/百万格点） | 每原子赋值 |
|---|---:|---:|---:|
| B-spline-6 AD | **8.9** | 0.137 | **17 ns/atom** |
| CubeS₂-6 AD | 9.7 | 0.126 | **18 ns/atom** |
| CubeS₂-6 ik | 25.9 | 0.241 | **49 ns/atom** |
| QuadS-6 AD | 26.0 | 0.093 | **49 ns/atom** |
| QuadS-6 ik | 21.9 | 0.201 | 42 ns/atom |

同一尺寸内，网格从 225³（11.4 M 点）到 750³（422 M 点）变 37 倍，
每步只从 74.5 ms 到 98.9 ms（1.33 倍）。在 225³–384³ 这些生产网格上，
**B·N 只有 1–7 ms，而 A（每原子赋值）是 9–26 ms。** 这就是"缩网格为什么不快"的直接答案：

> 在等精度对上，midtown 比 B-spline 少的 25–27% 格点只值
> `B·ΔG ≈ 0.14 ms`（331 k）到 `0.5 ms`（526 k / 225³ vs 250³）；
> 而每原子赋值是 9–26 ms。**优化 spread/gather 的收益是缩网格的 10–40 倍。**

### 1.3 CubeS₂：节点数优势被每节点成本吃掉

> **2026-11-13 更正**：下面的 18 vs 17 ns、2.6× 每节点成本是 pre-fix
> 数据。修复后同网格 operator graph replay 是 **B-spline 0.514 ms vs
> CubeS₂ 0.385 ms（−25%）**，CubeS₂ 已经兑现节点数优势。下面的 SASS 模型
> 仍然用于 P1（第二阶）的收益估计，但不要再引用 18/17 ns 作为现状。

CubeS₂-6 的节点数是 **88**，B-spline-6 是 **216**（0.407×）。
但实测每原子赋值是 18 vs 17 ns（1.06×），于是：

```
CubeS₂ 每节点等效成本 / B-spline 每节点等效成本
  = (18/88) / (17/216)
  = 0.205 / 0.0787
  = 2.60
```

> 这里 18/17 ns 是"每步 assigning 成本 ÷ 原子数"的拟合值，包含固定的
> launch/规约开销；把它摊到节点只用于**同口径比较**，不是单节点指令周期。
> 但两个引擎在同一体系、同一测量窗口下测得，比值是可用的。

**2.45× 的节点数优势，被 2.60× 的每节点成本完整抵消。**

CUDA 审查（sm_89 SASS，按 warp round 统计，class 在编译期固定、其余节点字段
运行时）给出了更精确的账：

| 实现 | 每原子 warp 指令 | 相对今天 | 寄存器 | occupancy |
|---|---:|---:|---:|---|
| 今天：`c_nodes6c` + `cs2_wg6` | 748+585+404 = **1737** | 1.0× | 63 | 4 block（1024 线程） |
| 只换 `c_nodes6s`、仍用 `cs2_wg6` | 1443+1443+1173 = **4059** | **2.3× 回归** | — | — |
| **4 项收缩 + `c_nodes6s`** | **483** | **0.28×** | **39** | **6 block（1536 线程）** |

> 这是 SASS 成本模型（class 在编译期固定、其余字段运行时），不是 ncu 实测；
> 绝对值只用于排序，不用于承诺墙钟。

两个关键修正（它们改变了本方案的叙事）：

1. **当前不是 5 路发散。** `k_gather_cubes2_warp_r` 读的是
   `c_nodes6c`（class 排序，`sog_gpu.cu:980`），三轮分别是
   `{cls0+cls1}`、`{cls2+cls3}`、`{cls3+cls4}`，**每轮最多 2 路**。
   `MIDTOWN_AD_DESIGN.md:30-41` 的"5 路"描述已经过时。
2. **不能只改排序。** 把节点表换成 offset 排序、权重仍用 `cs2_wg6`，
   指令数会从 1737 涨到 4059（2.3× 回归），因为 offset 排序后每轮的 class
   组合从 2 类变成最多 5 类。**4 项收缩是 offset 排序的前提**，不是可选优化。

**合并访存的收益也远比想象小。** 以 `nx=ny=250` 为例：

| 节点表 | sector/atom（32 B） | 说明 |
|---|---:|---|
| `c_nodes6c`（今天） | 53 | 同一轮同 class，但网格上分散 |
| `c_nodes6s`（目标） | 36 | 同一轮 32 个连续 offset |
| 理论下限 | 32 | 整个 88 节点模板本身只覆盖 32 个 32 B sector |

即 **1.47×**，不是 4–8×。原因：一行只有 6 个 `dx`，32 个 lane 必然跨
~5.3 个 y 行，每轮落 11–13 个 128 B line；36 已经离下限 12%。

所以真正值钱的是 **3.6× 的指令数 + 50% 的 occupancy**，而不是"去分支/合并"。
这也解释了为什么收益是温和的：gather 只占 LR 的 ~18%，且不是纯指令受限。
**修复后的真实对比是 operator mesh 0.385 ms（CubeS₂ 108³）vs 0.517 ms（B-spline 108³）；P1 的目标是 0.30–0.33 ms，而不是 9–12 ns 的旧口径。**

每节点贵在哪里（`sog_gpu.cu:966-1032`）：

```cuda
// 当前 gather：每个节点一次 cs2_wg6
const GpuWeightGrad wg = (ORD==4) ? cs2_wg4(...) : cs2_wg6(tf[0],tf[1],tf[2],nd,kXi6);
const float u = live ? pot[m] : 0.f;
dtx += wg.dx*u; ...
```

`cs2_wg6`（`sog_gpu.cu:478-484`）在每节点上做一次 5 类分支 + 4 分量前向模式
多项式；class 排序把它压到每轮 2 类，但每节点仍要重算整条多项式。

### 1.4 QuadS：旧结论已撤回

> **2026-11-13 更正（重要）**：`49 vs 17 ns`、`2.5–2.9×`、`LR 27.1→10–12 ms`
> 都是 **pre-fix**（eager / 默认 stream）数据，已失效。修复后
> `torch.cuda.CUDAGraph` 直接 capture 的 operator replay：
>
> | 配置 | 网格 | replay |
> |---|---:|---:|
> | B-spline-6 | 108³ | 0.517 ms |
> | CubeS₂-6 | 96³ / 108³ | **0.377 / 0.385 ms** |
> | QuadS-6 | 81³ / 108³ | 0.405 / 0.413 ms |
>
> **QuadS 仍慢于 CubeS₂**；网格从 108³ 缩到 81³ 只省 2%（0.413→0.405），
> 因为每原子赋值主导。旧的"QuadS 是旗舰"判断撤回。

结构上仍有一个可做的微优化：B-spline 的 `node_stencils` 是独立稠密 pass，
算一次、存 216 B/atom；QuadS 在 spread 和 gather 里各调一次
`quads_fill_tab`。把表移到稠密 pass 预计能省 5–15%，但**不足以让 QuadS
超过 CubeS₂**，也不足以支撑"2.5–2.9×"的旧估计。

**验证顺序**：如果还要做 P2，先做 A/B；若 operator replay 收益 < 10%，
就停，把精力放在 CubeS₂ 的 P1 和显存 P3/P4。

### 1.5 显存：实测峰值不是 mesh buffer

> **2026-11-13 更正**：修复前的 `+724/+2298 MiB` 主因不是采样噪声，而是
> **CUDA graph 捕获失败后的 eager 回退**（AOTI graph pool / 每步分配）。
> 修复后同一脚本在 GPU 3 上是 **+312/+1102 MiB**，方向与下面的静态账本一致。
> 旧 CSV 的显存数字不能再用于引擎比较；下面的 buffer 账本仍然有效。

按源码布局逐项算（`mesh_memory_ledger.py`，不含 cuFFT work area）：

| N | 网格 | B-spline | midtown 当前 | midtown 目标 |
|---:|---|---:|---:|---:|
| 98 304 | mid 96³ / bs 108³ | 45.8 MiB | **28.5 MiB（0.62×）** | **23.6 MiB（0.52×）** |
| 331 776 | mid 144³ / bs 160³ | 152.4 MiB | **96.2 MiB（0.63×）** | **79.7 MiB（0.52×）** |

修复前 CSV 说 331 776 原子上 midtown 比 B-spline 多 **+1146 MiB**；
这个差值是整个 midtown buffer（96 MiB）的 **12 倍**，不可能来自 mesh 布局。
现在原因已经查清：**graph 捕获失败 → eager 回退**，AOTI graph pool / 每步
分配把峰值抬高。修复后同一脚本是 **+1102 MiB vs B-spline +1152 MiB**，
方向与账本一致。旧 CSV 的显存列不要再用于引擎比较。

**同一个 bs6 配置在 98 304 原子上的三次测量是 1724 / 1882 / 2930 MiB
（1.2 GiB 的 run-to-run 散布）；mid6ik 是 2138 / 1952 / 3000（1.05 GiB）。
声称的 398–1146 MiB 家族差，比同一配置的重复测量噪声还小**，所以不可归因。

但采样器并非完全失灵：**同一次 campaign 内的 grid-scaled 差值能对上账本**——
`mid6ad → mid6ik` 是 2298 → 2346 = **+48 MiB**，而 16 B/pt × 2.986 M = +45.6；
`mid4 → mid6` 是 748 → 724 = **−24 MiB**，而 20 B/pt × 1.07 M = −20.4；
`QuadS4ik − mid4ik` 是 +8，而 4 B/pt = +7.8。这说明：
**grid buffer 本身能被这个采样器测到；bs-vs-mid 的那个 offset 是一个
per-process 常数**（邻居进程、torch reserved-but-cached、AOTI graph pool），
不是 mesh 布局。

结论：**当前显存数字被"首次调用瞬态 + 整卡采样 + 其他进程/分配器保留"污染，
不能用来判断引擎优劣。** 真实的可优化项仍然存在，见 §2 R3 与 §4 P3–P5：

1. **green 表按 full `nx` 存储**：`f_gp` 是 `nx*ny*nz` 个 `float2`，
   但 `k_kspace_h` 只读 `kh < nx/2+1` 的列。**一半的表永远不读**，
   8 B/pt → 4 B/pt。
2. **`ge` 与 `gf` 完全相同**（`sog_green.h:164-167`），host 仍建三份
   `double`（24 B/pt），再打包成 `float2`（8 B/pt）。
3. **`d_force`/`d_vpot` 是 double**（24+8 B/atom），而输出力是 float；
   gather 写 double、`copy_` 转 float、finish 再读 double——一次多余往返。
4. **每次调用 10+ 个 `at::empty`/`at::zeros`**，B-spline 只有 1 个 workspace。
5. **首次调用**：host 上 `ge/gf/gv` 24 B/pt + `fgp` 8 B/pt（32 B/pt，
   144³ ≈ 100 MiB，384³ ≈ 1.9 GiB）；device 上 Form A 的 `acc` 只多 4 B/pt
   （11 / 216 MiB）；但 Form A 要跑 `nth³ = 512` 次 **host 驱动、同步**的
   spread+FFT，每次一次 `ng·4 B` 的 H2D，144³ 约 5.9 GB、0.6 s，
   384³ 约 116 GB、12 s。**这是 host + latency 尖峰，不是 GiB 级 device 尖峰。**
6. 单 shape 的 `MidtownState`：网格一变就 `free_grid_buffers` + 重建 plan，
   B-spline 有 16 个 shape 的 LRU plan cache 和 graph hold。
7. **cuFFT plan 的创建顺序**：midtown 先 `cufftPlan3d`（自动分配 work），
   再 `cufftSetAutoAllocation(0)` + `cufftSetWorkArea`（`sog_gpu.cu:3221-3235`）；
   B-spline 是 `cufftCreate` → `cufftSetAutoAllocation(0)` → `cufftMakePlanMany64`
   （`mesh_sum.cu:1086-1097`）。前者的自动分配可能已经常驻，等于多一份 work
   area。改成 B-spline 的顺序，并让 R2C/C2R 共享同一块 work area。

### 1.6 测量口径（必须先修）

现有 `backend_9_*.csv` 的 `lr_mib` 来自 `bench_backend_6model.py:78-99`：
**0.1 s 轮询 `nvidia-smi --query-gpu=memory.used`（整卡）**，取 `peak - base`。
它同时包含：模型、AOTI graph pool、cuFFT plan/work、首次 green/Form A 瞬态，
以及**同一张卡上其他进程**。`docs/MESH_ENGINE_TEST_REPRO.md` §3.4 已经写明
"0.1 s 采样会漏掉/碰巧抓到首次分配尖峰"，并用 20 ms 重采样；但
`backend_9_*.csv` 用的仍是 0.1 s。

修正后的口径（建议直接写进 `perf_iso_grid.py`）：

```bash
# 1) 进程级，而不是整卡
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
# 2) 20 ms 采样，分别报 steady（最后 0.8 s 中位数）与 peak
# 3) 打开引擎自己的账本
SOG_MEM_BREAKDOWN=1 ...
# 4) 首次调用单独测：第 1 次 vs 第 10 次的峰值差
```

**验收时以"steady"为主、peak 为辅**；首次瞬态单独列，不计入稳态对比。

---

## 2. 根因清单

### R1 — gather 每节点重算权重梯度；offset 排序在旧权重下不可行
- 位置：`sog_gpu.cu:966-1032`（`k_gather_cubes2_warp_r`）、`:478-484`（`cs2_wg6`）。
- 现状：`c_nodes6c` class 排序，每轮 2 类、1737 warp instr/atom；
  88 节点 × 2.6× 每节点成本 ≈ 216 节点 B-spline 的成本。
- **不能只改排序**：`c_nodes6s` + 旧 `cs2_wg6` = 4059 instr/atom（2.3× 回归）。
- **4 项收缩 + `c_nodes6s`** = 483 instr/atom、39 regs、6 block（今天 63 regs、4 block）。
- coalescing 只从 53 降到 36 sector/atom（1.47×，下限 32）。
- 结论：收益主要来自**指令数 + occupancy**，不是"去 5 路发散"；gather 只占
  LR 的 ~18%，所以整体是温和收益（LR −5~12%）。
- 对照：B-spline `interpolate_owners`（`mesh_sum.cu:731-837`）读预先算好的
  `value/slope/offset`，内层无分支、无多项式。

### R2 — 逐原子一维表没有独立成 pass
- CubeS₂：gather 没有表，每节点重算整条多项式；修法是在 gather block 内
  用 shared 填 `Cs2Basis`（不建全局表，保持显存优势）。
- QuadS：`quads_fill_tab` 在 spread 和 gather 各调一次；修法是补一个稠密
  stencil pass，写 216 B/atom 全局表，spread/gather 只读。
- 对照：B-spline `node_stencils`（`mesh_sum.cu:254-287`）一次算好 3 轴
  `value/slope/offset`，spread 和 gather 复用。
- 后果：QuadS 49 vs 17 ns（**最大单项**）；CubeS₂ 每节点多项式无法摊薄（温和）。

### R3 — green 全 `nx` 表 + `ge/gf` 重复 + cuFFT 顺序 + 首次瞬态
- 位置：`sog_gpu.cu:3213`（`f_gp` `ng*sizeof(float2)`）、`:1234-1377`（只读 `nh`）、
  `sog_green.h:164-167`（`ge=gf`）、`:3221-3235`（plan 创建顺序）。
- 后果：设备 2× 冗余；host 24 B/pt 建表 + 8 B/pt 打包；Form A 512 次同步迭代；
  plan 可能多留一份 auto-allocated work area。
- 修法：复用 `build_green_slab_half` 的 compact 半谱布局（`sog_green.h:177-315`），
  让 replicated 与 slab 共用一份 builder；plan 改成
  `cufftCreate → cufftSetAutoAllocation(0) → cufftMakePlan3d`。

### R4 — double 力/势临时量 + 逐次分配
- 位置：`midtown_sum.cu:685-704`（`d_force` double、`d_vpot` double、`copy_`、`reshape`）。
- 后果：24+8 B/atom 设备临时量；多 2–3 个 kernel；与 B-spline 的 float 输出不一致。
- 修法：gather 直接写 float；finish 读 float；tilted 旋转在 kernel 内做。

### R5 — 单 shape state，无 plan cache
- 位置：`midtown_sum.cu:557-668`；对照 `mesh_sum.cu:1040-1286`（`MeshPlans` LRU + graph hold）。
- 后果：NPT / 训练中网格变化时销毁并重建 cuFFT plan；graph 无法安全持有。
- 修法：`MidtownPlans` 按 `(device, nx,ny,nz, spline, order, form)` 缓存，
  `cudaGraphRetainUserObject` 持有；green 按 cell 单独重建。

### R6 — ik 路径的结构性成本
- 位置：`sog_gpu.cu:3692-3712`；每次 3 张 float 梯度网格（12 B/pt）+
  半谱 scratch（4 B/pt）+ 3 次 `k_mul_ik` + 3 次额外 C2R；gather 每节点 4 路散读。
- 后果：CubeS₂-6 ik 49 ns/atom（AD 18）；只在 CubeS₂-4 上因为力的阶数才有价值。
- 修法：**默认退役**；若要保留，把 `pot/gx/gy/gz` 交织成 `float4`（一次 16 B 读），
  并把 `k_mul_ik` 融合进 green 乘法。

### R7 — 网格规则（物理层，非 CUDA）
- 在 526 k / 174 Å 的 campaign 里 CubeS₂-6 是 **384³ vs B-spline-6 250³**
  （网格更大），此时无论引擎多好都不可能省显存/时间。
- A 链的 `grid_rule.spacing()` 对 midtown 仍套 `d ≤ 0.95 σ_min/(√2 ξ)` 的
  方差扣除上限；**Form A 默认开启时这个上限在物理上已无必要**（窗口被精确除掉）。
- 但这是精度/规则改动，必须单独做误差审计，不要和 CUDA 重构混在一起。

---

## 3. 目标架构：与 B-spline 同构的三段式

```
                    B-spline (mesh_sum.cu)              midtown target
per-atom metadata   node_stencils<Order>                 k_midtown_meta
                    (fp64, dense, 1 thread/atom)        (frac + family 1-D table)
                    -> AxisStencil[3] 216 B/atom        -> 336 B (CubeS2) / 216 B (QuadS)
spread              spread_charges<Order>                k_midtown_spread
                    (warp/atom, slot, fixed-point)      (same; table lookup)
FFT                 cuFFT R2C/C2R                        cuFFT R2C/C2R
k-space             apply_influence                      k_midtown_kspace
                    (on-the-fly 12 Gaussians)           (compact green / per-axis)
gather              interpolate_owners<Order>            k_midtown_gather
                    (value+slope contraction)           (≤4-term contraction)
finish              owner_slices + finish_owners        reuse / same shape
plans               MeshPlans (LRU + graph hold)         MidtownPlans (same)
workspace           Workspace (1 allocation)             Workspace (same)
```

### 3.1 逐原子元数据（两家共用一套骨架）

| 家族 | 每原子表 | 大小 | 说明 |
|---|---|---:|---|
| B-spline | 3×p 个 `(offset, value, slope)` | 216 B (p=6) | 已有 |
| CubeS₂ | 7 kind × 3 axis × 2 orientation × (v,d) | **336 B** | 4 项收缩；可压到 240 B（L211 = −L111/4 − 5L311/2） |
| QuadS | 3×P 个 `(v, d, offset)` | **216 B** (P=6) | 严格可分离；与 B-spline 同构 |

CubeS₂ 的 336 B 比 B-spline 大，所以 **CubeS₂ 不建全局表**：
在 gather 的 block 内用 shared 填表（8 atom/block → 2.7 KB），
填充成本相对于它替代掉的 88×多项式可以忽略。spread 保持现有的
`Cs2Tab`/`Cs2Tab4`（6 lane/atom，60 floats）不变——它已经很快，
且不在本文的性能瓶颈上。QuadS 的 216 B 与 B-spline 相同，可以建全局表
（一次写、两次读），用 +216 B/atom 换掉 spread/gather 里的两遍填充——
这正是它 49→17 ns 的关键。

### 3.2 CubeS₂ 的"≤4 项一维之积"

`verify_cs2_terms.py` 已逐节点验证（值与三个导数，order 4/6，随机 500 组）：

```
order 4: 32 nodes, classes {0:8, 1:24},        terms {1:24, 3:8}
order 6: 88 nodes, classes {0:8,1:24,2:24,3:24,4:8}, terms {1:56, 2:24, 4:8}
max |value/gradient difference| = 1.11e-16
PASS
```

即：**88 个节点里 56 个只有 1 项、24 个 2 项、8 个 4 项；平均 1.55 项。**
一个统一的"最多 4 项"循环就能覆盖全部节点，不需要 class 分支：

```cuda
// 每个节点：uint4 T（4 个 term），未用项 kind=0（值为 0、导数为 0）
float w = 0.f, gx = 0.f, gy = 0.f, gz = 0.f;
#pragma unroll
for (int t = 0; t < 4; ++t) {
  const unsigned word = (t == 0) ? T.x : (t == 1) ? T.y : (t == 2) ? T.z : T.w;
  const int ix = word & 0xff, iy = (word >> 8) & 0xff, iz = (word >> 16) & 0xff;
  const float ax = B.v[ix&7][0][(ix>>3)&1];
  const float ay = B.v[iy&7][1][(iy>>3)&1];
  const float az = B.v[iz&7][2][(iz>>3)&1];
  w  += ax*ay*az;
  gx += B.d[ix&7][0][(ix>>3)&1]*ay*az;
  gy += ax*B.d[iy&7][1][(iy>>3)&1]*az;
  gz += ax*ay*B.d[iz&7][2][(iz>>3)&1];
}
```

`T` 是每节点一个 `uint4`（4 个 term，每个 term 3 个轴的 `[kind|e]` 字节），
`B` 是每原子的 `Cs2Basis`。
`kind 0` 的 `v/d` 全 0，所以未用项自然贡献 0，不需要 `if`。

### 3.3 节点表：offset 排序，恢复合并访存

- 用 `c_nodes6s`（按 `dx + nx*(dy + ny*dz)` 排序，x 最快），不要 `c_nodes6c`。
- term 表按**同一顺序**生成；`sh_off[k]` 仍可预算 `dx + nx*(dy+ny*dz)`。
- 一轮 32 个 lane 覆盖相邻 `(x,y,z)`；整个 88 节点模板覆盖 **32 个 32 B sector**，
  `c_nodes6s` 实测 **36 sector/atom**（`c_nodes6c` 是 53），每轮约 11–13 个
  128 B line。**收益 1.47×，不是 4–8×**——这是 CUDA 审查纠正过的数字。
- 对边界原子仍走 `midx(wrap_near(...))` 的慢路径；内部原子用 `base + sh_off[k]`。

### 3.4 green：compact 半谱 + 按 shape 缓存

replicated 路径直接复用 slab 的 `build_green_slab_half`（`sog_green.h:177-315`）：

- 布局 `idx = kh + Nh*(iy + ny*iz)`，`Nh = nx/2+1`，与 `f_spec` 完全一致；
- `f_gp` 从 `8 B/pt` 降到 `4 B/pt`；
- `ge`/`gf` 只留一份（两者恒等），host 从 24 B/pt 降到 16 B/pt；
- Form A 的 window 表也按这个布局存，`k_gp_divide_form_a` 的 `gi` 改成 compact。

**Form A 的 window 是 shape-only 的。** 从 `k_gp_divide_form_a`
（`sog_gpu.cu:3322-3349`）：

```
D = exp(xi² (dx²kx² + dy²ky² + dz²kz²))
  = exp(xi² ((2π kh/nx)² + (2π my/ny)² + (2π mz/nz)²))
```

`dx*kx = (lx/nx)(2π kh/lx) = 2π kh/nx`，与 `lx` 无关；测量的 `w2` 是离散
FFT 的模方，也与 `lx` 无关。**所以 window 只依赖 `(nx,ny,nz,order,nth)`。**
把它从"每次 cell 漂移就重测"改成"按 shape 缓存一次"，NPT 下固定网格时
**零重测**，网格变化时也只测一次。

### 3.5 一次性 green 构建移到 device

现在 `build_green_range` 在 host 上对 `M × G` 个模式做 `exp` + 3 表填充
（`sog_green.h:142-170`），再 `cudaMemcpy` 到 device。改为：

1. host 只建**每轴**表（`Bx[M][nx], By[M][ny], Bz[M][nz]` + 去除因子），
   大小 `O(M(nx+ny+nz))`，144³ 时 < 100 KB；
2. device kernel 对每个 mode 组合，写 compact `f_gp`（CubeS₂ Form A）
   或直接给 k-space 用（可分离家族甚至可以不建表）。

对 CubeS₂ Form A（window 不可分离）仍需一张 4 B/pt 的 3D 表；但
**它只依赖 shape**，可以在 shape 首次出现时建一次，之后 NPT 只重建
每轴表 + 一个轻量的 mode 组合。对 QuadS/B-spline，green 完全可分离，
可以彻底不建 3D 表。

---

## 4. 优化项（按依赖与收益排序）

> **优先级（2026-11-13 复核后）**：CubeS₂ 的 mesh 已经比 B-spline 快 25%，
> 所以先做 **P1（CubeS₂ 4 项 gather）** 和 **P3/P4（显存/临时量）**；
> **P2（QuadS）降级为"待复核的微优化"**——修复后 QuadS 仍慢于 CubeS₂，
> 只有 A/B 显示 operator replay >10% 收益时才做。

### P0 — 度量与验证（本次已落地，0.5 d）
**内容**
- `docs/repro/verify_cs2_terms.py`：4 项分解的值+梯度验证，纳入 CI。
- `docs/repro/mesh_memory_ledger.py`：buffer 账本，按 `N/G/order` 出表。
- 修正 benchmark：process-scoped memory、20 ms 采样、steady/peak 分开、
  `SOG_MEM_BREAKDOWN=1`，首次调用单独列。

**验收**：同一进程重复 5 次，steady 波动 < 3%；peak 与 `SOG_MEM_BREAKDOWN`
的账本和 < 15%；首次瞬态单独报告。

### P1 — CubeS₂ 无分支 4 项 gather（结构对齐，收益温和，2–3 d）
> **2026-11-13 已落地（replicated AD，order 6）**：`k_gather_cubes2_term<6>`
> + `Cs2Basis` + `cs2gen::build_and_check` 自检；`SOG_TERM_GATHER=0` 可回退。
> op graph replay：mid4ad 0.4058→0.3863（−4.8%）、mid6ad 0.3780→0.3569
> （−5.6%）、mid6@108 0.3843→0.3636（−5.4%）。全步 3 次中位数
> CubeS₂-6 1.207/4.933 ms（旧 1.225/4.999）；CubeS₂-4 全步未见收益，
> **默认只对 6 阶启用**。brick 尚未移植，slab/replicated AD 差 ~1e-6。
> 详见 [`MIDTOWN_ALL_CONFIGS.md`](MIDTOWN_ALL_CONFIGS.md) §5。
**现状**：`k_gather_cubes2_warp_r` 每节点 `cs2_wg6`，class 排序；1737 instr/atom。
**改法**：
1. `sog_gpu_create` 里按 `c_nodes6s` 的顺序生成 `__constant__ uint4
   c_cs2_terms[2][96]`：每节点 4 个 term，每个 term 打包 3 个 8-bit
   `[kind|e]`；节点 88–95 与未用 term 全 0。用 `verify_cs2_terms.py`
   同一映射，并加三条生成器单测（见 §5.1）。
2. 新增 `Cs2Basis`（`float v[7][3][2]; float d[7][3][2]`，84 floats）+
   `__constant__ float c_poly[7][6]` / `c_dpoly[7][6]`（kind 0 全 0）。
3. gather 每个 block 先：
   - 把 88 个 node/offset 和 `uint4 sh_T[96]`（1536 B）载入 shared；
   - 8 个 atom 的 `Cs2Basis` 由 256 线程块内填（8×42=336 slot，2 轮），
     `__syncthreads()`；
4. 每 warp 一个 atom，读 `sh_T[kk]`（1 次 LDS.128）后用 4 项收缩；
   节点 88–95 是零 term，padding lane 不需要 `break`/谓词。
5. 内部原子 `base + sh_off[k]`；边界原子 `midx(wrap_near(...))`。
6. **同一套收缩必须同时移植到 `k_gather_cubes2_brick`（`sog_gpu.cu:3936/3991`）**，
   否则 replicated 与 slab 的力会差 ~1e-7，Tier-3 的"全零"验收直接失效。
7. 加 `SOG_TERM_GATHER=0/1` 开关；当前默认 6 阶开、4 阶关，A/B 通过；
   brick 未移植，Tier-3 的 replicated/slab 逐位一致需后续补。

**预期（复核后口径）**：gather −25~40%；operator mesh **0.385 → 0.30–0.33 ms**
（108³，−13~20%）；LR 98 k **1.23 → 1.18–1.21**、331 k **4.94 → 4.75–4.85**；
526 k 待重测。**不要按 3.6× 指令数线性外推**——gather 只占 mesh 的一半，
mesh 又只占每步 0.4/9.3 ms。小体系（12 k）因 occupancy 4→6 block，
收益更明显。

**风险**：
- 4 项关联顺序与 `cs2_wg6` 不同，权重差 ~1e-7；端到端力 < 1e-6（Tier-2 口径）。
- **P1b 不能做到逐位一致**：spread 的 `L211 = -0.25*L111 - 2.5*L311` 是在
  double 里用已舍入的 fp32 `L111/L311` 算的（`sog_gpu.cu:876-877`），
  而 basis 把它折进 fp32 常数；两者仍差 ~1e-7。要逐位就得让 basis 复用
  spread 的 double 组合，成本不低。
- **brick 路径**不改则 slab/replicated 不再逐位一致（见上）。
**验收**：同一网格上 `max|ΔF| ≤ 1e-5`、`rel rms ≤ 1e-6`；`pe` 不变；
`test_mesh_backend.py` / `test_long_range_sum.py` 全绿；`mp_parity.py`
在**brick 也改完**后全零；CUDA graph 由 AOTI 侧验证（`sog_gpu.cu` 本身
没有显式 graph 代码）。

**P1b（可选，0.5 d）**：spread 也用同一套 4 项表，让 spread/gather 的
权重来自同一项序。**目标是"同一代数路线"，不是逐位**；性能变化不大，
正确性上是加分项。

### P2 — QuadS 稠密 stencil pass（待复核的微优化，2 d）
**现状**：`k_spread_quads_warp_r` 和 `k_gather_quads_warp_r` 各填一次
`QuadsTab`。
**改法**：
1. 新增 `k_quads_stencils`：1 thread/atom，算 `FracNode` + `QuadsTab`
   （v[3][P], d[3][P], off[3][P]，216 B）写入全局；
   **必须调用与现 kernel 完全相同的 `quads_fill_tab`（同一 fp32 多项式、
   同一 `quads_w1d`）**，只改存储位置，算术逐位不变。
2. spread/gather 改为从全局表读（或 warp 内 `load_stencil` 到 shared，
   与 B-spline 的 `load_stencil` 同构）；
3. brick / slab 路径的表依赖 clip，保持现状；因为它们用同一个
   `quads_fill_tab_values`，值逐位相同，replicated vs slab 仍可全零。
**预期（复核后）**：operator replay **0.405 → 0.35–0.38 ms（−5~15%）**；
即使做到上限，也只是接近 CubeS₂ 的 0.377 ms，不会反超。旧 49 ns /
27.1 ms 是 pre-fix，不再作为 P2 的目标。
**风险**：+216 B/atom 设备内存（与 B-spline 相同）；网格更小时总账仍赢。
**验收**：与现 QuadS 逐位一致（只改表的存储位置，不改算术）。

### P3 — compact green + 去掉 `ge/gf` 重复（1 d）
**改法**：
1. `f_gp` 改为 `Nh*ny*nz` 个 `float2`（4 B/pt）；`k_kspace_h` 的
   `gp` stride 从 `nx` 改 `nh`；
2. `build_green_range` / `build_green` 改为调用 `build_green_slab_half`
   （`ylo=0, YL=ny`），只填 `ge,gv`；`setup_replicated` 只上传两份；
3. `k_gp_divide_form_a` 的 `gi` 改 compact；Form A 的 `acc` 仍按 compact 半谱；
4. 增加 `SOG_MIDTOWN_GP_COMPACT=0` 回退开关（一个版本周期后删除）。
**预期**：331 k / 144³ 上设备 −11.4 MiB；526 k / 384³ 上 **−226 MiB**；
host green 从 24 B/pt 降到 16 B/pt，首次构建时间 −1/3。
**风险**：任何仍按 full-`nx` 索引 `f_gp` 的路径（Form A、slab、LAMMPS）
必须一起改；用 `SOG_MIDTOWN_DEBUG` 打印 stride，加断言 `gp_index < Nh*ny*nz`。
**验收**：与现实现逐位一致（同一张表的值，只改布局）。

**顺带清理（0 MiB 现状，但防止未来误分配）**：`ensure_quads_mesh` /
`ensure_quads_ik`（`sog_gpu.cu:3495-3511`）在本算子路径上**没有被任何代码调用**
（grep 只有定义）；`f_rho/f_pot`（16 B/pt）和 `f_qgx/y/z/qgspec`（32 B/pt）
是旧 full-complex QuadS 路径的遗留。确认该路径退役后删掉这两个函数和字段——
一旦有人重新接上，144³ 会多 136 MiB（48 B/pt），正好把 compact green 的收益吃回去。

### P4 — float 力/势 + 单 workspace + finish 合并（1–2 d）
**改法**：
1. `sog_gpu_compute_dev_defer` 的 `d_force`/`d_vpot` 改为 float；
   gather 写 float；`midtown_sum.cu` 删除 `copy_(f3)` 与 `reshape`
   （正交盒直接是输出；tilted 用一个小 kernel 做 `F_s A^{-T}`）；
2. `k_midtown_finish1/finish2/virial` 合并为 2 个 kernel：
   finish1 写 `phi` 到 `charge_gradient` 缓冲，finish2 覆盖为 grad，
   并把帧 virial 的前 9 个线程并入 finish2；
3. 用 `Workspace`（`frame_sums.cuh:25-50`）一次分配 q/csums/d_force/d_vpot/
   sums；输出仍是独立 tensor；只对 `virial` 做一次 72 B memset。
**预期**：331 k 上 −10.6 MiB（double→float）+ 少 3 个 launch；
小体系（12 k）host/launch 开销 −10–15%。
**风险**：力在 fp32 上提前舍入，virial 矩的 fp64 累加仍从 float 力算，
与 B-spline `finish_owners` 一致；Tier-2 允许 1e-6。
**验收**：`equiv_dump.py` 前后 `pe` 一致、力 rel rms < 1e-6。

### P5 — MidtownPlans + Form A 按 shape 缓存 + green 在 device 构建（2–3 d）
**改法**：
1. 把 `MidtownState` 的 grid/plan/green 拆成：
   - `MidtownPlans`：按 `(device,nx,ny,nz,spline,order,form)` 缓存
     `cufftPlan3d` / work area / compact green / Form A window；
     LRU + `cudaGraphRetainUserObject`（照抄 `MeshPlans`，`mesh_sum.cu:1040-1135`）；
   - `cell`/`bands`：green 的物理部分，cell 漂移 >1e-3 才重建；
2. Form A window 只按 shape 测一次；用 `nth=8` 的并行 device 版本或
   host 侧一次算好（见 §3.4/§3.5）；
3. green 每轴表 host 建、3D 组合 device 做；
4. 删除 `sog_gpu_green_form_a` 里 512 次 host 同步循环。
**预期**：NPT 下 green 重建从"每 cell 漂移一次 + 512 次同步"降到
"每 shape 一次 + 每 cell 一次轻量组合"；**同一个 shape 的后续 cell 重建
< 50 ms**（首次 shape 的 window 测量仍需一次，目标 < 0.5 s）；plan 重建
从"每次网格变化"降到"shape 首次出现"。
**风险**：graph hold 的生命周期；window 的 shape-only 假设必须用
两个不同 `L`、同 shape 的单元测（`max|Δwindow| < 1e-6`）。

### P6 — ik 路径（条件项，1–2 d）
**决策**：CubeS₂-6 ik 默认退役（49 vs 18 ns，且非变分）；CubeS₂-4 ik
仅在需要 4 阶力精度时保留。若保留：
1. `pot/gx/gy/gz` 交织成 `float4`（一次 16 B 读，替代 4 路 4 B 散读）；
2. `k_mul_ik` 与 green 乘法融合，避免 3 次重读半谱；
3. 三张梯度网格可复用同一块 `float4` 网格（16 B/pt，而非 3×4+4）。
**预期**：ik gather 18.6 → 4–6 ms（526 k）；仅当 P2 决定保留 ik 时做。

### P7 — 网格规则/精度审计（物理层，单独决策）
- 复核 A 链 `grid_rule.spacing()` 对 midtown 的方差扣除上限在 Form A 下是否
  仍必要；若去掉，526 k 的 CubeS₂-6 可能从 384³ 回到 ~225³，这比任何 CUDA
  优化都更能省显存。
- QuadS 的力精度尚未做力量级审计（`GRID_FROM_ACCURACY.md` §5 已点名）；
  在把它作为"新样条主力"之前，必须补 `FORCE_RULE` 的 8 构型 × 2 核测试。
- 这两项会改变物理结果，**不要和 P1–P6 混在一个 PR**。

---

## 5. 代码级设计

### 5.1 静态表

```cuda
// kind: 0 zero, 1 E(t), 2 L111/L, 3 L211, 4 L311, 5 S6, 6 R
// 每个轴一个 byte: 低 3 位 kind，高 1 位 e（0 -> f(t), 1 -> f(1-t)）
// 每节点 4 个 term，每个 term 的 3 个轴各 1 字节 [kind|e]；
// 4 个 term 放进 uint4 的 x/y/z/w，节点 88..95 与未用 term 全 0
__constant__ uint4 c_cs2_terms[2][96];   // 2*96*16 = 3072 B

struct Cs2Basis {
  float v[7][3][2];
  float d[7][3][2];   // d f / d t，反射的 -1 已并入 e
};                    // 84 floats = 336 B
```

host 侧生成（伪代码）：

```cpp
for (order, node in offset_sorted_nodes) {          // 0..87
  auto terms = cs2_terms(node, order);              // 与 verify_cs2_terms.py 同一映射
  uint4 packed{0, 0, 0, 0};
  unsigned* word = &packed.x;                       // x/y/z/w 各一个 term
  for (t = 0; t < 4; ++t) {
    unsigned w = 0;
    for (axis = 0; axis < 3; ++axis) {
      const unsigned code = terms.size() > t
          ? (terms[t][axis].kind | (terms[t][axis].e << 3)) : 0u;
      w |= code << (8 * axis);
    }
    word[t] = w;
  }
  c_cs2_terms[order][node] = packed;
}
// 88..95 保持 {0,0,0,0}：padding lane 读到零 term，w/g 自然为 0
```

**必须用 `c_nodes6s` 的顺序**（offset 排序），不要用 `c_nodes6c`（class 排序）。
`kind=0` 的 `v/d` 全 0，所以未用项自然贡献 0，不需要 `if`。

**生成器必须加三条单测**（CUDA 审查实测：写错任一条，力会静默偏 ~5e-2）：

1. **cls 3 的 s1/s2 必须取排序后的 `min/max`**。`cubes2_weight_6` 里
   `s1,s2 = sort((nm+1)%3, (nm+2)%3)`；若直接用 `(nm+1, nm+2)`，
   `sp_axis==1` 的每个节点都会把 bit1/bit2 的符号装反
   （例如 node 59 `(2,0,-1,cls3,sp=1,spn=2)`、node 74）。
2. **cls 3 的 nm 轴用 offset 选 `t/1-t`，s1/s2 轴用 `spn` 的 bit 选反**；
   两者不能混用 `e_of()`。d=+2 的节点尤其容易错。
3. **未用 slot 必须是零基**，不能用常数 1；用 1 会每个节点多加 3.0。

**代价**：固定 4 项对平均 1.55 项来说做了 ~2.6× 无用功。这是为"无分支 +
无逐节点表解码"付的；如果后续还要压指令，可以按 term 数（1/2/4）把节点分组，
做 warp-uniform 的三段代码——但那会把 class 排序的复杂度重新引进来，**先不做**。

### 5.2 多项式系数

`c_poly[7][6]` / `c_dpoly[7][6]` 由 `kXi4` / `kXi6` 在 `sog_gpu_create` 时
算好并 `cudaMemcpyToSymbol`。kind 0 全 0；kind 1 是 `[0,1]`；kind 2/3/4/5
是 5 次；kind 6 是 3 次。填表时统一按 6 次 Horner（缺项系数 0），
每个 slot 一次 value + 一次 derivative：

```cuda
__device__ __forceinline__ void fill_basis(Cs2Basis& B, int kind, int axis,
                                           int e, float t) {
  const float x = e ? 1.f - t : t;
  const float* c  = c_poly[kind];
  const float* dc = c_dpoly[kind];
  float v = c[5], d = dc[5];
  #pragma unroll
  for (int k = 4; k >= 0; --k) { v = fmaf(v, x, c[k]); d = fmaf(d, x, dc[k]); }
  const float s = e ? -1.f : 1.f;
  B.v[kind][axis][e] = v;
  B.d[kind][axis][e] = d * s;
}
```

block 内 8 个 atom × 42 slot = 336 个 `(atom, kind, axis, e)`，256 线程 2 轮：

```cuda
for (int s = threadIdx.x; s < 8 * 42; s += blockDim.x) {
  const int a = s / 42, r = s - a * 42;
  const int kind = r / 6, q = r - kind * 6;   // q = axis*2 + e
  fill_basis(sh_basis[a], kind, q >> 1, q & 1, sh_t[a][q >> 1]);
}
__syncthreads();
```

### 5.3 gather（替换 `k_gather_cubes2_warp_r` 的收缩段）

```cuda
template <int ORD>
__global__ void __launch_bounds__(8 * 32)
k_gather_cubes2_term(int nlocal, const FracNode* fr, const double* q,
                     double qscale, double ilx, double ily, double ilz,
                     int nx, int ny, int nz, const float* pot,
                     double* force, double* vpot) {
  constexpr int NN = (ORD == 4) ? 32 : 88;
  __shared__ GpuNode      sh_nodes[88];
  __shared__ long long    sh_off[88];
  __shared__ uint4        sh_T[96];          // 1 次 LDS.128 / node；88..95 为零
  __shared__ Cs2Basis     sh_basis[8];
  __shared__ float        sh_t[8][3];
  __shared__ int          sh_i[8][3];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int i = blockIdx.x * 8 + warp;

  for (int t = threadIdx.x; t < NN; t += blockDim.x) {
    sh_nodes[t] = (ORD == 4) ? c_nodes4s[t] : c_nodes6s[t];
    sh_off[t]  = (long long)sh_nodes[t].dx +
                 (long long)nx * (sh_nodes[t].dy + (long long)ny * sh_nodes[t].dz);
  }
  for (int n = threadIdx.x; n < 96; n += blockDim.x)
    sh_T[n] = c_cs2_terms[ORD == 4 ? 0 : 1][n];

  // 8 个原子的 frac 先由 lane 0..2 写入 shared，再块内填表
  if (lane < 3) {
    const int a = warp;
    sh_t[a][lane] = (i < nlocal) ? fr[i].t[lane] : 0.f;
    sh_i[a][lane] = (i < nlocal) ? fr[i].i[lane] : 0;
  }
  __syncthreads();
  for (int s = threadIdx.x; s < 8 * 42; s += blockDim.x) { /* 5.2 */ }
  __syncthreads();

  if (i >= nlocal) return;
  const int ix0 = sh_i[warp][0], iy0 = sh_i[warp][1], iz0 = sh_i[warp][2];
  const bool interior = ix0 >= 2 && ix0 + 3 < nx && iy0 >= 2 && iy0 + 3 < ny
                     && iz0 >= 2 && iz0 + 3 < nz;
  const long long base = interior
      ? (long long)ix0 + (long long)nx * (iy0 + (long long)ny * iz0) : 0;

  const Cs2Basis& B = sh_basis[warp];
  float dtx = 0.f, dty = 0.f, dtz = 0.f, gp = 0.f;
  #pragma unroll
  for (int round = 0; round < 3; ++round) {
    const int k = lane + 32 * round;
    const bool live = (k < NN);
    const int kk = live ? k : NN - 1;
    const GpuNode& nd = sh_nodes[kk];
    const long long m = interior
        ? base + sh_off[kk]
        : (long long)midx(wrap_near(ix0 + nd.dx, nx),
                          wrap_near(iy0 + nd.dy, ny),
                          wrap_near(iz0 + nd.dz, nz), nx, ny);
    const float u = live ? pot[m] : 0.f;

    const uint4 T = sh_T[kk];
    float w = 0.f, gx = 0.f, gy = 0.f, gz = 0.f;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      const unsigned word = (t == 0) ? T.x : (t == 1) ? T.y : (t == 2) ? T.z : T.w;
      const int ix = word & 0xff, iy = (word >> 8) & 0xff, iz = (word >> 16) & 0xff;
      const float ax = B.v[ix & 7][0][(ix >> 3) & 1];
      const float ay = B.v[iy & 7][1][(iy >> 3) & 1];
      const float az = B.v[iz & 7][2][(iz >> 3) & 1];
      w  += ax * ay * az;
      gx += B.d[ix & 7][0][(ix >> 3) & 1] * ay * az;
      gy += ax * B.d[iy & 7][1][(iy >> 3) & 1] * az;
      gz += ax * ay * B.d[iz & 7][2][(iz >> 3) & 1];
    }
    gp += w * u; dtx += gx * u; dty += gy * u; dtz += gz * u;
  }
  // warp 归约同现状
}
```

> `sh_T[kk]` 是 1 次 LDS.128：32 个 lane 读 32 个连续 `uint4`，地址 stride
> 16 B，按 128-bit 访问分 4 个 phase，phase 内 bank 连续，无冲突。
> 若按 `[term][node]` 存 4 个 `uint32`，每个 lane 要 4 次 LDS.32，
> 指令数多 3 条/节点；若按 `[node][term]` 存，lane 的 stride 是 4 word，
> 32 个 lane 只落到 8 个 bank，形成 4-way 冲突。

### 5.4 spread

**CubeS₂**：保持现有 `k_spread_cubes2_warp_fx_packed` 不变（它已经把表
摊到 4 atom/warp，26 µs @12 k，是整套里最成熟的部分）。
若要 P1b 的逐位一致，再把它换成同一 4 项表；但要重新 A/B。

**QuadS**：新增 `k_quads_stencils`，把 `QuadsTab` 写成全局 216 B/atom；
spread/gather 用 `load_stencil` 式拷贝到 shared。核心循环不变。

### 5.5 compact green 的索引改动

```cuda
// 现在（full nx）
const size_t gstride = (size_t)gridDim.y * nx * ny;
const float2* gp0 = gp + (size_t)nx * (iy + (size_t)ny * blockIdx.y) + khc0;

// 目标（compact half spectrum）
const int nh = nx / 2 + 1;
const size_t gstride = (size_t)gridDim.y * nh * ny;
const float2* gp0 = gp + (size_t)nh * (iy + (size_t)ny * blockIdx.y) + khc0;
```

`k_gp_divide_form_a` 的 `gi` 同理改成 `kh + nh*(iy + ny*iz)`；
`setup_replicated` 的 `cudaMemcpy(f_gp, fgp, gn*sizeof(float2))` 的 `gn`
改为 `nh*ny*nz`。

### 5.6 host/workspace

```cpp
Workspace ws;
const int64_t q_at      = ws.reserve<double>(count);
const int64_t csum_at   = ws.reserve<double>(2);
const int64_t vpot_at   = ws.reserve<double>(count);
const int64_t force_at  = ws.reserve<float>(3 * count);   // P4
const int64_t sums_at   = ws.reserve<double>(10);
ws.allocate(coord.device());
// 输出 tensor 仍是 5 个；virial 只 memset 这一次
```

B-spline 的 `Workspace` 已有 256 B 对齐和单次 `torch::empty`，直接复用。

---

## 6. 预期收益（等精度、replicated、AD）

> **2026-11-13 更正**：§6.2 的"现在"列是**修复前**的 526 k 数据
> （SPLINE_MATRIX，graph 捕获失败时测的），不能当作修复后的基线。
> §0.1 已在 98 k / 331 k 实测到 CubeS₂-6 AD **LR −5%、显存 −4~5%**；
> 526 k 需按同样口径重测。下表的目标是"mesh 内部再降 13–20%，LR 折 1–3%"，
> 与修复 bug 的 −5% 不重复。

### 6.1 每原子赋值 / operator mesh

| 配置 | pre-fix | 修复后（graph replay） | P1/P2 目标 |
|---|---:|---:|---:|
| B-spline-6 @108³ | 17 ns | **0.517 ms** | — |
| CubeS₂-6 @96³ / 108³ | 18 ns | **0.377 / 0.385 ms** | **0.30–0.33 ms**（gather −25~40%） |
| QuadS-6 @81³ / 108³ | 49 ns（失效） | **0.405 / 0.413 ms** | 0.35–0.38 ms（仅微优化） |
| CubeS₂-6 ik | 49 ns（失效） | 退役；若要留，见 P6 | — |

### 6.2 LR 时间（ms/step，实测）

| 体系 | 配置 | 网格 | pre-fix | **修复后** | P1 后目标 |
|---|---|---:|---:|---:|---:|
| 98 k | B-spline-6 | 108³ | 1.40 | 1.30 | 1.30 |
| 98 k | CubeS₂-6 | 96³ | 1.40 | **1.23** | 1.18–1.21 |
| 331 k | B-spline-6 | 160³ | 5.30 | 5.21 | 5.21 |
| 331 k | CubeS₂-6 | 144³ | 4.99 | **4.94** | 4.75–4.85 |
| 526 k | B-spline-6 | 180³ | — | 8.84 | 8.84 |
| 526 k | CubeS₂-6 | 162³ | — | **8.46** | 8.10–8.25（P1 后） |

> **怎么读**：CubeS₂ 的 **mesh** 已经 −25%，但整步/LR 只 −5%；P1 的
> 4 项收缩是"mesh 再降 13–20%"，折到 LR 只剩 **−1~3%**。QuadS 不再承诺
> 2.5–2.9×（pre-fix 49 ns 已失效）。这就是 Amdahl：mesh 只占 0.4/9.3 ms。
>
> 这些是**算子级 LR**，不是整步。整步收益 = LR 收益 × LR 占比。
> 98 k / 9.3 ms 步长时，CubeS₂ 的 1.23 vs 1.30 约等于整步 **−0.7%**；
> 331 k / 31.5 ms 时约 **−0.5%**。要拿整步数字，按
> `MESH_ENGINE_TEST_REPRO.md` §3.4 的口径重测。

### 6.3 设备 buffer（不含 cuFFT work area）

| N | 网格 | B-spline | midtown 现在 | midtown 目标 | 目标/基准 |
|---:|---|---:|---:|---:|---:|
| 98 304 | 96³ / 108³ | 45.8 MiB | 28.5 | **23.6** | 0.52× |
| 331 776 | 144³ / 160³ | 152.4 MiB | 96.2 | **79.7** | 0.52× |
| 526 848 | 384³ / 250³ | 381 MiB | 1142 | **918** | 2.4×（网格更大，赢不了） |

最后一行说明：**当等精度规则给出更大的网格时，任何实现都救不了显存。**
这一行的问题在 R7，不在 CUDA。

### 6.4 首次调用

| 项 | 现在（144³） | 目标 |
|---|---:|---:|
| host green ge/gf/gv + fgp | 32 B/pt ≈ 91 MiB（384³ ≈ 1.9 GiB） | 每轴表 < 1 MiB |
| device f_gp（full）+ Form A `acc` | 22.8 + 11.4 ≈ 34 MiB | compact 11.4 + acc 11.4 ≈ 23 MiB peak |
| Form A 同步迭代 | 512 次 spread+FFT（0.6 s @144³，12 s @384³） | 每 shape 一次，device 并行 |
| 首次 wall | 0.6 s（144³）/ 12 s（384³） | 首个 shape < 0.5 s；cell 重建 < 50 ms |

---

## 7. 验收

### 7.1 功能/数值
0. **先确认 graph 真的捕获了 mesh**：直接 capture `latent_charge_sum`，
   `replay` 时间应 ≈ eager 的 mesh 时间（0.3–0.5 ms），不是 µs；改坐标再
   replay，`max|ΔF(graph−eager)|` 应为 0。修复前 mid6ad 的 replay 是 5.5 µs
   （空图），这个检查能一眼发现。
1. `verify_cs2_terms.py` PASS（CI，无 GPU）。
2. Tier-1：
   ```bash
   $PY -m pytest source/tests/common/dpmodel/test_grid_rule.py -q
   CUDA_VISIBLE_DEVICES=3 $PY -m pytest source/tests/pt_expt/kernels/test_mesh_backend.py -q
   ```
3. Tier-2：`equiv_dump.py` 前后，**未触碰路径 max|dF|=0**；
   触碰路径 `rel rms ≤ 1e-6`、`max|dF| ≤ 1e-5`、`pe` 不变。
4. Tier-3：`mp_parity.py 12288` 全零（定点 spread 的顺序无关性不能被破坏）。
5. A/B 开关：`SOG_MIDTOWN_TERM_GATHER=0/1`、`SOG_MIDTOWN_GP_COMPACT=0/1`；
   同一输入两种路径的 `max|ΔF|` 记录在 PR 里。

### 7.2 性能
```bash
# 同网格实现对比（108³）
TIMING=1 MEAS=110 $PY docs/repro/perf_iso_grid.py
# 等精度、全尺寸
$PY docs/repro/perf_dist.py 98304
# 逐 kernel
SOG_GPU_TIMING=1 ...   # 两边同名 phase
```
**门槛（复核后）**：
- CubeS₂-6 AD：gather −25~40%，operator mesh **−13~20%**，LR 在修复后
  1.23/4.94 ms 的基础上再 **−1~3%**；不得让任一尺寸变慢 > 2%。
- QuadS-6 AD（若做 P2）：operator replay **−5~15%**；不再要求 −40%。
- 修复 stream/capture 本身已经给出 LR −5%，不要把它算进 P1 的 KPI。

### 7.3 显存（必须用新口径）
```bash
SOG_MEM_BREAKDOWN=1 ...
# 20 ms，process-scoped，steady + peak 分开
```
**门槛（复核后）**：steady **不高于 B-spline（≤1.0×）**，目标 0.9×；
静态 buffer 仍按账本的 0.52× 验收；peak 与账本之和差 < 15%；
首次 shape 瞬态单独报告且 < 0.5 s，之后 cell 重建 < 50 ms。
（修复后实测 331 k：midtown +1102 vs B-spline +1152 MiB，约 0.96×。）

### 7.4 回归
`test_long_range_sum.py`（50）、`test_mesh_backend.py`（4）、
`common/dpmodel/test_long_range.py`（61）、`test_grid_rule.py`（20）全绿；
`.pt2` 冻结产物不因内核重构而需要重冻（算子 schema 不变）。

---

## 8. 排期与风险

| 步 | 内容 | 估时 | 依赖 |
|---|---|---|---|
| P0 | 度量 + 验证脚本 | 0.5 d | 无 |
| **P0.5** | **pinned coefficient + 当前 stream（§0.1）** | **已完成** | 无 |
| P1 | CubeS₂ 4 项 gather（**先做**） | 2–3 d | P0, P0.5 |
| P2 | QuadS stencil pass（**待复核**） | 2 d | P1 A/B |
| P1b | spread 同步统一（可选） | 0.5 d | P1 |
| P3 | compact green | 1 d | P0 |
| P4 | float 力/势 + workspace | 1–2 d | P3 |
| P5 | MidtownPlans + Form A shape 缓存 | 2–3 d | P3 |
| P6 | ik float4（条件） | 1–2 d | 独立决策 |
| P7 | 网格规则/QuadS 力精度审计 | 单独 | 物理 |

**主要风险**

1. **数值关联顺序**：4 项收缩与 `cs2_wg6` 的乘法顺序不同，权重差 ~1e-7；
   端到端在 Tier-2 阈值内，但不是逐位。**P1b 也不能做到逐位**（spread 的
   L211 走 double 组合），要逐位必须让 basis 复用同一条 double 路线。
2. **shared 占用**：8 atom × 336 B + 1536 B `sh_T` + 2112 B node + 704 B offset
   ≈ 7.1 KB/block；256 线程/block 下 occupancy 不受限，但要用
   `__launch_bounds__(256)` 钉住。
3. **term 表越界**：`k >= NN` 的 padding lane 必须读 `sh_T[kk]` 的零 entry
   （或复用 `NN-1` 且令 `u=0`），不能读 `sh_T[k]`；上面的 `kk` 写法已处理。
4. **brick 路径**：P1 只改 replicated 的 `k_gather_cubes2_warp_r` 会让
   replicated/slab 的力差 ~1e-7，`mp_parity.py` 不再是全零；必须同时改
   `k_gather_cubes2_brick`。
5. **不要按 3.6× 指令数外推**：coalescing 只有 1.47×，gather 只占 LR ~18%；
   按"gather −25~40%"估整体，不要按"gather 快 3.6×"估。
6. **compact green 的隐藏路径**：Form A、slab、LAMMPS 的 green 都按 full `nx`
   索引过；必须一起改，并在 debug 下打印 stride。
7. **Form A shape-only 假设**：必须用同 shape、不同 `L` 的两个盒子验证
   `max|Δwindow| < 1e-6`；tilted 盒子不适用（走 metric 路径）。
8. **ik 退役**：会破坏已冻结的 `.pt2` 的 force_mode 选择；只能标"实验性"，
   不能物理删除枚举。
9. **pinned coefficient mirror 的缓存键**：目前是 `data_ptr + numel`。冻结图的
   系数是常量，安全；若未来训练图在**同一个 CUDA buffer 原地改值**，
   mirror 会读到旧值——届时加 `_version` 检查，或强制走 CPU coefficient。

---

## 附录 A：4 项分解的验证输出

```text
$ $PY docs/repro/verify_cs2_terms.py
order 4: 32 nodes, classes {0: 8, 1: 24}, terms {1: 24, 3: 8}
order 6: 88 nodes, classes {0: 8, 1: 24, 2: 24, 3: 24, 4: 8},
         terms {1: 56, 2: 24, 4: 8}
max |value/gradient difference| = 1.110e-16
PASS: the four-term form reproduces the reference for every node
```

映射规则（与 `sog_spline.h` 的 `cubes2_weight_6` / `cubes2_weight_grad_6` 逐项对应）：

| class | 项数 | 项 |
|---|---:|---|
| 0 | 4 | `L111(ex)ey ez + L111(ey)ez ex + L111(ez)ex ey + S6(ex)S6(ey)S6(ez)` |
| 1 | 2 | `L211(es)η(n1)η(n2) + R(es)S6(η(n1))S6(η(n2))` |
| 2 | 1 | `L311(es)η(n1)η(n2)` |
| 3 | 1 | `S6(η(nm))R(es1)R(es2)` |
| 4 | 1 | `R(ex)R(ey)R(ez)` |

其中 `η(a) = t_a if offset_a == 1 else 1-t_a`；`es = 1-t_sp if sp_is_neg else t_sp`；
`e` 位编码"取 t 还是 1-t"，导数表已含反射的 −1。

## 附录 B：显存账本输出

```text
$ $PY docs/repro/mesh_memory_ledger.py --n 331776 --grid-mid 144 --grid-bs 160
...
bspline:                 152.4 MiB   (16 B/pt + 284 B/atom)
midtown-current:          96.2 MiB   (20 B/pt + 124 B/atom)
midtown-target:           79.7 MiB   (16 B/pt + 108 B/atom)
midtown / B-spline total: current 0.63, target 0.52
```

账本不含 cuFFT work area / plan twiddle / CUDA context / 模型。
这三项对两个引擎在同一网格上相同，比较时应单独列。

## 附录 C：相关文档

- [`MIDTOWN_PERF_PLAN.md`](MIDTOWN_PERF_PLAN.md)：M0 设备级五段拆分与
  P1 止损结论；本文的 §1.2 直接引用它的数据。
- [`MIDTOWN_AD_DESIGN.md`](MIDTOWN_AD_DESIGN.md)：4 项可分离性的设计与
  Python 验证；本文 P1 把它从"止损"重新捡起，因为目标从"整步 <3%"
  改成了"每原子赋值 −50%、LR −30%"。
- [`SPLINE_MATRIX_BENCH.md`](SPLINE_MATRIX_BENCH.md) §4.5：
  17/18/49 ns per atom 的来源。
- [`GRID_FROM_ACCURACY.md`](GRID_FROM_ACCURACY.md)：网格规则与
  CubeS₂ 方差扣除上限。
- [`MESH_ENGINE_TEST_REPRO.md`](MESH_ENGINE_TEST_REPRO.md)：Tier 1–4
  与"20 ms 采样"的原始约定。
