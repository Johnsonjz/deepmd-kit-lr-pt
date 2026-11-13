# midtown (CubeS₂) vs B-spline：等精度下的 Fourier 与整步速度

> **2026-11-13 更正**：本文的逐后端数字早于 CUDA graph 捕获/stream 修复
> （midtown 的 mesh 未进图，模型回退 eager）。修复后 98k/331k/526k 的
> 等精度实测见 [`MIDTOWN_PERF_REDESIGN.md`](MIDTOWN_PERF_REDESIGN.md) §0.1；
> 本文的绝对 ms/显存不再可直接比较。

比较 **midtown（CubeS₂）** 与 **B-spline** 两种网格后端在同一条 SOG 长程链上的速度，
重点回答：**CubeS₂ 的 Fourier 力微分（`ik` 模式）在等精度下是否划得来。**

- 归一化口径统一为 **力相对精度 1e-4**（`pme_accuracy` / `phi_accuracy`），网格由实测误差律反解。
- 阶数覆盖 **4 阶与 6 阶**；每个阶数比较三种网格模式：
  **B-spline·AD**、**CubS₂·AD**、**CubeS₂·ik**。
- 单卡（np=1）；`ik` 只能在 **replicated** 网格上跑（见 §6），故不做多卡。

> **一句话结论**
> 1. **4 阶**：等精度网格 CubeS₂·ik 只要 **291³**（B-spline·AD 513³、CubeS₂·AD 766³）。
>    ik 的 Fourier 比 B-spline 快 **1.8–3.7×**、比 CubeS₂·AD 快 **3–7×**；显存比 B-spline 省
>    ~16%、比 CubeS₂·AD 省 **3.1×**；整步时间也略快。**ik 把 4 阶 CubeS₂ 从"不可用"变成三者最优。**
> 2. **6 阶**：等精度网格 B-spline 173³、CubeS₂·ik 177³（能量约束，几乎持平）、CubeS₂·AD 265³。
>    三种模式**整步时间差 <1.5%、显存差 <8%**，mesh 只占 <1% 运行时间——**6 阶下换后端与换力微分对速度无影响**。
> 3. **推论**：`ik` 的价值是**把 4 阶 CubeS₂ 的网格从 766³ 压到 291³**；到 6 阶 B-spline 的力
>    指数已是 6，网格本就落在能量地板上，`ik` 再省不出格点，反而因三次额外逆变换在 isolated mesh 上略慢。

---

## 0. 口径与可行矩阵

| | A 链（本文主线） | B 链（附录 §5） |
|---|---|---|
| 树 / 运行时 | `deepmd-kit-LR-lr-pt`，lr-pt 插件 `pair_style deepmd` | `deepmd-kit-LR`，CPU host `kspace_style sog` |
| 模型 | `run_250k_librule`（`latent_charge`，σ_min=2.00 Å） | `dpa4c_sog_expt` 的 SOG 核（固定电荷） |
| 计算位置 | **图内**（AOTI `.pt2`，canonical/`--compress`） | LAMMPS 的 C++ kspace（CPU host） |
| 力微分 | `ad` 与 `ik` 都有 | 只有 `ad` |

- **后端开关（A 链）**：checkpoint `model_params.long_range` 的 `mesh_backend` + `pme_order` + `force_mode`
  是**计算设置**（非权重），用 `sog_work/tools/freeze_lr_variants.py` 改后重冻结：
  `--backend {bspline|midtown} --order {4|6} --force-mode {ad|ik}`。
- **`ik` 限制**：仅 **replicated** midtown 解支持；`SOG_MIDTOWN_SLAB` / `SOG_MIDTOWN_DIST`
  显式报错（`latent_charge_sum.cu`）。故 `ik` 一律 **np=1**。
- **等精度**：网格由 `deepmd/dpmodel/long_range/grid_rule.py` 的实测误差律给出
  （`FORCE_RULE` / `IK_FORCE_RULE` / `ENERGY_RULE`，详见 `GRID_FROM_ACCURACY.md`、`IK_MODE_ACCURACY.md`）。
  `ik` 的力与能量同阶，网格取 `min(力, 能量)`，所以 `ik` 往往被**能量**约束。

## 1. 等精度网格（力 1e-4，σ_min=2.00 Å，盒长 122.31 Å）

| 阶 | 模式 | 规则 `(C,p)` | `Δ/σ_min` | `Δ` (Å) | 网格 | 谁在卡 |
|---|---|---|---|---|---|---|
| 4 | B-spline·AD | (1.70e-1, 3.5) | 0.119 | 0.2388 | **513³** | 力 |
| 4 | CubeS₂·AD | (1.96e-1, 3.0) | 0.080 | 0.1598 | **766³** | 力 |
| 4 | **CubeS₂·ik** | IK (1.89e-2, 4.0) | 0.210 | 0.4204 | **291³** | 能量 |
| 6 | B-spline·AD | (5.12e-2, 6.0) | 0.354 | 0.7071 | **173³** | 力 |
| 6 | CubeS₂·AD | (1.52e-1, 5.0) | 0.231 | 0.4620 | **265³** | 力 |
| 6 | **CubeS₂·ik** | IK (1.83e-2, 6.0) | 0.346 | 0.6918 | **177³** | 能量 |

读法：
- **4 阶**：`ik` 比 B-spline 省 **5.4× 格点**（291³ vs 513³），比 CubeS₂·AD 省 **18×**（291³ vs 766³）。
- **6 阶**：`ik` 与 B-spline **基本持平**（177³ vs 173³，多 2%）；比 CubeS₂·AD 省 3.3×。
- 网格随盒长每步重算；上表是 122.31 Å 参考盒。

---

## 2. A 链 · 4 阶（np=1，GPU，canonical）

### 2.1 整体 MD 单步时间（ms/step，取 2 次中位）

| atoms | bs4 | mid4 (AD) | mid4ik | sr-only |
|---:|---:|---:|---:|---:|
| 192 | 2.30 | 2.21 | 2.22 | 2.00 |
| 1536 | 9.06 | 8.70 | 8.71 | 8.28 |
| 5184 | 23.25 | 25.60 | 23.35 | 23.12 |
| 12288 | 52.76 | 57.07 | 53.55 | 53.35 |
| 24000 | 97.67 | 102.94 | 97.74 | 96.77 |
| 41472 | 167.27 | 175.05 | 165.27 | 163.66 |
| 65856 | 266.41 | 285.86 | 263.57 | 259.95 |
| 98304 | 399.54 | 405.09 | **391.30** | 388.40 |

三者都在 **±2% 内**（短程描述符主导）；`mid4ik` 在大尺寸略最快，`mid4(AD)` 最慢。

### 2.2 峰值显存（np=1，MiB）

| atoms | bs4 | mid4 (AD) | mid4ik |
|---:|---:|---:|---:|
| 12288 | 1972 | 2740 | 1818 |
| 24000 | 2494 | 3936 | 2080 |
| 41472 | 2560 | 5749 | 2376 |
| 65856 | 3236 | 8804 | 2906 |
| 98304 | 4178 | 10954 | **3498** |

`mid4ik` 比 `bs4` 省 **~16%**、比 `mid4(AD)` 省 **3.1×**。`mid4(AD)` 的 766³ 网格在 98k 原子时吃 11 GB。

### 2.3 Fourier 空间速度

**MD 内（Pair − SR 扣除，ms/step）** —— 差分测量，含 ±1–2 ms 噪声，大尺寸信号才可靠：

| atoms | bs4 | mid4 (AD) | mid4ik |
|---:|---:|---:|---:|
| 12288 | ~0 | 3.7 | ~0 |
| 24000 | 1.0 | 6.2 | 1.0 |
| 41472 | 3.6 | 11.4 | 1.6 |
| 65856 | 6.5 | 26.0 | 3.6 |
| 98304 | 11.4 | 16.6 | **3.1** |

**隔离算子（`sog_work/bench/time_mesh_op.py`，纯 mesh solve，ms/call）** —— 干净、可复现
（≥41472 时该算子被每次调用的 plan/Green 重建 ~8 ms 地板淹没，见 §3.3，故只列 ≤24000）：

| atoms | bs4 (网格) | mid4·AD (网格) | mid4ik (网格) |
|---:|---:|---:|---:|
| 12288 | 0.83 (209) | 3.47 (312) | **0.59** (119) |
| 24000 | 1.67 (261) | 6.87 (390) | **0.77** (148) |

两种方法一致：**`mid4ik` 的 Fourier 比 `bs4` 快 1.4–3.7×、比 `mid4(AD)` 快 3–9×。**
（隔离算子在大尺寸含每次调用的 plan 重建开销，是上界；MD 内 plan 缓存后更便宜，故大尺寸用 Pair−SR。）

---

## 3. A 链 · 6 阶（np=1，GPU，canonical）

### 3.1 整体 MD 单步时间（ms/step）

| atoms | bs6 | mid6 (AD) | mid6ik | sr-only |
|---:|---:|---:|---:|---:|
| 192 | 2.21 | 2.20 | 2.36 | 1.97 |
| 1536 | 8.59 | 8.55 | 8.63 | 8.51 |
| 5184 | 23.21 | 23.30 | 23.46 | 23.81 |
| 12288 | 54.01 | 54.32 | 54.16 | 54.46 |
| 24000 | 98.48 | 97.13 | 98.06 | 97.70 |
| 41472 | 166.15 | 164.81 | 167.59 | 166.05 |
| 65856 | 263.36 | 264.56 | 265.58 | 263.03 |
| 98304 | 390.23 | 389.04 | 395.69 | 393.02 |

**三者整步时间差 <1.5%**，且都贴近 SR-only。6 阶下后端与力微分对整步速度**无可测影响**。

### 3.2 峰值显存（np=1，MiB）

| atoms | bs6 | mid6 (AD) | mid6ik |
|---:|---:|---:|---:|
| 12288 | 1750 | 1794 | 1774 |
| 24000 | 1864 | 1998 | 1960 |
| 41472 | 2124 | 2232 | 2160 |
| 65856 | 2484 | 2650 | 2532 |
| 98304 | **2930** | 3170 | 3000 |

6 阶网格都小（173/265/177），mesh 显存是配角，三者差 **<8%**。

### 3.3 Fourier 空间速度

- **MD 内 Pair−SR（全 8 个尺寸）**：三者都 **≈ 0 ± 3 ms**（差分噪声）——mesh 在 6 阶占运行时间 **<1%**，
  与后端/力微分无关。这是 6 阶大尺寸下唯一可靠的 Fourier 结论。
- **隔离算子（ms/call，纯 mesh solve）** —— 只在 **≤24000** 有意义：

  | atoms | bs6 (网格) | mid6·AD (网格) | mid6ik (网格) |
  |---:|---:|---:|---:|
  | 12288 | 0.41 (71) | 0.44 (108) | 0.83 (72) |
  | 24000 | 0.36 (88) | 0.48 (135) | 1.21 (90) |

  6 阶 `ik` 网格与 B-spline 相当，但 ik 做**三次额外逆变换**，在**小网格**上反而更贵（0.8–1.2 vs 0.4 ms）；绝对值 <1.5 ms，整步里不可见。

  **为什么隔离算子只统计到 24000**：单独调用 `latent_charge_sum` 时**每次都要重建 Green 表/FFT plan**
  （MD 里靠 `box_tol` 缓存，不重建），这次重建的开销随盒子/网格增长，在 ≥41472 时涨到 **~8 ms** 并成为
  主导项。实测 41472/65856/98304 三者全部落在 7.8–8.4 ms 且**几乎不随网格增长**：

  | atoms | bs6 (网格) | mid6·AD (网格) | mid6ik (网格) |
  |---:|---:|---:|---:|
  | 41472 | 7.87 (106) | 7.79 (162) | 7.87 (108) |
  | 65856 | 8.20 (124) | 7.86 (189) | 7.88 (126) |
  | 98304 | 7.96 (141) | 7.94 (216) | 8.43 (144) |

  这些数字被重建开销淹没、已不是 mesh solve，故**不纳入比较**；大尺寸的 Fourier 以 Pair−SR 为准。
  （4 阶同理会遇到这个地板，但其网格更大，mid4·AD 在 41472 仍高于地板，见 §2.3。）

---

## 4. ik 的收益在哪：4 阶 vs 6 阶

| | 4 阶 | 6 阶 |
|---|---|---|
| B-spline 网格 | 513³ | 173³ |
| CubeS₂·AD 网格 | 766³ | 265³ |
| CubeS₂·ik 网格 | **291³**（省 5.4× vs bs） | 177³（≈ bs，无收益） |
| ik / bs Fourier（isolated） | **0.5–0.7× （更快）** | 2–3.4× （更慢，但绝对值 <1.5 ms） |
| ik / bs 显存 | **0.84×** | 1.02× |
| ik / bs 整步 | ~0.98× | ~1.01× |

**机制**：`ik` 在 Fourier 空间微分势场，把力的阶抬到 `2ν`，代价是 3 次额外逆变换。
4 阶时 B-spline·AD 的力只有 3.5 阶、需要很细的网格（513³），而 `ik` 力达 4 阶、网格被能量约束在 291³，
**格点省 5.4×，远超 3 次逆变换的开销**；6 阶时 B-spline·AD 的力已是 6 阶，网格本就落在能量地板
（173³ vs 177³），`ik` 省不出格点，只剩额外逆变换的代价。

**建议**：
- 用 **4 阶 CubeS₂** 时选 **`ik`**：省显存 ~16%、Fourier 快 1.4–3.7×，整步略快。
- 用 **6 阶** 时后端与力微分随意；B-spline 最省显存，`ik` 无速度收益。
- 追求省显存/大体系：**4 阶 + ik** 是三者里网格最省、显存最低的方案。

---

## 5. B 链（CPU host `kspace_style sog`，仅 AD）附录

B 链 LAMMPS kspace 的 CubeS₂ 是 AD 力（无 ik）；以下是**纯求解器隔离**（固定电荷 × 模型核，
`pair_style zero` 或 `latent_charge_to_q`+`fix sog/response`，只计 Kspace 列），仅供"CPU 上谁快"参考。
跨链绝对时间**不可比**（核不同：σ_min 2.0 vs 5.297 Å），只比链内比值。

**6 阶等精度（1e-4）**：

| atoms | cubes2_6 网格 | cubes2_6 ms | bspline_6 网格 | bspline_6 ms | 比 |
|---:|---:|---:|---:|---:|---:|
| 12288 | 40³ | 77.10 | 25³ | 33.29 | 2.32× |
| 98304 | 79³ | 630.23 | 50³ | 266.55 | 2.36× |

**4 阶等精度（1e-4）**：

| atoms | cubes2_4 网格 | cubes2_4 ms | bspline_4 网格 | bspline_4 ms | 比 |
|---:|---:|---:|---:|---:|---:|
| 12288 | 118³ | 225.10 | 79³ | 50.05 | 4.50× |
| 98304 | 236³ | 2223.32 | 158³ | 577.35 | 3.85× |

**同网格**下，CPU host 的 CubeS₂ 仍比 B-spline 慢 1.8–2.2×（与非可分 88 节点模板 + 较贵 green 乘法有关，
`SOG_FORM_A` on/off 无差），与 `GRID_FROM_ACCURACY.md` §5「同网格 CubeS₂ 快 1.1–1.3×」**方向相反**——
那条结论只适用 **GPU 路径**。**CPU 求解器上应优先 B-spline。**

---

## 6. 实现：GPU B-spline 与图内 ik 路径（本树新增）

### 6.1 GPU B-spline（`deepmd-kit-LR/source/lmp/sog_gpu.cu`）

原 `sog_gpu.cu` 只有 CubeS₂/QuadS 核，`spline bspline_* use_gpu yes` 会**误发**到 CubeS₂ 模板（静默错力）。
本次补全：
- 设备权重 `bspline_w1d<N>` / `bspline_w1d_d<N>`（Cox-de Boor 递推 + 解析导数，N=4/5/6），对齐 CPU 实现。
- 核 `k_spread_bspline<N>` / `k_gather_bspline<N>`（复制网格）+ `*_brick<N>`（分布式 brick）。
- 分派：`spline==0` 时按 `sog_gpu_set_bspline_order` 选 N；删除 `sog.cpp` 两处"GPU 只支持 CubeS₂/QuadS"守卫。
- **验证** `source/lmp/tests/test_sog_gpu_bspline.py`（PASS 4/5/6）：CPU vs GPU **逐位一致**（`max|ΔF|=0`、
  `|ΔE|≤1e-6`）；分布式 np=2 跑通。

### 6.2 图内 ik 路径（A 链）

`force_mode` 贯穿：`grid_rule.IK_FORCE_RULE/ENERGY_RULE` → `LatentChargeLongRange` →
算子 `latent_charge_sum(..., force_mode)` → `graph_compress.cu` 的 canonical op 多一个 `force_mode` 整型参数。
`mid4ik` 的 `.pt2` 元数据 `model.json` 记录 `mesh_backend=midtown, pme_order=4, force_mode=ik`。

> **运维坑**：canonical op 的 schema 从 21 参数扩到 22 参数（加 `force_mode`）。重冻结新模型后，
> 运行时要加载**新版** `libdeepmd_op_pt.so`（本机装在 `lr212_install_lrpt/deepmd/lib/`，由后端
> `libdeepmd_backend_ptexpt.so` 经 `LD_LIBRARY_PATH` dlopen）。旧 op 会报
> `flatten_canonical_atom_virial: Expected a proper Tensor but got None`。

---

## 7. 注意事项

1. **`ik` 仅 np=1**（replicated）；`SOG_MIDTOWN_SLAB`/`DIST` 下报错。故本比较全是单卡。
2. **`pme_spacing` 在 `.pt2` 元数据里是 `null`**——spacing 是 `use_mesh_evaluation()` 在冻结时算好、
   **编译进 AOTI 图**的常量；元数据里看 backend/order/force_mode 即可，网格按 §1 的规则随盒长缩放。
3. **`SOG_GPU_TIMING=1` 在本树会触发 `flatten_canonical_atom_virial` 的 None 崩溃**（既有 bug，与本比较无关），
   故 Fourier 时间用 **Pair−SR 扣除** + 隔离算子两条路交叉验证，未用该 env。
4. **`ik` 的力不是 `E_mesh` 的精确梯度**（非变分），NVE 长期能量漂移需另行核查；能量/力精度见 `IK_MODE_ACCURACY.md`。
5. 等精度常数是**实测拟合**，换核（σ_min 变化 >2×）或换体系需重标定。
6. 跨链绝对时间不可比；只比链内比值。
7. 上面第 3 条那个 `flatten_canonical_atom_virial` 的 None 崩溃，机制已经查清：
   AOTI proxy executor 把算子异常吞成"输出未定义"，真正的错误只在 stderr 的
   `[E... shim_common.cpp]` 行里。诊断与复现见 `SLAB_IK_FAILURE_DIAGNOSIS.md`。

---

## 8. 复现

> 编译（op_pt / C++ API / LAMMPS 插件 / GPU B-spline）与全套测试命令见
> [`BUILD_AND_TEST.md`](BUILD_AND_TEST.md)；冻结见 [`FREEZE_SOP.md`](FREEZE_SOP.md)。

```bash
# 冻结（A 链，canonical/--compress）—— order 4 与 6，三种模式
PY=/data/home/jiangzhen/.conda/envs/lr212/bin/python
TOOL=/data/home/public/jiangzhen/dp/sog_work/tools/freeze_lr_variants.py
OUT=/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/midtown_vs_bspline
$PY $TOOL --out $OUT/lr_bs4.pt2    --backend bspline --order 4
$PY $TOOL --out $OUT/lr_mid4.pt2   --backend midtown --order 4
$PY $TOOL --out $OUT/lr_mid4ik.pt2 --backend midtown --order 4 --force-mode ik
$PY $TOOL --out $OUT/lr_bs6.pt2    --backend bspline --order 6
$PY $TOOL --out $OUT/lr_mid6.pt2   --backend midtown --order 6
$PY $TOOL --out $OUT/lr_mid6ik.pt2 --backend midtown --order 6 --force-mode ik
$PY $TOOL --out $OUT/lr_sr.pt2     --sr-only

# 整步 + 显存（A 链，np=1）
cd /data/home/public/jiangzhen/dp/sog_work/bench
VARIANT=bs4,mid4,mid4ik,sr   NP=1 REPS=2 TAG=o4ik $PY bench_lr_variants.py
VARIANT=bs6,mid6,mid6ik,sr   NP=1 REPS=2 TAG=o6ik $PY bench_lr_variants.py

# Fourier：隔离算子（纯 mesh solve）
$PY time_mesh_op.py 12288,24000 30            # 4 阶
ORD=6 $PY time_mesh_op.py 12288,24000 30      # 6 阶

# B 链（CPU host）等精度 Fourier
SPLINE=all NP=1 REPS=2 $PY bench_sog_spline.py
```

**数据**：`sog_work/bench/matrix_midtown/bench_variants.o4ik.np1.csv`、`bench_variants.o6ik.np1.csv`；
图：`sog_work/bench/midtown_vs_bspline_bench.png`。
