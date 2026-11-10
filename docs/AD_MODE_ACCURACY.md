# AD 模式：设置、精度表与网格规则

本文记录"全线 AD"之后的实际状态：**两条链、两个后端、所有路径都用权重微分
（weight differentiation）算力**，给出能量与力的实测精度表，以及据此更新的网格规则。

配套：

- 力的微分方式对照与改造计划：[AD_EVERYWHERE_PLAN.md](AD_EVERYWHERE_PLAN.md)
- midtown AD 的结构优化设计：[MIDTOWN_AD_DESIGN.md](MIDTOWN_AD_DESIGN.md)
- 精度 → 网格的工作流（两棵树）：[GRID_FROM_ACCURACY.md](GRID_FROM_ACCURACY.md)
- 误差模型本身：[MESH_ERROR_MODEL_AND_FORM_A.md](MESH_ERROR_MODEL_AND_FORM_A.md)

---

## 1. AD 模式的设置

### 1.1 力的定义

```
E_mesh(r) = ½ Σ_i q_i φ_i(r),      F_i = −∂E_mesh/∂r_i = −q_i Σ_k (∂w_k/∂r_i) φ_k
```

因为 `E = ½ qᵀWᵀGWq`，`dE/dx_i` 的两半（spread 侧与 gather 侧）在 `G` 对称下相等，
所以**只对 gather 求导就是精确梯度**——不需要补 spread 项的导数。这也解释了为什么
CUDA 的 gather-only 实现能与完整 autograd 的 CPU 参考对齐到 1e-6。

### 1.2 各实现的落点

| 实现 | 力的微分 | 位置 |
|---|---|---|
| lr-pt · B-spline | AD（预计算 `value`/`slope` + 张量积缩并） | `mesh_sum.cu` `node_stencils` / `interpolate_owners` |
| lr-pt · CubeS₂ | AD（逐节点 `cs2_wg6`，值 + 三个偏导） | `midtown/sog_gpu.cu` `k_gather_cubes2_warp_r` |
| lr-pt · CPU/参考 | AD（完整 autograd） | `pt_expt/kernels/long_range.py` |
| LR · B-spline | AD（2026-11 由 ik 改来） | `sog.cpp` `sog_bspline_weights_1d_order_d` |
| LR · CubeS₂ / QuadS | AD（adjoint，1 次逆变换） | `sog.cpp` |

**AD 是默认，且除一处外没有 ik。** LR 的 B-spline 从三个 ik 梯度场 + 三次
`BACKWARD` 改成一次逆变换 + 权重微分，同时删掉了 `mesh_gradx/y/z`；
`use_gpu yes` + `spline bspline_*` 以前会静默按 CubeS₂-4 跑，现在是显式报错。
唯一保留 ik 的地方是 **lr-pt 的 CubeS₂**，作为可选模式（§1.4）：它不改变默认行为。

### 1.3 与 ik 的差别（供对照）

在**同一条管线**里量过（B-spline，σ_min = 2.00 Å）：

| 阶 | scheme | `p_F` | `C_F` | `Δ/σ` @1e-4 |
|---|---|---|---|---|
| 4 | AD | 3.40 | 7.63e-2 | 0.1419 |
| 4 | ik | 4.37 | 5.10e-2 | 0.2402 |
| 6 | AD | 6.36 | 3.52e-2 | 0.3979 |
| 6 | ik | 7.40 | 2.83e-2 | 0.4663 |

ik 恰好比 AD 高一个阶（对 `C⁰` 窗口求导掉一阶）。换成 AD 的代价是等精度下
多 **1.61×（6 阶）/ 4.84×（4 阶）** 格点，换来的是精确 `−∇E_mesh`
（能量/力/维里自洽、NVE 无漂移）和更少的逆变换。

### 1.4 CubeS₂ 的可选 ik 模式（`force_mode`）

lr-pt 的 CubeS₂ 保留 ik 作为**可选**模式，默认仍是 AD：

```json
"long_range": {"type": "latent_charge", "periodic": true,
               "mesh_backend": "midtown", "force_mode": "ik"}
```

管道与实现（`midtown/sog_gpu.cu`）：

```
spread → R2C → 影响函数(φ̂) ─┬→ k_mul_ik(axis=0/1/2) → C2R → gradx/y/z ┐
                          └→ C2R → φ 网格                              ┴→ 值权重 gather
```

- `k_mul_ik` 把半谱乘以 `i·k_a`；自共轭点（`k = −k`）的系数对实场必须是实数，
  乘 `i·k_a` 后不再是 Hermitian，故这些点清零。
- **顺序有坑**：势场的 `c2r` 会重用 `spec` 所别名的那块缓冲（`f_rho_fx`），
  所以三次梯度变换必须**排在势场 `c2r` 之前**。写反时能量不变、力却全错且随
  `n` 快速增长（实测误差 ∝ n¹·⁹，与影响函数无关）——这是本条实现里最容易
  复现的静默错误。
- `force_mode="ik"` 仅在 **replicated** midtown 解上实现；`SOG_MIDTOWN_SLAB` /
  `SOG_MIDTOWN_DIST` 下会显式报错，不会静默退回 AD。

实测（同一 CubeS₂ Form A 管道，`σ_min = 2.00 Å`，32 构型 × 直接 k-sum 参考，
`docs/check_cubes2_ik_mode.py`）：

| 阶 | AD `p_F` | ik `p_F` | ik/AD 误差比 @Δ/σ=0.2 |
|---|---|---|---|
| 4 | 3.25 | **4.20** | 0.04 |
| 6 | 5.16 | **6.26** | 0.05 |

ik 严格高一阶，细网格上误差小 4–25×。能量与维里在两种模式下完全相同
（同一 φ̂），力在细网格上与 AD 收敛到同一值（corr = 1.0000）。改变前后的
**AD 默认路径逐位不变**（12 case × 5 输出，`max|Δ| = 0`）。

ik 的完整误差律（能量 `C_E`、力 `C_F`）、与 B-spline 的逐项对比、以及落进
`grid_rule.py` 的网格规则见 **[IK_MODE_ACCURACY.md](IK_MODE_ACCURACY.md)**。一句话：
ik 把 CubeS₂ 的力指数从 `2ν−1` 抬到 `2ν`（4.4 / 6.6），力常数降一个量级
（`C_F = 1.89e-2 / 1.83e-2`）；等精度网格 4 阶比 B-spline 省 5.4×、6 阶基本持平。

---

## 2. 数值结果

**方法**：两个训练核（`σ_min = 2.00 Å` 与 `1.13 Å`）× 32 个独立 64 原子构型；
参考解为同核的收敛直接 k-sum（`kcut = 14`，核尾 ~1e-46）；力为固定电荷、
rms 相对误差；横坐标取网格**真实**间距 `Δ = L/n`（阶梯上的 n 两个后端都原样接受，
不会被量化掉）。全部**经发布算子**测量，即包含真实的 Form A 影响函数与分配因子。

### 2.1 能量

`ε_E = C_E (Δ/σ_min)^{p_E}`，逐核拟合（R² 见表后）：

| 阶 | backend | 核 A (σ=2.00) `p_E` / `C_E` | 核 B (σ=1.13) `p_E` / `C_E` |
|---|---|---|---|
| 4 | B-spline | 4.28 / 9.87e-1 | 4.28 / 7.11e-1 |
| 4 | **CubeS₂** | **5.09 / 1.65e-1** | **4.41 / 7.14e-2** |
| 6 | B-spline | 7.06 / 4.02e-1 | 7.18 / 2.92e-1 |
| 6 | **CubeS₂** | **6.42 / 6.64e-2** | **6.40 / 8.03e-2** |

同网格比值（CubeS₂ / B-spline）：**4 阶 0.02–0.21×（好 5–50 倍）**，
**6 阶 0.09–0.43×（好 2–11 倍）**（n=18 档例外，见 §2.4）。

### 2.2 力

`ε_F = C_F (Δ/σ_min)^{p_F}`：

| 阶 | backend | 核 A `p_F` / `C_F` | 核 B `p_F` / `C_F` |
|---|---|---|---|
| 4 | B-spline | 3.40 / 7.62e-2 | 3.35 / 1.46e-1 |
| 4 | CubeS₂ | 3.21 / 1.11e-1 | 3.16 / 2.30e-1 |
| 6 | B-spline | 6.36 / 3.52e-2 | 6.23 / 6.13e-2 |
| 6 | CubeS₂ | 5.34 / 9.08e-2 | 5.32 / 1.95e-1 |

同网格比值（CubeS₂ / B-spline）：**4 阶 1.2–2.0×（差 1.2–2 倍）**，
**6 阶 2.1–9.6×**，且随网格变细而变大——因为 CubeS₂ 的力指数低一阶。

### 2.3 等精度下的网格

| 口径 | 阶 | B-spline `Δ/σ` | CubeS₂ `Δ/σ` | 谁省格点 |
|---|---|---|---|---|
| **力** 1e-4 | 4 | 0.119 | 0.080 | B-spline 省 **3.3×** |
| **力** 1e-4 | 6 | 0.354 | 0.231 | B-spline 省 **3.6×** |
| 能量 1e-4 | 4 | 0.126 | 0.225 | CubeS₂ 省 **5.7×** |
| 能量 1e-4 | 6 | 0.329 | 0.352 | CubeS₂ 省 1.2× |

**结论（AD 模式下）**：

- **能量：CubeS₂ Form A 明显更准**（常数小 4–14×），细网格上优势更大；
- **力：B-spline 明显更准**（1.2–9.6×），且网格越细差距越大；
- 这不是实现差异，而是**分配窗口的连续性类别**决定的：力要对窗口求导，
  两条曲线都掉一阶，但 B-spline 的起点更高。

### 2.4 一个例外

n = 18（`Δ/σ = 0.333`）处 CubeS₂-6 的能量误差是相邻档的 3.6 倍，比 B-spline 还差
1.19×。把 Form A 的角向求积从 8 阶提到 64 阶误差不变（收敛到 1.52e-4），
换盒长后尖峰仍在 n=18 → 是**该网格尺寸下窗口符号本身**在放大混叠，
不是求积没收敛，也不是编码错误。选网格时应留余量。

---

## 3. 落到代码：更新后的网格规则

`C` 在固定的 `p`（沿用原表）上、对**较差的那个核**重新拟合，取两核较大者，
这样同一张网格对两个核都满足 1e-4。`p` 不变（实测 3.35–3.40 / 6.23–6.36 /
3.16–3.21 / 5.32–5.34，原表的 3.5 / 6.0 / 3.0 / 5.0 都在 0.2 以内）。

| backend | 阶 | 旧 `(C, p)` | **新 `(C, p)`** | 变化 |
|---|---|---|---|---|
| bspline | 4 | (1.17e-1, 3.5) | **(1.70e-1, 3.5)** | +45% |
| bspline | 5 | (6.24e-2, 4.5) | **(9.32e-2, 4.5)** | +49% |
| bspline | 6 | (3.34e-2, 6.0) | **(5.12e-2, 6.0)** | +53% |
| bspline | 7 | (2.02e-2, 7.0) | **(2.82e-2, 7.0)** | +40% |
| bspline | 8 | (1.22e-2, 8.0) | **(1.55e-2, 8.0)** | +27% |
| midtown | 4 | (1.66e-1, 3.0) | **(1.96e-1, 3.0)** | +18% |
| midtown | 6 | (1.24e-1, 5.0) | **(1.52e-1, 5.0)** | +23% |

B-spline 的逐阶比 `_BSPLINE_C_RATIO` 1.65 → **1.82**。

**参考核上的网格变化**（`σ_min = 2.00 Å`，盒长 122.31 Å，1e-4；误差律直接给网格，无 Nyquist 上限）：

| backend | 阶 | 旧网格 | **新网格** |
|---|---|---|---|
| bspline | 4 | 461³ | **513³** |
| bspline | 6 | 162³ | **173³** |
| bspline | 8（缺省） | 158³ | **115³** |
| midtown | 4 | 725³ | **766³** |
| midtown | 6 | 255³ | **265³** |

网格细了 4–10%。原因是旧常数只反映了 `σ_min = 2.00 Å` 那个核；
`σ_min = 1.13 Å` 的核常数大约大一倍，取大者后网格必须更细才真的达到 1e-4。

已同步的三处（必须逐项一致）：

- `deepmd/dpmodel/long_range/grid_rule.py`（`FORCE_RULE` + `_BSPLINE_*`）
- `../deepmd-kit-LR/source/lmp/sog.cpp`（`C_nu`，`phi_accuracy` 分支）
- `source/tests/common/dpmodel/test_grid_rule.py`（常数、插值/外推行、参考网格）

---

## 4. 复现

```bash
# 表 2.1 / 2.2 的原始扫描（两个核各一次）
PY=/data/home/jiangzhen/.conda/envs/lr212/bin/python
export PYTHONPATH=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
$PY docs/measure_ad_tables.py \
    /data/home/public/jiangzhen/dp/master-merge-test/water/run_150k_widthfloor_archive/model.ckpt-150000.pt /tmp/k200.json
$PY docs/measure_ad_tables.py \
    /data/home/public/jiangzhen/dp/master-merge-test/water/run_200k_noramp_archive/model.ckpt-200000.pt /tmp/k113.json
# 规则自检
$PY -m pytest source/tests/common/dpmodel/test_grid_rule.py -q
```

`docs/measure_ad_tables.py` 通过发布算子测量（因此包含 Form A 影响函数），
参考解为直接 k-sum，横坐标是真实 `Δ = L/n`。
