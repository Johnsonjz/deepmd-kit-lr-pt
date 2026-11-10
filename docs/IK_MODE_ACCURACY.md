# ik 模式：CubeS₂ 的 Fourier 力微分精度表与网格规则

lr-pt 的 CubeS₂ midtown 网格现在有两种力微分：默认的 **AD**（权重微分，`force_mode="ad"`）
和可选的 **ik**（Fourier 微分，`force_mode="ik"`）。本文给出两种模式的实测误差律
`ε = C (Δ/σ_min)^p`，与 B-spline（只有 AD）逐项对比，并记录据此落进代码的网格规则。

配套：

- 模式本身、管道与两个静默坑：[AD_MODE_ACCURACY.md §1.4](AD_MODE_ACCURACY.md)
- AD 模式的全套精度表：[AD_MODE_ACCURACY.md §2](AD_MODE_ACCURACY.md)
- 误差模型的三分项形式：[MESH_ERROR_MODEL_AND_FORM_A.md](MESH_ERROR_MODEL_AND_FORM_A.md)
- 精度 → 网格的两棵树工作流：[GRID_FROM_ACCURACY.md](GRID_FROM_ACCURACY.md)

---

## 1. 两种力的定义

```
AD : F_i = −q_i Σ_k (∂w_k/∂r_i) φ_k            对分配窗口求导（精确 −∇E_mesh）
ik : u_a = IFFT(i k_a φ̂);  F_i = −q_i Σ_k w_k u_a[k]    对势场做谱微分
```

- AD 的力是离散网格能量 `E_mesh = ½ qᵀ Wᵀ G W q` 的精确梯度（`G` 对称 ⇒ 只对 gather
  求导即可），因此能量/力/维里自洽、NVE 无漂移；代价是要对**只 C⁰ 的 CubeS₂ 窗口**求导，
  掉一阶。
- ik 的力是 `−q_i ∇_具体 φ_i`，把求导搬进 Fourier 空间：`∂/∂x ↔ i k_x`。它不再对分配
  窗口求导，所以**力拿到与能量相同的阶 `2ν`**；代价是力不是 `E_mesh` 的梯度（非变分）。

`ik` 只在 **replicated** midtown 解上实现；`SOG_MIDTOWN_SLAB` / `SOG_MIDTOWN_DIST` 下显式报错。
`ik` 仅 lr-pt（A 链）有；LAMMPS 插件（B 链）的 CubeS₂ 仍是 AD adjoint。

---

## 2. 实测方法

- **参考解**：同核的收敛直接 k-sum（`kcut = 14`，`L = 12 Å`，核尾 `~1e-46`），解析力。
- **系综**：32 个独立 64 原子构型，中性电荷。
- **指标**：逐构型 `|ΔE|/|E|` 与 `rms|ΔF|/rms|F|`，再对系综取 RMS。
- **横坐标**：网格**真实**间距 `Δ = L/n`（`n` 取两个后端都原样接受的档，不被量化）。
- **两个训练核**：`σ_min = 2.00 Å`（核 A，`run_150k_widthfloor`）与 `1.13 Å`
  （核 B，`run_200k_noramp`）。
- **经发布算子测量**：包含真实的 Form A 影响函数、分配因子与 stencil，不是重实现。

---

## 3. 能量误差（`ik` 与 `ad` 相同，同一 `φ̂`）

`ε_E = C_E (Δ/σ_min)^{p_E}`，逐核拟合：

| 阶 | backend | `p_E` (A / B) | `C_E` (A / B) |
|---|---|---|---|
| 4 | B-spline | 4.28 / 4.28 | 9.87e-1 / 7.11e-1 |
| 6 | B-spline | 7.06 / 7.18 | 4.02e-1 / 2.92e-1 |
| 4 | CubeS₂（ad=ik） | 5.09 / 4.41 | 1.65e-1 / 7.14e-2 |
| 6 | CubeS₂（ad=ik） | 6.42 / 6.40 | 6.64e-2 / 8.03e-2 |

能量指数实测都**高于** `2ν`（4.28→4.28、7.06→7.18、5.09→4.41、6.42→6.40），所以
把规则指数**向下取到 `2ν`**（4 / 6）是保守的。取 `p_E = 2ν`、`C_E` 在较差核上重拟合：

| backend | 阶 | 规则 `(C_E, p_E)` |
|---|---|---|
| B-spline | 4 | (7.34e-1, 4.0) |
| B-spline | 6 | (1.58e-1, 6.0) |
| CubeS₂ | 4 | (5.12e-2, 4.0) |
| CubeS₂ | 6 | (5.84e-2, 6.0) |

**CubeS₂ 的能量常数比 B-spline 小 3–12×**（同网格能量更准），这与 AD 模式一致。

---

## 4. 力误差

`ε_F = C_F (Δ/σ_min)^{p_F}`，逐核拟合：

| 阶 | backend | mode | `p_F` (A / B) | `C_F` (A / B) |
|---|---|---|---|---|
| 4 | B-spline | ad | 3.40 / 3.35 | 7.62e-2 / 1.46e-1 |
| 6 | B-spline | ad | 6.36 / 6.23 | 3.52e-2 / 6.13e-2 |
| 4 | CubeS₂ | ad | 3.21 / 3.16 | 1.11e-1 / 2.30e-1 |
| 6 | CubeS₂ | ad | 5.34 / 5.32 | 9.08e-2 / 1.95e-1 |
| **4** | **CubeS₂** | **ik** | **4.37 / 4.38** | **2.83e-2 / 2.76e-2** |
| **6** | **CubeS₂** | **ik** | **6.59 / 6.64** | **3.04e-2 / 3.05e-2** |

**关键结果**：

- ik 的力指数**比 AD 高一阶**：`3.2 → 4.4`（4 阶）、`5.3 → 6.6`（6 阶），且达到 `2ν`。
  4/6 阶都如此——不只是 C⁰ 的 6 阶窗口。
- ik 的力常数**比 AD 小 4–7×**；两个核之间差 <1 %（AD 差 ~2×），所以 ik 的常数近乎普适。
- 指数向下取到理论值 `p_F = 2ν`（4 / 6）后，`C_F` 在较差核上重拟合：

| backend | 阶 | 规则 `(C_F, p_F)` |
|---|---|---|
| **CubeS₂ ik** | **4** | **(1.89e-2, 4.0)** |
| **CubeS₂ ik** | **6** | **(1.83e-2, 6.0)** |

---

## 5. 同网格对比：ik vs B-spline

力相对误差比值（`CubeS₂-ik / B-spline-ad`，<1 表示 ik 更准），`σ_min = 2.00 Å`：

| `Δ/σ_min` | 4 阶 | 6 阶 |
|---|---|---|
| 0.75 | 0.29× | 0.78× |
| 0.50 | 0.17× | 0.76× |
| 0.30 | 0.12× | 0.66× |
| 0.20 | 0.07× | — |
| 0.15 | 0.06× | — |

- **4 阶：ik 同网格比 B-spline 准 3–17×**（网格越细差距越大，因为指数更高）。
- **6 阶：ik 同网格比 B-spline 准 1.3–1.5×**（接近，但稳定领先）。
- 对照 AD：`CubeS₂ / B-spline` 在 4 阶是 **1.4–2.0×（更差）**，6 阶是 **2–10×（更差）**。
  **ik 把 CubeS₂ 的力从"明显不如 B-spline"翻转为"与 B-spline 相当或更准"。**

---

## 6. 等精度网格（`1e-4`，参考核 `σ=2.00 Å`、盒长 122.31 Å）

`Δ/σ_min` 反解自误差律；格点数是 `ceil(122.31/Δ)³`：

| backend·mode | 阶 | `Δ/σ_min` | 网格 | 谁在卡 |
|---|---|---|---|---|
| B-spline·ad | 4 | 0.119 | 513³ | 力 |
| B-spline·ad | 6 | 0.354 | 173³ | 力 |
| B-spline·ad | 8（缺省） | 0.532 | 115³ | 力 |
| CubeS₂·ad | 4 | 0.080 | 766³ | 力 |
| CubeS₂·ad | 6 | 0.231 | 265³ | 力 |
| **CubeS₂·ik** | **4** | **0.210** | **291³** | **能量** |
| **CubeS₂·ik** | **6** | **0.346** | **177³** | **能量** |

**读法**：

- ik 模式下力与能量同阶，且能量常数更大，所以**能量成为约束**（`spacing()` 取两者较小者）。
  若不考虑能量、只按力，ik 的网格是 `0.270 / 0.420` → **227³ / 146³**（见 §6.1）。
- **4 阶：ik 比 B-spline 省 5.4× 格点**（291³ vs 513³），比 AD 省 18×（291³ vs 766³）。
- **6 阶：ik 与 B-spline 基本持平**（177³ vs 173³，多 2%）。只按力则 ik 省 1.7×（146³）。

### 6.1 只按力的 ik 网格（不取能量约束）

| 阶 | 力约束 `Δ/σ` | 网格 | 该网格下能量误差 |
|---|---|---|---|
| 4 | 0.270 | 227³ | ~3×10⁻⁴（3× 目标） |
| 6 | 0.420 | 146³ | ~1.2×10⁻⁴（1.2× 目标） |

即：只按力时 ik 更省格点，但 4 阶的能量会到目标的 ~3×。**发布规则取 `min(能量, 力)`**，
对两者都保证 `pme_accuracy`；若要最大化省格点且只关心力，把 `pme_accuracy` 当作力口径
（改动只有 `grid_rule.spacing` 的 energy 分支一行）。

---

## 7. 落到代码

### 7.1 A 链（`deepmd/dpmodel/long_range/grid_rule.py`）

新增两张表与 `force_mode` 贯穿：

```python
IK_FORCE_RULE = {                     # ik 力（CubeS₂ 专有）
    ("midtown", 4): (1.89e-2, 4.0, "measured"),
    ("midtown", 6): (1.83e-2, 6.0, "measured"),
}
ENERGY_RULE = {                       # 能量（仅 ik 会与力比）
    ("bspline", 4): (7.34e-1, 4.0, "measured"),
    ("bspline", 6): (1.58e-1, 6.0, "measured"),
    ("midtown", 4): (5.12e-2, 4.0, "measured"),
    ("midtown", 6): (5.84e-2, 6.0, "measured"),
    ...
}
rule(backend, order, force_mode="ad")      # ik 取 IK_FORCE_RULE，其它走旧表
energy_rule(backend, order)                # 能量律
spacing(..., force_mode="ad")             # ik 取 min(力, 能量)，其余仍是力
```

`LatentChargeLongRange.mesh_spacing()` 把 `self.force_mode` 传进 `grid_rule.spacing`，
冻结（`use_mesh_evaluation`）与训练网格都随之切换。

### 7.2 B 链（`../deepmd-kit-LR/source/lmp/sog.cpp`）

LAMMPS 的 SOG kspace 算的是 **AD** 力，本树**没有** ik 力路径，所以网格仍用 AD 常数。
ik 的常数（`C_F = 0.0189 / 0.0183 @ p=4/6`、`C_E = 0.0512 / 0.0584`）记在该处
"canonical table" 注释里，并注明**仅图内算子的 Fourier 模式适用**。两树的 **AD** 常数
保持逐项一致（本次未改）。

> 若要让 B 链也支持 ik（LAMMPS 内实现三次 `i k_a` 逆变换 + 梯度 gather），那是独立的
> 力路径实现，不属于本次常数同步。

### 7.3 单测（`source/tests/common/dpmodel/test_grid_rule.py`）

新增 `TestIkForceMode`：钉住 ik 力常数、`ik` 仅 midtown、默认仍是 AD、ik 网格取
`min(能量, 力)`、ik 比 AD 粗、参考核网格 `291³ / 177³`。

---

## 8. 复现

```bash
PY=/data/home/jiangzhen/.conda/envs/lr212/bin/python
cd /data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
$PY docs/measure_ik_tables.py \
    /data/home/public/jiangzhen/dp/master-merge-test/water/run_150k_widthfloor_archive/model.ckpt-150000.pt /tmp/ik_k200.json
$PY docs/measure_ik_tables.py \
    /data/home/public/jiangzhen/dp/master-merge-test/water/run_200k_noramp_archive/model.ckpt-200000.pt /tmp/ik_k113.json
$PY -m pytest source/tests/common/dpmodel/test_grid_rule.py -q
```

`docs/measure_ik_tables.py` 三行 `(bspline,ad) / (midtown,ad) / (midtown,ik)` 走发布算子；
`docs/check_cubes2_ik_mode.py` 是更早的 ik/AD 斜率对照与护栏检查。
