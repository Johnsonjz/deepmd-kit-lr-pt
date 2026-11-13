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

| backend | 阶 | 规则 `(C_E, p_E)`，k 空间归一 `|δE|/|E_k|` |
|---|---|---|
| B-spline | 4 | (3.33e-2, 4.0) |
| B-spline | 6 | (7.01e-3, 6.0) |
| CubeS₂ | 4 | (3.81e-3, 4.0) |
| CubeS₂ | 6 | (4.03e-3, 6.0) |

**CubeS₂ 的能量常数比 B-spline 小 2–9×**（同网格能量更准），这与 AD 模式一致。
（`spacing()` 现在对所有 backend / force mode 都按这张能量律定网格，B-spline 行也不例外；
力律只供需要显式力保证的调用方反解。）

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

`Δ/σ_min` 反解自**能量律**（`ENERGY_RULE`，k 空间归一）；格点数是 `ceil(122.31/Δ)³`。
`spacing()` 对 AD 和 ik 用同一张能量律，所以两种力模式给**同一个网格**：

| backend·mode | 阶 | `Δ/σ_min` | `Δ` (Å) | 网格 |
|---|---|---|---|---|
| B-spline·ad | 4 | 0.234 | 0.468 | 262³ |
| B-spline·ad | 6 | 0.493 | 0.985 | 125³ |
| B-spline·ad | 8（缺省） | 0.714 | 1.428 | 86³ |
| CubeS₂·ad | 4 | 0.403 | 0.805 | 152³ |
| CubeS₂·ad | 6 | 0.540 | 1.080 | 114³ |
| **CubeS₂·ik** | **4** | **0.403** | **0.805** | **152³** |
| **CubeS₂·ik** | **6** | **0.540** | **1.080** | **114³** |

**读法**：

- **网格由能量律唯一决定，与力模式无关**：AD 和 ik 给同一网格（能量口径下两者本就同一
  `φ̂`）。这是「代码统一为能量口径」的关键改动。
- 能量口径下 **CubeS₂ 比 B-spline 省格点**：4 阶 152³ vs 262³（省 5.1×），6 阶 114³ vs
  125³（省 1.3×）——正是 CubeS₂ 能量常数小 2–9× 的体现。

### 6.1 能量口径下的力误差（已接受的代价）

能量律网格下力**不再**受 1e-4 保证：AD 力指数比能量低一阶，这些网格下 AD 力误差约为目标的
7–60×（CubeS₂-4 AD 最差，~6×10⁻³）；ik 力误差 ~5×。需要力保证时反解 `IK_FORCE_RULE`
（力约束的 ik 网格 227³ / 146³，见 §4）或 `FORCE_RULE`。

> 注：早先的 `ENERGY_RULE`（`C_E = 5.12e-2 / 5.84e-2`）用**总能量** `|E|` 归一，且来自旧核
> `run_150k`（`G(0)=0.19`）。总能量含自能 `½G(0)Σq²`，使 `ε_E` 依赖核特异的 `G(0)`；
> 换成 k 空间归一 `|δE|/|E_k|` 后，`C_E = 3.81e-3 / 4.03e-3`（部署核 `G(0)=0.33` 与旧核
> 逐位一致）。随后 `spacing()` 对 AD 也改用能量律，力保证从部署路径移除。

---

## 7. 落到代码

### 7.1 A 链（`deepmd/dpmodel/long_range/grid_rule.py`）

新增两张表与 `force_mode` 贯穿：

```python
IK_FORCE_RULE = {                     # ik 力（CubeS₂ 专有）
    ("midtown", 4): (1.89e-2, 4.0, "measured"),
    ("midtown", 6): (1.83e-2, 6.0, "measured"),
}
ENERGY_RULE = {                       # 能量，k 空间归一 |δE|/|E_k|
    ("bspline", 4): (3.33e-2, 4.0, "measured"),
    ("bspline", 6): (7.01e-3, 6.0, "measured"),
    ("midtown", 4): (3.81e-3, 4.0, "measured"),
    ("midtown", 6): (4.03e-3, 6.0, "measured"),
    ...
}
rule(backend, order, force_mode="ad")      # ik 取 IK_FORCE_RULE，其它走旧表
energy_rule(backend, order)                # 能量律
spacing(..., force_mode="ad")             # 全部走能量律；force_mode 不再改变网格
```

`LatentChargeLongRange.mesh_spacing()` 仍把 `self.force_mode` 传进 `grid_rule.spacing`
（API 兼容），但能量律与力模式无关，所以 AD / ik 的冻结与训练网格相同。

### 7.2 B 链（`../deepmd-kit-LR/source/lmp/sog.cpp`）

LAMMPS 的 SOG kspace 算的是 **AD** 力，本树**没有** ik 力路径，所以网格仍用 AD 常数。
ik 的常数（`C_F = 0.0189 / 0.0183 @ p=4/6`、`C_E = 0.00381 / 0.00403`，k 空间归一）记在该处
"canonical table" 注释里，并注明**仅图内算子的 Fourier 模式适用**。两树的 **AD** 常数
保持逐项一致（本次未改）。

> 若要让 B 链也支持 ik（LAMMPS 内实现三次 `i k_a` 逆变换 + 梯度 gather），那是独立的
> 力路径实现，不属于本次常数同步。

### 7.3 单测（`source/tests/common/dpmodel/test_grid_rule.py`）

新增 `TestIkForceMode`：钉住 ik 力常数、`ik` 仅 midtown、默认仍是 AD、`spacing` 对 AD 与
ik 给同一能量律网格、参考核网格 `152³ / 114³`（AD = ik）。

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
