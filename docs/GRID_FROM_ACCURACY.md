# 目标精度 → 网格：两条链的工作流

本文记录 SOG 长程静电在**两棵代码树**里如何把目标精度（缺省 `1e-4`，**能量**相对误差、k 空间归一）反推成
网格参数，以及两条链的差异、已知陷阱与验证方式。

- **A 链（图内算子）** `/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt`
  —— `deepmd::latent_charge_sum` 自定义算子，训练与冻结模型、AOTI 图内长程都走它。
- **B 链（LAMMPS 插件）** `/data/home/public/jiangzhen/dp/deepmd-kit-LR`
  —— `pair_style deepmd` + `kspace_style sog`，MD 生产走它。

两条链共享同一张 `(C, p)` 实测常数表；**差异在于谁执行反解**：A 链在 Python，
B 链在 LAMMPS init 的 C++ 里。

---

## 0. 误差律本身

$$
\varepsilon_{\text{mesh}} \;=\; \underbrace{\varepsilon_{\text{tail}}(k_{\max})}_{\text{核的 k 截断}}
\;+\; \underbrace{(C_{\text{assign}} + C_{\text{infl}})\left(\frac{\Delta}{\sigma_{\min}}\right)^{p}}_{\text{网格}}
$$

- `σ_min`：核中最窄高斯的宽度，`σ_min = exp(min(log_width))`。
- `Δ`：网格间距。
- `C_assign`：样条的插值/混叠残差，样条固有。
- `C_infl`：影响函数残差。**Form A（精确除掉窗口符号 `|W|²`）下恒为零**，只有 Form B
  （方差扣除）非零。
- **指数 `p` 不是 `2ν`**：力要对分配窗口求导，只有 `C⁰` 的窗口掉一阶。

| backend | order | `C` | `p` | 来源 |
|---|---|---|---|---|
| bspline | 4 | 1.70e-1 | 3.5 | 实测 |
| bspline | 5 | 9.32e-2 | 4.5 | 对数插值 |
| bspline | 6 | 5.12e-2 | 6.0 | 实测 |
| bspline | 7 | 2.82e-2 | 7.0 | 外推 |
| bspline | 8 | 1.55e-2 | 8.0 | 外推 |
| midtown (CubeS₂) | 4 | 1.96e-1 | 3.0 | 实测 |
| midtown (CubeS₂) | 6 | 1.52e-1 | 5.0 | 实测 |
| midtown (CubeS₂)·**ik** | 4 | 1.89e-2 | 4.0 | 实测，**仅 A 链** |
| midtown (CubeS₂)·**ik** | 6 | 1.83e-2 | 6.0 | 实测，**仅 A 链** |
| QuadS | 4 | 2.0e-4 | 4.0 | **仅 B 链**，Form A，未做力量级审计 |
| QuadS | 6 | 2.0e-4 | 6.0 | 同上 |

上表是**力律**（`FORCE_RULE` / `IK_FORCE_RULE`，固定电荷、力相对 RMS）。`.ik` 行是
`force_mode="ik"`（Fourier 微分）的力常律；它把力的指数抬到 `2ν`。

**部署网格现在由能量律 `ENERGY_RULE`（k 空间归一 `|δE|/|E_k|`）唯一决定**，A 链对所有
backend / force mode 都按它定网格：B-spline 4/6 阶 `C_E = 3.33e-2 / 7.01e-3`，CubeS₂ 4/6
阶 `C_E = 3.81e-3 / 4.03e-3`，QuadS 4/6 阶 `C_E = 3.39e-3 / 1.81e-3`，`p = 4/6`。
归一化用 `|E_k|` 而非总能量 `|E|`：自能项 `½G(0)Σq²` 精确、不产生网格误差，用 `|E|`
会把核特异的 `G(0)` 混进来。能量口径下 AD 与 ik 给同一网格，但 AD 力误差会高于目标
（力指数低一阶）；需要力保证时反解上面的力律表。详见 [IK_MODE_ACCURACY.md](IK_MODE_ACCURACY.md)。
**B 链（LAMMPS kspace）仍是力口径**（只算 AD 力），不随本次能量口径改动。

常数口径：**固定电荷、力相对 RMS**，参考解为同一核的收敛直接 k-sum（解析力），
两个训练核（`σ_min = 1.13 Å` 与 `2.00 Å`）× 8 个 64 原子构型，取两核中较大者。
`p` 一律向下取整，所以规则只会多分辨、不会少分辨。

> 这张表是**实测拟合**，不是严格上界。指数 `2ν` 是严格的（Ewald 分裂 + 样条阶），
> 前因子是特定体系上的 RMS 拟合。换核请重新标定。

---

## 1. A 链：Python 决策，算子只量化

```
配置 pme_accuracy ──► grid_rule.spacing() ──► spacing (Å) ──► latent_charge_sum(spacing) ──► next_factorable(ceil(L/spacing))
        (缺省 1e-4)        │                                                        ▲
                           └── σ_min, k_max ← 冻结/当前的核参数                      └── 算子只做「向上取整到可变换尺寸」
```

### 1.1 逐步

**(1) 目标精度** —— `deepmd/utils/argcheck.py:3833` 的 `pme_accuracy`，
缺省走 `grid_rule.DEFAULT_ACCURACY = 1e-4`。语义是**能量**相对误差（k 空间归一）。

**(2) 定阶** —— `deepmd/dpmodel/long_range/latent_charge.py:512-523`：

```python
def effective_order(self):
    if self.mesh_backend == "midtown":
        return 4 if self.pme_order == 4 else 6    # CubeS₂ 只有 4/6 阶
    return int(self.pme_order)                    # B-spline；PME_ORDER = 8
```

**陷阱**：缺省 `pme_order = 8`，所以切到 `mesh_backend: midtown` 时**实际是 6 阶**，
但 B-spline 侧仍是 8 阶 —— 两者的网格不是同一个精度口径。

**(3) 取核的两个尺度** —— `latent_charge.py:487-543`：

- `σ_min = exp(min(log_width))`（LES 核用 `1/(α√2)`）
- `k_max = sqrt(2 ln(1/tol)) / σ_min`，`tol = kspace_tolerance = 1e-8`

**(4) 反解** —— `grid_rule.spacing()`：

```python
ce, pe, _ = energy_rule(backend, order)          # 能量律，与 force_mode 无关
d_table = sigma_min * (accuracy / ce) ** (1.0 / pe)
```

**(5) 一道夹逼（CubeS₂ 有效性上限）**

```python
if backend == "midtown":
    d = min(d_table, 0.95 * sigma_min / (sqrt(2) * xi))  # 方差扣除要求 Δ ≤ σ_min/(√2 ξ)
```

- **没有 Nyquist 上限**。误差律本身就是在目标间距上测的，`k_max` 以上的窗口混叠已经算在
  `C` 里；再叠一道 `0.75π/k_max` 会让高阶样条"白升"——精度上去了网格却没变。所以
  **高阶样条提升的是精度，不是网格**，缺省 B-spline-8 的 1.428 Å 比旧规则的 0.7764 Å 粗。
- 已知代价：缺省 B-spline 路径的网格因此**比历史版本粗 ~1.8×**（结果随模型/冻结产物
  一起变；`pme_spacing` 可显式钉死旧网格）。
- CubeS₂ 上限来自方差扣除：`Δ > σ_min/(√2 ξ)` 后网格上的方差不再恒正。

**(6) 交给算子** —— `deepmd/pt_expt/long_range/latent_charge.py:194` 把 `self.mesh_spacing()`
作为 `spacing` 传给算子；算子（`source/op/pt/long_range/midtown_sum.cu:154`）只做

```cpp
n = next_factorable(ceil(L / spacing))    // 只允许因子 2/3/5
```

**网格随当前盒长每步重算，spacing 固定。** 非正间距会被拒绝（与 B-spline 后端
`mesh_sum.cu` 一致），不再有第二份 C++ 精度规则。

### 1.2 实测数值（参考核 `ckpt-250000`）

`σ_min = 2.0000 Å`，`k_max = 3.0349 Å⁻¹`，盒长 `122.31 Å`，目标 `1e-4`（能量口径；旧规则 0.75π/k_max = 0.7764 Å 已不再生效）：

| backend | order | `(C_E,p_E)` | `Δ/σ_min` | `Δ` (Å) | 网格 | 格点数 |
|---|---|---|---|---|---|---|
| bspline | 4 | (3.33e-2, 4.0) | 0.234 | 0.468 | 262³ | 18.0 M |
| bspline | 5 | (1.53e-2, 5.0) | 0.366 | 0.731 | 168³ | 4.7 M |
| bspline | 6 | (7.01e-3, 6.0) | 0.493 | 0.985 | 125³ | 1.95 M |
| bspline | 7 | (3.22e-3, 7.0) | 0.609 | 1.218 | 101³ | 1.03 M |
| bspline | **8（缺省）** | (1.48e-3, 8.0) | 0.714 | 1.428 | **86³** | **0.64 M** |
| midtown | 4 | (3.81e-3, 4.0) | 0.403 | 0.805 | **152³** | **3.5 M** |
| midtown | 6 | (4.03e-3, 6.0) | 0.540 | 1.080 | 114³ | 1.48 M |
| midtown | 4 (`ik`) | (3.81e-3, 4.0) | 0.403 | 0.805 | 152³ | 3.5 M |
| midtown | 6 (`ik`) | (4.03e-3, 6.0) | 0.540 | 1.080 | 114³ | 1.48 M |
| quads | 4 | (3.39e-3, 4.0) | 0.414 | 0.829 | 148³ | 3.2 M |
| quads | 6 | (1.81e-3, 6.0) | 0.617 | 1.234 | 100³ | 1.0 M |
| quads | 6 (`ik`) | (1.81e-3, 6.0) | 0.617 | 1.234 | 100³ | 1.0 M |

**读法**：同一个 `1e-4` **能量**精度下，网格由能量律唯一决定，**AD 与 ik 给同一网格**。
CubeS₂ 的能量常数比 B-spline 小 2–9×，所以同样精度下网格更粗：4 阶 152³ vs 262³（省 5.1×），
6 阶 114³ vs 125³（省 1.3×）。注意这是**能量**口径——AD 力在这些网格下会超过 1e-4（力指数
低一阶，最差 CubeS₂-4 ~60×）；需要力保证请看 §5 陷阱 3 或反解 `FORCE_RULE`。

### 1.3 冻结与训练

- **冻结**：`deepmd/pt_expt/utils/serialization.py:2264` 调 `use_mesh_evaluation()`，
  把 `pme_spacing = mesh_spacing()` **写进模型**。此后 `mesh_spacing()` 直接返回常数，
  与冻结核一致。
- **训练（eager）**：每步按当前 `log_width` 现算，`σ_min` 漂移会让网格变化并触发绿函数重建。
  这是「训练时网格跟着参数动」的预期行为，但意味着**训练吞吐会随 `σ_min` 变化**。

---

## 2. B 链：LAMMPS init 时 C++ 自算

入口 `SOGKSpace::ensure_fft_plan`（`source/lmp/sog.cpp:963`，网格块在 `:981-1085`），
`grid_method == 0`（缺省，`sog.cpp:398`）走 SOG-bandwidth 分支。

### 2.1 逐步

1. **目标精度**：`phi_accuracy` 关键字（`sog.cpp:606-612`），缺省 `1e-4`（`:1057`）。
   显式 `phi_max` 关键字可直接跳过整个反解（`:991`）。
2. **取 σ_min**：`sqrt(min(bandwidth))`，`bandwidth` 由 `finalize_kernel_parameters`
   在 init 时填好（`:1008-1012`）。
3. **选常数**（`:1019-1056`），三分支：
   - `is_quads`（QuadS，Form A 只剩散布误差）：`C = 2e-4`，`p = 6 / 4`
   - `spline_type == 0`（B-spline）：读 `bspline_order`，同上表 5 档
   - 其余（CubeS₂）：`C = 0.196 / 0.152`（阶 4 / 6），`p = 3 / 5`
4. **反解**：`ds = (ε/C)^(1/p)`（`:1058`）。
5. **夹逼**：`delta_want = min(ds·σ_min, 0.95·σ_min/(√2 ξ))`（`:1059-1061`）。
6. **转 φ_max**：`phi_max = delta_want / rcut`，`rcut` 取 `p_cutoff`，缺则取 `n_dl`（`:975-977`）。
7. **定网格**：`n = max(8, ceil(L / (phi_max·rcut)))`（`:1065-1075`）。
8. **只在 init 算一次**（`mesh_ready` 门控）。NPT 下盒变但 `nx/ny/nz` 不变，
   只有 `box_tol` 控制绿函数重建。

### 2.2 与 A 链的差异

| | A 链（lr-pt 图内） | B 链（LR LAMMPS） |
|---|---|---|
| **精度口径** | **能量**（`ENERGY_RULE`，k 空间归一；AD/ik 同网格） | **力**（`FORCE_RULE`，AD） |
| 反解位置 | Python `grid_rule.py` | C++ `sog.cpp:1019-1075` |
| 精度开关 | `pme_accuracy` 配置项 | `phi_accuracy` 关键字 |
| Nyquist 上限 `0.75π/k_max` | **无**（误差律直接给网格） | **无**（`:1497-1501` 事后检查网格能否解析 `k_max`） |
| QuadS 常数 | 无（图内无 QuadS 后端） | 有 `(2e-4, 6/4)` |
| 网格是否随盒走 | 每步按当前盒长重算 | init 固定 |
| 因子限制 | 只允许 2/3/5 | 无（cuFFT 直接吃 `ceil`） |

> **注意**：本次「统一为能量口径」只改了 A 链（`grid_rule.spacing()`）；B 链的 LAMMPS
> kspace 仍按力定网格（`sog.cpp` 注释里的 "force is the binding constraint" 依然成立）。
> 因此两链现在对同一模型/精度给**不同**网格——`sog.cpp:1066` 那句「so both chains size
> the same grid」已不再成立，属待同步项。

**A 链为什么排除因子 7**：`L = 122.31 Å`、`spacing = 0.5` 时请求 245 点，
2-3-5 取 250 = 2·5³，允许 7 则接受 245 = 5·7²。实测 245³ 的 cuFFT 走
`regular_fft_factor<49>` + `regular_fft_factor<5>` + `vector_fft`，三次变换 1.42 ms
加前后处理 0.39 ms = **1.81 ms**，而 250³ 只要 0.59 ms。radix-7 通道让更小的网格贵 3 倍。

---

## 3. Form A 在哪、怎么开

| | A 链 | B 链 |
|---|---|---|
| 实现 | `sog_gpu_green_form_a()`（`midtown_sum.cu:605`） | `SOGKSpace::apply_form_a_green()`（`sog.cpp:1724`） |
| 开关 | `SOG_MIDTOWN_FORM_A`，**未设 = 开**（`midtown_sum.cu:560`） | `SOG_FORM_A`，**未设 = 开**（`sog.cpp:1716`） |
| 角向求积 | `SOG_MIDTOWN_FORM_A_NTH`，缺省 8 | `SOG_FORM_A_NTH` + `SOG_FORM_A_FLOOR` |
| 覆盖 | 仅复制网格路径；`SOG_MIDTOWN_SLAB` / `DIST` 显式拒绝 | 全部 |
| 例外 | 三斜盒无实测窗口符号 → 留在 Form B | 同 |

Form A 把 `C_infl` 置零，所以 `C` 只剩分配项。对力而言 Form A 与 Form B 的差别在
1–4% 量级（见 `docs/MESH_ERROR_MODEL_AND_FORM_A.md` §4.6），**常数表对两者共用**。

---

## 4. 验证与回归

- **规则单测**：`source/tests/common/dpmodel/test_grid_rule.py`
  钉住常数表、反解、CubeS₂ 有效性上限、参考核的具体网格（86³ / 125³ / 114³），
  以及「等能量精度下 midtown 需要更少格点」这一不变式。
- **算子数值**：`source/tests/pt_expt/kernels/test_long_range_sum.py`（50 例）、
  `test_mesh_backend.py`（4 例）。
- **模型级**：`source/tests/common/dpmodel/test_long_range.py`（61 例）全过；
  `source/tests/pt_expt/model/test_long_range.py` 中 `.pte` 冻结用例全过，
  `.pt2`（AOTI 包）用例在本机 dlopen 失败——Inductor 生成的 wrapper 需要
  `GLIBCXX_3.4.30`，而系统 `libstdc++` 没有、机器上也没有 gcc13 toolset。
  这是环境限制，与规则无关（同一条比较的 `.pte` 版本是过的）。
- **跨链一致性**：`grid_rule.py` 与 `sog.cpp` 的常数必须逐项相同；本次已核对。
- **生产冻结**：`--compress`（canonical 下界）的命令、验证与排错见
  [`FREEZE_SOP.md`](FREEZE_SOP.md)。

复算本文表格：

```python
from deepmd.dpmodel.long_range import grid_rule
import math
sigma, kmax = 2.0000, 3.0349
for be, o in (("bspline", 4), ("bspline", 5), ("bspline", 6), ("bspline", 7),
              ("bspline", 8), ("midtown", 4), ("midtown", 6)):
    d = grid_rule.spacing(be, o, sigma, kmax, 1e-4)
    print(f"{be:9s} {o}  d={d:.4f} A  n={math.ceil(122.31/d)}^3")
```

---

## 5. 已知陷阱

1. **`pme_order: 8` + `mesh_backend: midtown` 实际是 6 阶**。想比精度必须显式对齐阶数。
2. **等网格 ≠ 等精度**。同网格下 CubeS₂ 比 B-spline 能量更准；能量口径 1e-4 下
   CubeS₂-6 需 114³、B-spline-6 需 125³（CubeS₂ 省 1.3× 格点）。比较必须说明口径
   （能量 vs 力）。
3. **能量口径下 AD 力不再受 1e-4 保证**。CubeS₂-4 的能量网格是 152³（3.5 M），但其 AD 力
   误差 ~6×10⁻³（60× 目标）。需要力保证时改 `force_mode: ik`（力约束 227³）或反解
   `FORCE_RULE`。
4. **两条链都没有 Nyquist 上限**。A 链由误差律直接给网格（目标间距上的混叠已计入 `C`）；
   B 链靠 init 后的显式检查兜底。
5. **常数是拟合值**。换核（`σ_min` 变化超过 ~2×）或换体系请重新标定，
   不要假设 `1e-4` 在别处仍是 1e-4。
