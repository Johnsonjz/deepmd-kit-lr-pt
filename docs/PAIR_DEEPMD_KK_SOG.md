# `pair_style deepmd/kk` 如何调用 SOG 模型

本文回答一个问题：**LAMMPS 里 `pair_style deepmd/kk` 到底是怎么把长程 SOG 项算出来的**，
并给出可直接改两条路径就能跑的标准输入与命令。

代码全部在 `/data/home/public/jiangzhen/dp/deepmd-kit-LR`（`feat/sezm-sog` 分支），
标准输入在 `examples/lammps/deepmd_kk_sog/`。

> **一句话**：`deepmd/kk` 本身**不算长程**。它做两件事——算短程能量/力，以及把模型的
> **隐变量电荷写成 `atom->q`**（在 device 上，不过主机）；长程由 `kspace_style sog/kk`
> 读同一个电荷数组、在同一批 device 数组上完成；`fix sog/response` 再补上
> **电荷随坐标变化**的那一项力。三者的分工是硬的：缺 `fix` 力不守恒，缺 `kspace`
> 就没有长程（除非走 §3 的图内链）。

---

## 1. 两条部署链

模型冻结产物（`.pt2`）里可能带三种长程相关的东西，`deepmd/kk` 用哪个由输入决定：

| 产物里的东西 | 谁用它 | 怎么开 |
|---|---|---|
| `latent_charge` 输出（一定在） | `sog/kk` 的电荷来源 | `latent_charge_to_q yes` |
| `model/extra/latent_charge_response.pt2` | `fix sog/response` | `fix ... sog/response model <同一 .pt2>` |
| `model/extra/sog_energy_only.pt2` + `sog_response_combined.pt2` | 融合链（可选）：pair 只算能量、`fix` 做一次联合反向 | 自动（`latent_charge_to_q yes` 时尝试启用，缺了就不融合） |
| `model/extra/sog_full_lr.pt2` / `sog_full_lr_wc.pt2` | `deepmd/kk` 图内长程 | `full_lr yes` |

### 链 A：pair + kspace + fix（本文主线）

```
  ┌─────────────── device ────────────────────────────────────────────┐
  │ deepmd/kk:  隐变量 q  ──► atom->q  (Kokkos kernel, 无 host 往返)  │
  │ kspace sog/kk ── 读 atom->q                                       │
  │      spread → cuFFT(R2C) → 绿函数 → cuFFT⁻¹ → gather               │
  │      → 力写回 Kokkos 力视图；v_i = ∂E_k/∂q_i 挂起待用              │
  │ fix sog/response ── 拿 v_i 做 ∂q/∂r 的一次反向，得 ΔF, Δvirial     │
  └───────────────────────────────────────────────────────────────────┘
```

*k-space 网格与力都在 device 上；只有能量/维里归约（160 B）和 `fix` 需要的 `v_i`
（nlocal 个 double）过总线。*

### 链 B：图内长程（`full_lr`）

```
deepmd/kk  (一次前向就把 SR + LR 的 E/F/virial 全算出来) ──► 结果
kspace_style none
```

不需要 `sog/kk`，也不需要 `fix sog/response`：电荷响应项在导出图里已经算进去了。
**两条链不能同时用**（见 §7 错误 2）。

---

## 2. 链 A 的关键字

### 2.1 `pair_style deepmd/kk`

沿用 `pair deepmd` 的全部参数（模型文件、`pair_coeff * * <type names>`、`out_freq`、
`fparam` …），另加三个长程相关开关：

| 关键字 | 作用 |
|---|---|
| `latent_charge_to_q yes` | 把模型的 `latent_charge` 通道 0 在 **device 上**写进 `atom->q`。没有它，`sog/kk` 读到的是数据文件里的 `q`（通常是 0），长程项直接是错数。要求 `atom_style charge`（Kokkos 下 `charge/kk`）、**单模型**、**单通道**。 |
| `full_lr no` | **显式关掉图内长程**。带 `sog_full_lr` 图谱的产物，在**没有** `kspace_style sog` 时会把 `full_lr` **自动打开**（`source/lmp/pair_deepmd_kokkos.cpp:239`），而 `latent_charge_to_q` 的写回被 `if (latent_charge_to_q && !full_lr)` 跳过——于是 q 恒为 0、长程算的是错数。链 A 里写死 `full_lr no` 最稳。 |
| `full_lr yes` | 链 B。仅 Kokkos pair 支持；与 SOG kspace 同时出现直接报错。 |

> **`deepmd/kk` 与 `deepmd` 的区别**：前者把坐标/电荷/力留在 Kokkos device 数组里，
> 是链 A 能全程不落主机的前提；后者每一步要 device→host→device 搬 5 个 double/原子。
> 数值上两者一致。

### 2.2 `kspace_style sog/kk`

两种写法：**推荐显式核**（`amp` + `bandwidth`），或 legacy 头部 `accuracy b sigma M [n_dl]`。
生产一律用显式核——核必须与训练出来的那个一致。

```lammps
kspace_style sog/kk amp <M 个> bandwidth <M 个> remove_self_interaction yes n_dl 1.0 use_gpu yes
```

| 关键字 | 含义 |
|---|---|
| `amp a_0 … a_{M-1}` | 核的系数，C++ 口径：`K(k²) = Σ_m amp_m exp(-½·bandwidth_m·k²)` |
| `bandwidth b_0 … b_{M-1}` | **`σ_m²`（平方宽度）**，不是 `σ_m` |
| `remove_self_interaction yes` | 扣掉 `½·Σqᵢ²·g(0)`。模型侧无条件扣，所以这里也必须是 `yes` |
| `n_dl 1.0` | 只用显式核时它是 legacy 网格参数（k_max = 2π/n_dl），可省略；为了和训练配置一致通常写上 |
| `use_gpu yes` | **打开 cuFFT device 路径**。不写就静默回落到主机求解器（结果对、慢约 2×） |
| `spline cubes2_6` | 样条族，默认 `cubes2_4`；可选 `cubes2_4/6`、`quads_4/6`、`bspline_4/5/6` |
| `phi_accuracy ε` | 网格精度目标（默认按阶数取），控制网格粗细 |
| `grid_method sog_bandwidth` | 默认。`pppm_legacy` 是旧网格规则 |
| `box_tol 1e-3` | NPT 下盒漂移小于此比例时跳过绿函数重建（`sog/kk` 每步都会 `setup()`，不跳就是每步一次 3D 绿函数重建） |
| `direct yes [kmax]` | 直接 k 求和（无网格），只用于小体系对拍 |

**`sog/kk` 只支持正交盒、单电荷通道、全周期**（`source/lmp/sog.cpp:880-895`）。

### 2.3 `fix sog/response`

```lammps
fix ID group sog/response model <同一个 .pt2> [pair_deepmd_index 0]
```

- 作用：把 `q = q(r)` 的响应补上，即 `ΔF_i = Σ_j (∂q_j/∂r_i)·v_j`，
  `v_j = ∂E_k/∂q_j` 由 SOG 求解器给（`sog_kspace->set_want_potential(true)`）。
- **产物必须带 `model/extra/latent_charge_response.pt2`**，否则第一步就抛
  `this .pt2 ships no charge-response`。
- 不装它，短程力是对的、长程力少一项：单点看不出来，NVE 下表现为能量漂移
  （实测 1.915 → 0.0040 eV / 那条轨迹）。
- 只支持 `kspace_style sog` / `sog/kk`；与 `full_lr` 互斥。
- `SOG_KK_DEVICE_RESPONSE=1` 时走 device 响应路径（与 `sog/kk` 同一套 device 数组）。

---

## 3. 链 B：`full_lr yes`

```lammps
pair_style      deepmd/kk ${MODEL_FULL_LR} full_lr yes
pair_coeff      * * O H
kspace_style    none
```

- 要求产物带 `model/extra/sog_full_lr.pt2`（单卡）与 `sog_full_lr_wc.pt2`（`np>1`，
  带通信的图）。只有前者时 `np>1` 会拒绝。
- 不支持 `--compress` 产出的压缩下界（会静默报 0 能量）。
- 多卡：`deepmd/kk` + 图内长程按 `np` 分域；电荷中和在 with-comm 图里做，等价主机求解器。

---

## 4. 标准输入

`examples/lammps/deepmd_kk_sog/in.deepmd_kk_sog.lammps`（链 A，全文带注释）。核心 5 行：

```lammps
atom_style      charge/kk
pair_style      deepmd/kk ${MODEL} latent_charge_to_q yes full_lr no
pair_coeff      * * ${TYPES}
kspace_style    sog/kk ${KERNEL} use_gpu yes
fix             resp all sog/response model ${MODEL}
```

链 B 的最小输入：

```lammps
units           metal
boundary        p p p
atom_style      charge/kk          # 电荷不是必需的，但和已测通的输入一致
atom_modify     map yes
read_data       ${DATA}
mass            1 16
mass            2 2
pair_style      deepmd/kk ${MODEL_FULL_LR} full_lr yes
pair_coeff      * * O H
kspace_style    none
fix             1 all nve/kk
run_style       verlet/kk
timestep        0.0005
thermo_style    custom step temp pe press vol
run             100000
```

---

## 5. 运行命令

```bash
# 环境（每次都要）
export LAMMPS_PLUGIN_PATH=/tmp/lmpkk_plugin        # 含 deepmd_lmp.so（或 *plugin.so 链接）
export DP_BACKEND_PLUGIN_PATH=<dp_install>/deepmd/lib
export LD_LIBRARY_PATH=<conda>/lib:<torch_nv>/cuda_runtime/lib:<torch_nv>/cuda_cupti/lib:<torch_nv>/nvjitlink/lib:<lmp_install>/lib:<dp_install>/deepmd/lib

# 单卡
lmp -k on g 1 -in in.deepmd_kk_sog.lammps

# 多卡：一个 rank 一张卡，按 LOCAL rank 绑定（别用全局 rank）
mpirun -np 4 ./run.sh -np 4 in.deepmd_kk_sog.lammps
```

`examples/lammps/deepmd_kk_sog/run.sh` 把上面这些包好了，只改 4 个路径变量即可。

**已验证**：本文的标准输入在 192 原子水盒上 `run 0` 跑通，`pe = -29971.346 eV`
（与主机链 `pair deepmd` + `kspace sog` 在 np=1 与 np=2 上的值逐位相同）。
**多卡的现状比单卡差**——见 §9，np>1 目前请用主机链。

要点：

- **`-k on g 1`**：每个进程一张卡。多 rank 时 `CUDA_VISIBLE_DEVICES` 必须按
  `OMPI_COMM_WORLD_LOCAL_RANK` 设（`run.sh` 已处理）。
- **`LD_LIBRARY_PATH` 顺序有意义**：conda 的 `libstdc++` 要在系统之前（AOTI wrapper
  需要 GLIBCXX_3.4.30），torch 自带的 CUDA 12.6 运行时要在系统 12.4 之前。
- 插件按 `LAMMPS_PLUGIN_PATH` 自动加载；也可以写 `plugin load <path>` 显式加载。
- `use_gpu yes` 缺省是 `no`——**结果对但会掉到主机求解器**，日志里 `SOG/KK init` 的
  `device_ok` 会显示（`SOG_KK_DEBUG=1`）。

---

## 6. `amp` / `bandwidth` 从哪来

**不要手推**。它们是训练后的核参数，训练会把宽度从初值拉走。

**LR 谱系（`fitting=sog_energy`）**：模型自带权威导出

```python
from deepmd.pt.model.model import get_model
model = get_model(cfg["model"]); model.load_state_dict(ckpt["model"])
p = model.get_fitting_net().lammps_kernel_params()
# p = {"b","sigma","M","n_dl","remove_self_interaction","amp":[...],"bandwidth":[...]}
print("kspace_style sog/kk amp", *p["amp"], "bandwidth", *p["bandwidth"],
      "remove_self_interaction yes n_dl 1.0 use_gpu yes")
```

或者直接用 `dp_example/water/lmp/scripts/extract_sog_params.py <config.json> <ckpt.pt> out.json --include out.inc`，
它会写出 `kspace_style ...` 那一行。

**lr-pt 谱系（`latent_charge`）**：核是 `long_range.kernel` 的 `amplitude a_k` / `width σ_k`，
换算到 C++ 口径：

```python
# K(r) = Σ a_k exp(-r²/(2σ_k²))  →  K(k²) = Σ a_k (2πσ_k²)^{3/2} exp(-½σ_k²k²)
amp       = [a * (2*np.pi)**1.5 * s**3 for a, s in zip(amplitude, width)]
bandwidth = [s**2 for s in width]
```

> 两种口径的对照记在 `source/lmp/sog.cpp:740-790`（外部核分支）与
> `deepmd/dpmodel/long_range/grid_rule.py` 的注释里。

---

## 7. 部署约束与错误对照

| 约束 | 原因 / 现象 |
|---|---|
| **正交盒** | `sog` 拒绝带倾斜的 cell（`domain->triclinic` 且 tilt≠0）。三斜请用 `rbsog` 系列。 |
| **全周期** | `sog` 要求 `p p p`。 |
| **单电荷通道** | `sog/kk` 显式报错：多通道请用 `kspace_style sog`（主机）。 |
| **`latent_charge_to_q` 要 `atom_style charge`** | 否则 `atom->q` 不存在，直接报错。 |
| **多卡不省显存** | `sog/kk` 多 rank 是**复制网格**：每个 rank spread 自己的原子 → `MPI_Allreduce` → 每个 rank 各自跑完整 FFT。所以每卡显存 = 全局网格，rank 数只分摊原子与近程计算。要按 rank 分网格，用 lr-pt 侧的 z-slab（`docs/SLAB_DEPLOYMENT.md`）。 |
| **多卡目前只能用主机链** | `fix sog/response` 的 device 响应在 np>1 段错误，且 replicated 路径的 k-space virial 被乘 np——见 §9。 |
| **`fix sog/response` 与 `full_lr` 互斥** | 两者都在补响应项，同时开会重复计数。 |
| **压缩下界（`--compress`）不支持 `full_lr`** | 压缩图不上报 SR-only 能量，会静默报 0。链 A 不受影响。 |

常见报错 → 原因：

| 报错 | 原因 |
|---|---|
| `full_lr cannot be combined with a SOG kspace style` | 产物带图内长程且没写 `full_lr no` |
| `latent_charge_to_q requires an atom style with charge` | 用了 `atom_style atomic` |
| `latent_charge_to_q requires a single model` | 多模型 ensemble |
| `kspace_style sog/kk supports a single latent-charge channel` | 多通道模型 |
| `fix sog/response requires kspace_style sog` | 没开 kspace（或写了 `full_lr yes`） |
| `this .pt2 ships no charge-response` | 产物缺少 response extra，需重新冻结 |
| q 恒为 0、能量明显偏小 | `full_lr` 被自动打开，写 `full_lr no` |
| 每步 60+ ms 的慢 | `kspace_style` 没写 `use_gpu yes`，掉到主机求解器 |
| NPT 每步偏慢 | 绿函数每步重建（`fix npt` 每步调 `kspace->setup()`）；调 `box_tol` |

---

## 8. 校验清单

上线一个新模型，按顺序过一遍：

1. **`run 0` 对拍**：同一构型，LAMMPS 的 `pe` / 力 / 压强 vs
   `deepmd.infer.DeepPot(ckpt)` 的前向结果。链 A 在训练盒上应当到
   ~0.1 meV/atom、~1e-3 eV/Å（`sog_work/docs/MD_DEPLOYMENT_PARITY.md`）。
2. **`q` 非零**：`dump ... custom 1 q.dump id type q`，与模型 `latent_charge` 逐原子比。
3. **NVE 能量守恒**：不加恒温器跑 10 ps，`etotal` 漂移应 ≲ 1e-2 eV；有量级漂移先查
   `fix sog/response` 是否真的生效。
4. **压强量级**：`thermo_style custom step temp pe press vol`，金属单位下压强是 **bar**。
5. **`device_ok`**：`SOG_KK_DEBUG=1` 确认走的是 device 路径。
6. **多卡一致**：`pe` 应逐位相同；**压强目前不会**（§9.2），在修好之前不要拿多卡的
   `press` 做结论。

---

## 9. 已知问题（2026-11-10 实测，附一行修复）

三条都在 192 原子水盒（`sog_work/lmp/lmp150k/water_charge.lmp`，12.44³ Å）、
`dpa4_sog_150k_fused.pt2` 与同一套 `amp/bandwidth` 上复现。

### 9.1 多卡 virial 被乘了 np —— `sog/kk` 的 replicated 路径

`sog_kokkos.cpp::compute_device_replicated` 每个 rank 都拿到**全局**势场，
于是把**全局 virial** 写进 `KSpace::virial`：

```cpp
energy = e;
for (int j = 0; j < 6; ++j) virial[j] = vv[j];   // ← 每个 rank 都是全局值
```

但 LAMMPS 对这两者的约定**不同**：`energy` 不被归约（`thermo.cpp:2038` 直接读，
所以每 rank 持全局值是对的），`virial` 却被 `ComputePressure` **MPI 求和**
（`compute_pressure.cpp:342`）。结果 k-space 压强 ×np：

| np | pe (eV) | press (bar) | 说明 |
|---|---|---|---|
| 1 | -29971.346 | **+3662.48** | |
| 2 | -29971.346 | **-7558.33** | 与 np=1 差 −11220.8 bar，正好等于多算一份 k-space virial |

**修复**（一行）：

```cpp
const double vshare = 1.0 / static_cast<double>(comm->nprocs);
for (int j = 0; j < 6; ++j) virial[j] = vv[j] * vshare;
```

主机链（`pair deepmd` + `kspace sog`，PPPM 式分布式网格）没有这个问题：
np=1 / np=2 实测 `pe = -29971.346`、`press = -9100.665 / -9100.667`，一致到 1e-3 bar。

### 9.2 `fix sog/response` 的 device 响应在 np>1 段错误

`deepmd/kk` + `sog/kk` + `fix sog/response` 在 np=1 正常；np=2 直接 SIGABRT，
栈在 `PairDeepMDKokkos<>::compute_charge_response_device` →
`DeepPotPTExpt::compute_charge_response_edges_gpu`（`pair_deepmd_kokkos.cpp:893`）。
去掉 `fix` 只留 pair+kspace，np=2/4 正常。

根因上看，device 响应分支在执行后**直接 return，不做 `comm->reverse_comm(this)`**，
而主机分支会做；也就是说这条路径本来就是按 np=1 写的，只是没有 np>1 的护栏。

**现状建议**：np>1 用主机链

```lammps
pair_style   deepmd ${MODEL} latent_charge_to_q yes
kspace_style sog  ${KERNEL}
fix          resp all sog/response model ${MODEL}
```

（`energy`、`force` 与 device 链一致；只是每步多一次 device↔host 往返。）

### 9.3 device 响应与主机响应的压强不一致（np=1）

同一份输入、同一个 np=1：

| 配置 | pe (eV) | press (bar) |
|---|---|---|
| device pair + `sog/kk`，**无** fix | -29971.346 | +3662.48 |
| device pair + `sog/kk` + fix（device 响应） | -29971.346 | **-4995.93** |
| 主机 pair + `sog` + fix（主机响应） | -29971.346 | **-9100.67** |

能量完全相同，`fix` 贡献的 virial 却差 ~4100 bar（device 补 −8658 bar、主机补
−12763 bar）。已知 `virial_global_flag` / `thermo_virial` 都已置 1
（`fix_sog_response.cpp:84-89`），所以不是"virial 没进压强"，而是**两条响应路径的
virial 口径不同**，尚未定位。**在查清之前，多卡请用主机链、单卡用哪条都行但不要跨链比压强。**
