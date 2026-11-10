# `pair_style deepmd/kk` 如何调用 SOG 模型

本文用到的两棵树**完全分开叙述**，因为它们的分工是硬的：

| | 树 | 提供什么 |
|---|---|---|
| **第一部分** | `/data/home/public/jiangzhen/dp/deepmd-kit-LR`（分支 `feat/sezm-sog`） | **LAMMPS 侧的全部代码**：`pair deepmd/kk`、`kspace sog/kk`、`fix sog/response`、两个 plugin 构建 |
| **第二部分** | `/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt`（分支 `feat/long-range-pt`） | **模型与训练侧**：`latent_charge` 长程算子、网格误差律、冻结成 `.pt2` |

两棵树**不能互相替换**，也不能把一方的文件名套到另一方：

- 第二部分**没有** `sog.cpp` / `sog_kokkos.cpp` / `fix_sog_response.cpp`，它的
  `pair_deepmd*` 里**没有** `full_lr`、`latent_charge_to_q` 这两个关键字（`grep` 为零）。
  它冻结的模型把长程**整段放进图**，由 `pair deepmd` 直接调用。
- 第一部分（`deepmd-kit-LR`）**没有** `deepmd/dpmodel/long_range/`（`grid_rule.py`、
  `latent_charge.py` 所在），也没有 `deepmd/pt_expt/long_range/`；它不做训练，也不做冻结。
- 这两者都在第二部分：`deepmd-kit-LR-lr-pt/deepmd/pt_expt/long_range/`
  （`base_long_range.py`、`kernel.py`、`latent_charge.py`）与
  `deepmd-kit-LR-lr-pt/deepmd/dpmodel/long_range/`。

两棵树唯一共享的东西是 **`deepmd/lib/libdeepmd_cc.so` + `libdeepmd_backend_ptexpt.so`**
这两个模型加载库（两棵树各自都会编译），LAMMPS 通过 `DP_BACKEND_PLUGIN_PATH` 选一个。

---

# 第一部分　`deepmd-kit-LR`：LAMMPS 的 SOG 侧

## 1. 这棵树里有什么

| 文件 | 作用 |
|---|---|
| `source/lmp/pair_deepmd.cpp` / `.h` | 主机版 `pair deepmd`；`latent_charge_to_q`、`full_lr` 的参数解析在这里 |
| `source/lmp/pair_deepmd_kokkos.cpp` / `.h` | `pair deepmd/kk`（device），把坐标/电荷/力留在 Kokkos 数组里 |
| `source/lmp/sog.cpp` / `.h` | `kspace sog`：SOG 粒子网格求解器（主机），PPPM 式分布式网格 |
| `source/lmp/sog_kokkos.cpp` / `.h` | `kspace sog/kk`：device 版，只换数据通路 |
| `source/lmp/sog_gpu.cu` / `.cuh` | 原始 CUDA + cuFFT 的 spread / green / gather |
| `source/lmp/fix_sog_response.cpp` / `.h` | `fix sog/response`：电荷响应项 |
| `source/lmp/plugin_kokkos/` | 独立的 torch-free CMake 工程，编出 Kokkos plugin |
| `source/lmp/plugin/` | 常规 plugin（`deepmdplugin.cpp` 注册所有 style） |

三条 style 的注册点：`PairStyle(deepmd/kk, …)` 在 `pair_deepmd_kokkos.h:11`，
`KSpaceStyle(sog, SOGKSpace)` 在 `sog.h:21`，`sog/kk` 与 `fix sog/response` 在
`plugin/deepmdplugin.cpp` 里显式注册（runtime plugin 不消费 `*_CLASS` 宏）。

## 2. 两条部署链

> 本节只管**第一部分这棵树自己的产物**（`fitting=sog_energy` 谱系）。第二部分的
> `latent_charge` 产物怎么进来，见 §10。

模型产物（`.pt2`）里可能带四种长程相关的东西，用哪个由输入决定：

| 产物里的条目 | 谁用它 | 怎么开 |
|---|---|---|
| `latent_charge` 输出 | `sog/kk` 的电荷来源 | `latent_charge_to_q yes` |
| `model/extra/latent_charge_response.pt2` | `fix sog/response` | `fix … sog/response model <同一 .pt2>` |
| `model/extra/sog_energy_only.pt2` + `sog_response_combined.pt2` | 融合链：pair 只算能量、fix 做一次联合反向 | 自动（`latent_charge_to_q yes` 时尝试启用，缺了就不融合） |
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

`deepmd/kk` **不算长程**。它做两件事：算短程能量/力，以及把模型的隐变量电荷写进
`atom->q`。长程由 `sog/kk` 读同一个电荷数组完成，`fix sog/response` 再补上
"电荷随坐标变化"的那一项。缺 `fix` 力不守恒；缺 `kspace` 就没有长程（除非走链 B）。

### 链 B：图内长程（`full_lr`）

```
deepmd/kk  (一次前向把 SR + LR 的 E/F/virial 全算出来) ──► 结果
kspace_style none
```

不需要 `sog/kk`，也不需要 `fix sog/response`。**两条链不能同时用**（与 `kspace_style
sog` 同时写 `full_lr yes` 直接报错）。

## 3. 关键字参考

### 3.1 `pair_style deepmd/kk`

沿用 `pair deepmd` 的全部参数（模型文件、`pair_coeff * * <type names>`、`out_freq`、
`fparam` …），另加两个长程开关：

| 关键字 | 作用 |
|---|---|
| `latent_charge_to_q yes` | 把模型的 `latent_charge` 通道 0 在 **device 上**写进 `atom->q`。没有它，`sog/kk` 读到的是数据文件里的 `q`（通常为 0），长程项直接是错数。要求 `atom_style charge`（Kokkos 下 `charge/kk`）、**单模型**、**单通道**。 |
| `full_lr no` | 显式关掉图内长程。带 `sog_full_lr` 图谱的产物，在**没有** `kspace_style sog` 时会把 `full_lr` 自动打开（`pair_deepmd_kokkos.cpp:239`），而 `latent_charge_to_q` 的写回被 `if (latent_charge_to_q && !full_lr)` 跳过——于是 q 恒为 0。链 A 写死 `full_lr no` 最稳。 |
| `full_lr yes` | 链 B。仅 Kokkos pair 支持；与 SOG kspace 同时出现直接报错。 |

> `deepmd/kk` 与 `deepmd` 的区别：前者把坐标/电荷/力留在 Kokkos device 数组里，是链 A
> 能全程不落主机的前提；后者每一步要 device→host→device 搬 5 个 double/原子。数值一致。

### 3.2 `kspace_style sog/kk`

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
| `n_dl 1.0` | 只用显式核时是 legacy 网格参数（k_max = 2π/n_dl），可省略；为与训练配置一致通常写上 |
| `use_gpu yes` | **打开 cuFFT device 路径**。不写就静默回落主机求解器（结果对、慢约 2×） |
| `spline cubes2_6` | 样条族，默认 `cubes2_4`；可选 `cubes2_4/6`、`quads_4/6`、`bspline_4/5/6` |
| `phi_accuracy ε` | 网格精度目标（默认按阶数取），控制网格粗细 |
| `grid_method sog_bandwidth` | 默认；`pppm_legacy` 是旧网格规则 |
| `box_tol 1e-3` | NPT 下盒漂移小于此比例时跳过绿函数重建（`sog/kk` 每步都 `setup()`） |
| `direct yes [kmax]` | 直接 k 求和（无网格），只用于小体系对拍 |

**`sog/kk` 只支持正交盒、单电荷通道、全周期**（`sog.cpp:880-895`）。

### 3.3 `fix sog/response`

```lammps
fix ID group sog/response model <同一个 .pt2> [pair_deepmd_index 0]
```

- 作用：补上 `q = q(r)` 的响应，`ΔF_i = Σ_j (∂q_j/∂r_i)·v_j`，其中
  `v_j = ∂E_k/∂q_j` 由 SOG 求解器给（`sog_kspace->set_want_potential(true)`）。
- **产物必须带 `model/extra/latent_charge_response.pt2`**，否则第一步抛
  `this .pt2 ships no charge-response`。
- 不装它，短程力对、长程力少一项：单点看不出，NVE 下表现为能量漂移
  （实测 1.915 → 0.0040 eV）。
- 只支持 `kspace_style sog` / `sog/kk`；与 `full_lr` 互斥。
- `SOG_KK_DEVICE_RESPONSE=1` 时走 device 响应路径（与 `sog/kk` 同一套 device 数组）。

## 4. 标准输入与运行

### 4.1 输入

标准输入随本文一起放在**第二棵树的 checkout** 里（本文所在的位置），因为它驱动的是第一棵树的
plugin；运行时只依赖第一棵树的 `deepmd_lmp.so` + 某棵树的 `deepmd/lib`：

```
/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/examples/lammps/deepmd_kk_sog/
    in.deepmd_kk_sog.lammps    # 链 A，全文带注释
    run.sh                     # 环境变量与多卡绑定
```

核心 5 行：

```lammps
atom_style      charge/kk
pair_style      deepmd/kk ${MODEL} latent_charge_to_q yes full_lr no
pair_coeff      * * ${TYPES}
kspace_style    sog/kk ${KERNEL} use_gpu yes
fix             resp all sog/response model ${MODEL}
```

只需改 5 个变量（模型、数据、类型名、核参数、步数），其余通用。

链 B 的最小输入：

```lammps
units           metal
boundary        p p p
atom_style      charge/kk
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

**已实测**：标准输入在 192 原子水盒上 `run 0` 跑通，`pe = -29971.346 eV`，
与主机链（`pair deepmd` + `kspace sog`）在 np=1 与 np=2 上逐位相同。

### 4.2 核参数 `amp` / `bandwidth` 从哪来

**不要手推**，它们是训练后的核参数。产物出自哪棵树，取法不同：

- **产物出自第一部分这棵树**（`fitting=sog_energy`）：模型自带权威导出
  `fitting_net.lammps_kernel_params()`，或用
  `/data/home/public/jiangzhen/dp/dp_example/water/lmp/scripts/extract_sog_params.py
  <config.json> <ckpt.pt> out.json --include out.inc` 直接生成 `kspace_style` 那一行。
- **产物出自第二部分那棵树**（`latent_charge`）：核是 `long_range.kernel` 的
  `amplitude a_k` / `width σ_k`，按 §附录 A 换算成 C++ 口径。

### 4.3 运行

```bash
export LAMMPS_PLUGIN_PATH=/tmp/lmpkk_plugin        # 含 deepmd_lmp.so（或 *plugin.so 链接）
export DP_BACKEND_PLUGIN_PATH=<deepmd_install>/deepmd/lib
export LD_LIBRARY_PATH=<conda>/lib:<torch_nv>/cuda_runtime/lib:<torch_nv>/cuda_cupti/lib:<torch_nv>/nvjitlink/lib:<lmp_install>/lib:<deepmd_install>/deepmd/lib

lmp -k on g 1 -in in.deepmd_kk_sog.lammps                    # 单卡
mpirun -np 4 ./run.sh -np 4 in.deepmd_kk_sog.lammps          # 多卡（见 §6，目前不推荐）
```

`examples/lammps/deepmd_kk_sog/run.sh` 把上面这些包好了，改 4 个路径变量即可。

要点：

- **`-k on g 1`**：每个进程一张卡。多 rank 时 `CUDA_VISIBLE_DEVICES` 必须按
  `OMPI_COMM_WORLD_LOCAL_RANK` 设（`run.sh` 已处理）。
- **`LD_LIBRARY_PATH` 顺序有意义**：conda 的 `libstdc++` 要在系统之前（AOTI wrapper
  需要 GLIBCXX_3.4.30），torch 自带的 CUDA 运行时要在系统之前。
- 插件按 `LAMMPS_PLUGIN_PATH` 自动加载，也可 `plugin load <path>`。
- `use_gpu yes` 缺省是 `no`——结果对但会掉到主机求解器；`SOG_KK_DEBUG=1` 会打印
  `SOG/KK init: … device_ok=…`。

## 5. 约束与错误对照

| 约束 | 原因 / 现象 |
|---|---|
| **正交盒** | `sog` 拒绝带倾斜的 cell（`domain->triclinic` 且 tilt≠0）。三斜请用 `rbsog` 系列。 |
| **全周期** | `sog` 要求 `p p p`。 |
| **单电荷通道** | `sog/kk` 显式报错；多通道请用 `kspace_style sog`（主机）。 |
| **`latent_charge_to_q` 要 `atom_style charge`** | 否则 `atom->q` 不存在。 |
| **多卡不省显存** | `sog/kk` 多 rank 是**复制网格**：每 rank spread 自己的原子 → `MPI_Allreduce` → 每个 rank 各自跑完整 FFT。每卡显存 = 全局网格。 |
| **多卡目前只能用主机链** | device 响应在 np>1 段错误，且 replicated 路径的 virial 被乘 np——见 §6。 |

常见报错 → 原因：

| 报错 | 原因 |
|---|---|
| `full_lr cannot be combined with a SOG kspace style` | 产物带图内长程却没写 `full_lr no` |
| `latent_charge_to_q requires an atom style with charge` | 用了 `atom_style atomic` |
| `latent_charge_to_q requires a single model` | 多模型 ensemble |
| `kspace_style sog/kk supports a single latent-charge channel` | 多通道模型 |
| `fix sog/response requires kspace_style sog` | 没开 kspace（或写了 `full_lr yes`） |
| `this .pt2 ships no charge-response` | 产物缺 response extra，需重新冻结 |
| q 恒为 0、能量明显偏小 | `full_lr` 被自动打开，写 `full_lr no` |
| 每步慢 2× | `kspace_style` 没写 `use_gpu yes` |
| NPT 每步偏慢 | 绿函数每步重建；调 `box_tol` |

## 6. 已知问题（2026-11-10 实测，附一行修复）

三条都在 192 原子水盒（`sog_work/lmp/lmp150k/water_charge.lmp`，12.44³ Å）、
`dpa4_sog_150k_fused.pt2` 与同一套 `amp/bandwidth` 上复现。

### 6.1 多卡 virial 被乘了 np

`source/lmp/sog_kokkos.cpp::compute_device_replicated` 每个 rank 都拿到**全局**势场，
于是把全局 virial 写进 `KSpace::virial`：

```cpp
energy = e;
for (int j = 0; j < 6; ++j) virial[j] = vv[j];   // ← 每个 rank 都是全局值
```

但 LAMMPS 对这两者的约定不同：`energy` 不被归约（`thermo.cpp:2038` 直接读，所以每 rank
持全局值是对的），`virial` 却被 `ComputePressure` **MPI 求和**（`compute_pressure.cpp:342`）。
结果 k-space 压强 ×np：

| np | pe (eV) | press (bar) |
|---|---|---|
| 1 | -29971.346 | **+3662.48** |
| 2 | -29971.346 | **-7558.33** |

两者差 −11220.8 bar，正好等于多算一份 k-space virial。**修复一行**：

```cpp
const double vshare = 1.0 / static_cast<double>(comm->nprocs);
for (int j = 0; j < 6; ++j) virial[j] = vv[j] * vshare;
```

主机链（`pair deepmd` + `kspace sog`，PPPM 式分布式网格）没有这个问题：np=1 / np=2
实测 `pe = -29971.346`、`press = -9100.665 / -9100.667`，一致到 1e-3 bar。

### 6.2 `fix sog/response` 的 device 响应在 np>1 段错误

`deepmd/kk` + `sog/kk` + `fix sog/response` 在 np=1 正常；np=2 直接 SIGABRT，
栈在 `PairDeepMDKokkos<>::compute_charge_response_device` →
`DeepPotPTExpt::compute_charge_response_edges_gpu`（`pair_deepmd_kokkos.cpp:893`）。
去掉 `fix` 只留 pair+kspace，np=2/4 正常。

根因上看，device 响应分支执行后**直接 return，不做 `comm->reverse_comm(this)`**，而主机
分支会做——这条路径本就是按 np=1 写的，只是缺 np>1 的护栏。

**现状建议**：np>1 用主机链

```lammps
pair_style   deepmd ${MODEL} latent_charge_to_q yes
kspace_style sog  ${KERNEL}
fix          resp all sog/response model ${MODEL}
```

### 6.3 device 响应与主机响应的压强不一致（np=1）

| 配置 | pe (eV) | press (bar) |
|---|---|---|
| device pair + `sog/kk`，**无** fix | -29971.346 | +3662.48 |
| device pair + `sog/kk` + fix（device 响应） | -29971.346 | **-4995.93** |
| 主机 pair + `sog` + fix（主机响应） | -29971.346 | **-9100.67** |

能量完全相同，`fix` 贡献的 virial 却差 ~4100 bar（device 补 −8658 bar、主机补 −12763 bar）。
`virial_global_flag` / `thermo_virial` 都已置 1（`fix_sog_response.cpp:84-89`），所以不是
"virial 没进压强"，而是**两条响应路径的 virial 口径不同**，尚未定位。

**在查清之前**：多卡用主机链；单卡两条都行，但不要跨链比压强。

---

# 第二部分　`deepmd-kit-LR-lr-pt`：模型与训练侧

## 7. 这棵树有什么、没有什么

**有**：

| 路径 | 作用 |
|---|---|
| `deepmd/dpmodel/long_range/kernel.py` | `SOGKernel`：`amplitude a_k` 与 `width σ_k`（`K(r)=Σ a_k e^{-r²/2σ_k²}`） |
| `deepmd/dpmodel/long_range/latent_charge.py` | `LatentChargeLongRange`：`mesh_backend`、`force_mode`、`pme_accuracy`、`mesh_spacing()` |
| `deepmd/dpmodel/long_range/grid_rule.py` | 网格误差律（按 backend/阶数/力模式给 `C, p`） |
| `deepmd/pt_expt/kernels/long_range.py` | 算子包装 `latent_charge_sum`（调 CUDA op） |
| `deepmd/pt_expt/long_range/latent_charge.py` | `pt_expt` 模块：把长程接到描述符图上 |
| `source/op/pt/long_range/` | 算子的 CUDA 实现（bspline / midtown / quads、`ad` / `ik`） |
| `docs/FREEZE_SOP.md`、`docs/GRID_FROM_ACCURACY.md` 等 | 冻结与精度规则 |

**没有**（这一点决定了两棵树不能混用）：

- `source/lmp/` 里**没有** `sog.cpp` / `sog_kokkos.cpp` / `fix_sog_response.cpp`；
- `pair_deepmd.cpp` / `pair_deepmd_kokkos.cpp` 里**没有** `full_lr`、`latent_charge_to_q`
  （`grep` 计数为 0）；
- 没有任何 `kspace_style`。

所以这棵树**不能**提供第一部分 §2 的链 A——它的长程模型必须**整段在图里**，由普通的
`pair deepmd` 调用（`kspace_style none`）。

## 8. 长程在这棵树里怎么表示

长程是一个**算子**，不是 k-space style：

```
deepmd::latent_charge_sum(coord, charge, mask, n_node, total_charge,
                          kernel, channel, prefactor, box,
                          order, spacing, mesh_backend, force_mode)
    → (atom_energy, charge_gradient, force, atom_virial, virial)
```

- **`mesh_backend`**：`"bspline"`（平滑 PME）、`"midtown"`（CubeS₂）、`"quads"`（QuadS）。
  midtown/Quads 只有 4 或 6 阶。
- **`force_mode`**：`"ad"`（对插值权重求导，是网格能量的精确梯度）、`"ik"`（在 Fourier
  空间求导，高一阶但需要 midtown/Quads 网格）。
- **`spacing`**：由 `grid_rule.spacing()` 从实测误差律反解（`pme_accuracy`，
  默认 1e-4），**没有 Nyquist 上限**——高阶样条提高的是同网格下的精度。

参考核（`σ_min=2.0 Å`、`k_max=3.0349 Å⁻¹`、盒 122.31 Å、1e-4）上的网格：
bspline-8 115³、bspline-6 173³、CubeS₂-6 265³、CubeS₂-6·ik 177³、CubeS₂-4·ik 291³。
详见 `docs/GRID_FROM_ACCURACY.md`、`docs/MIDTOWN_VS_BSPLINE_BENCH.md`。

## 9. 冻结成 `.pt2`

```bash
dp --pt-expt freeze -c model.ckpt.pt -o model --compress
```

- `--compress` 出 **compact canonical 下界**（`lower_input_kind=dpa4c_canonical`）：
  更快、显存省一两个量级，且**多卡必需**；默认 graph 下界在多卡下会拒绝。
- 长程模型在 `model/extra/metadata.json` 里声明
  `topologies: ["all_pairs", "pme_mesh"]` 与 `topology_params.pme_mesh.spacing`
  （= 冻结时的 `mesh_spacing()`）；`source/api_cc/src/DeepPotPTExpt.cc:313-330`
  读这两项，C++ 侧据此构造网格拓扑。
- 长程模型要求 `graph` 或 `compact canonical` 下界，其他下界直接抛异常。

完整流程见 `docs/FREEZE_SOP.md`。

## 10. 与第一部分的接口

两棵树唯一的接触点是 **模型加载库**。LAMMPS 的 pair（无论来自哪棵树）通过
`DP_BACKEND_PLUGIN_PATH` 选一个 `deepmd/lib`，那里有 `libdeepmd_cc.so` 与
`libdeepmd_backend_ptexpt.so`（两棵树都会编译这两个）。

本树的产物有两种用法，**都不需要把模型源码搬来搬去**：

| 用法 | LAMMPS 输入 | 说明 |
|---|---|---|
| **图内长程**（本树产物的默认） | `pair_style deepmd/kk ${PT2}` + `kspace_style none` | `.pt2` 自带 `pme_mesh` 拓扑（`topology_params.pme_mesh.spacing`），由 backend 按盒子建网格；**不需要** `kspace_style`，也不需要 `fix sog/response`。 |
| **交给第一部分的链 A** | `pair_style deepmd/kk ${PT2} latent_charge_to_q yes full_lr no` + `kspace_style sog/kk …` + `fix sog/response` | 仅当产物带 `latent_charge` 输出与 `latent_charge_response.pt2` extra 时可用；`amp/bandwidth` 按附录 A 从 `amplitude/width` 换算。 |

注意 `full_lr`（第一部分 §2 链 B）是**第一部分自己的 `sog_full_lr.pt2` 产物专用**的关键字；
本树的 `pme_mesh` 产物不需要它，也未必带它（查 `model/extra/sog_full_lr.pt2` 是否存在）。

## 11. 本树自己的文档

- `docs/FREEZE_SOP.md` —— 冻结 SOP（`--compress`、谱系锁、多卡）
- `docs/BUILD_AND_TEST.md` —— 两棵树的编译与测试全流程
- `docs/GRID_FROM_ACCURACY.md` —— 网格误差律与参考核网格表
- `docs/IK_MODE_ACCURACY.md`、`docs/AD_MODE_ACCURACY.md` —— 两种力微分的实测
- `docs/MIDTOWN_VS_BSPLINE_BENCH.md` —— 等精度下的后端选择
- `docs/SLAB_DEPLOYMENT.md` —— z-slab 分布式网格（算子侧，与 LAMMPS 的 `sog/kk` 无关）

本树自己的验收：`pytest source/tests/pt_expt/kernels/test_mesh_backend.py`、
`pytest source/tests/common/dpmodel/test_grid_rule.py`、`test_long_range.py`。

---

# 附录 A　核参数口径换算（第二部分的模型 → 第一部分的 `sog/kk`）

模型侧：`K(r) = Σ_k a_k exp(-r²/(2σ_k²))`（`amplitude a_k`、`width σ_k`）。

第一部分 `sog.cpp` 的口径：`K(k²) = Σ_m amp_m exp(-½·bandwidth_m·k²)`。

对 `K(r)` 做傅里叶变换 `∫K(r)e^{-ik·r}d³r = Σ_k a_k (2πσ_k²)^{3/2} exp(-σ_k²k²/2)`，
逐项对应即得：

```python
amp       = [a * (2*np.pi)**1.5 * s**3 for a, s in zip(amplitude, width)]
bandwidth = [s**2 for s in width]
```

核对：`sog.cpp` 的自能项 `self_coeff = ½·Σ amp_m (2π·bandwidth_m)^{-1.5}`
与实空间 `½·Σq²·g(0)`、`g(0)=Σ a_k (2πσ_k²)^{-3/2}` 完全一致。

# 附录 B　上线校验清单

**第一部分（LAMMPS 链）**：

1. `run 0` 对拍：`pe` / 力 / 压强 vs `deepmd.infer.DeepPot(ckpt)` 的前向。链 A 在训练盒
   上应到 ~0.1 meV/atom、~1e-3 eV/Å（`sog_work/docs/MD_DEPLOYMENT_PARITY.md`）。
2. `q` 非零：`dump … custom 1 q.dump id type q`，与模型 `latent_charge` 逐原子比。
3. NVE 能量守恒：不加恒温器 10 ps，`etotal` 漂移应 ≲ 1e-2 eV。
4. 压强单位：metal 下 `press` 是 **bar**。
5. `SOG_KK_DEBUG=1` 确认 `device_ok=1`。
6. 多卡：`pe` 应逐位相同；**压强目前不会**（§6.1）。

**第二部分（模型侧）**：

1. `pytest source/tests/pt_expt/kernels/test_mesh_backend.py`（各样条对 replicated 的偏差）。
2. `pytest source/tests/common/dpmodel/test_grid_rule.py test_long_range.py`。
3. 冻结后 `dp test -m model.pt2` 能加载、`topologies` 里含 `pme_mesh`。
