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
> `latent_charge` 产物怎么进来，见 §11。

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

## 9. 冻结成 `.pt2`：dpa4c-sog 与 dpa4-sog

```bash
dp --pt-expt freeze -c model.ckpt.pt -o model --compress
```

两条谱系**用同一条命令、出同一种工件**（`lower_input_kind=dpa4c_canonical`）。
"dpa4c" 在这里是**压缩构造的名字**，不是描述符的名字：DPA4C（全局部）和 DPA4（消息传递）
都被压到这一个 compact canonical 下界上，差别只在压缩后的产物里有没有交换工件。

### 9.1 压缩后的两条产物差在哪

| | **dpa4c-sog**（DPA4C + latent_charge） | **dpa4-sog**（DPA4/SeZM + latent_charge） |
|---|---|---|
| `lower_input_kind` | `dpa4c_canonical` | `dpa4c_canonical`（同） |
| `has_message_passing` | `false` | **`true`** |
| `has_comm_artifact` | `false` | **`true`**（多出 `model/extra/forward_lower_with_comm.pt2`） |
| 多卡时 halo 特征 | 就地算，不需要交换 | **每层之后跨 rank 交换** |
| 多卡特有约束 | 无 | 每个 rank 至少一个节点（见 §10.4） |
| CUDA graph 捕获 | 多卡不重放（halo 让 extent 每步变）；单卡可重放 | 多卡不重放（且交换是 host 集合通信，即使 extent 不变也不能捕获）；单卡折叠后仍可重放 |

压缩本身对两者是一样的：节点轴收缩到 model node（null-type 原子剔除，`loc2model` /
`model2loc` 每张邻居表重建一次）、键矢量 f32（模型长度单位）、边用 32-bit 节点号排两次
（destination-major CSR + 需要时 source-major CSR 及置换）。同权重下能量不变，约 3× 快、
约 1/40 显存。

### 9.2 产物自检

```bash
python - <<'PY'
import zipfile, json
z = zipfile.ZipFile("model.pt2")
m = json.loads(z.read("model/extra/metadata.json"))
print("lower:", m["lower_input_kind"])          # dpa4c_canonical
print("MP   :", m["has_message_passing"])       # dpa4-sog 为 True
print("comm :", m["has_comm_artifact"])         # dpa4-sog 为 True
print("1rank:", m["canonical_single_rank_only"])# 必须为 False 才能多卡
print("top  :", m["topologies"])               # 含 all_pairs + pme_mesh
print("spac :", m["topology_params"]["pme_mesh"]["spacing"])
print([n for n in z.namelist() if "with_comm" in n])
PY
```

四个字段要一起看：

- `has_comm_artifact=true` 但归档里**没有** `model/extra/forward_lower_with_comm.pt2`
  ——冻结树的版本比这条链旧，重新冻结即可。
- `canonical_single_rank_only` 由冻结时算出（`has_message_passing and not
  有交换工件`），为 `true` 时冻结日志会给警告。它是**记录与警告**，运行期不读；真正的运行期
  门是"有交换工件 + 拿到 comm 载荷"，缺一即报错（§10.4）。
- `canonical_index_dtype` / `canonical_index_storage` / `graph_edge_dtype` 是 compact ABI
  的编码：`auto`/`byte_words`/`float32`。驱动侧按它们决定 32 还是 64 位索引、键矢量精度。
- `canonical_requires_source_csr=true` 表示力按 source 分组装，算子每步都会读 source CSR。

### 9.3 交换工件是什么

`dpa4-sog` 的 `forward_lower_with_comm.pt2` 是同一条 compact ABI 加上：

1. 长程操作数——节点位置（f64, `3N`）、模块自己的 communicator、晶胞（f64, 9）；
2. 八个交换张量——send/recv 列表与计数、communicator、`nlocal`、`nghost`。它们**留在
   host**，因为消费它们的是 `deepmd_export::border_op` 这个不透明算子（host 侧解引用
   send-list 指针）。

**一个 communicator 兼两用**：它既是长程算子的集合通信句柄，也是 border 交换的句柄，所以
compact ABI 只带一个 MPI handle 而不是两个。

**为什么必须有它**：没有交换工件时，消息传递描述子会把每个 halo 节点**沿用第一层给出的
特征**——能量看着正常，但跨 rank 的消息从未发生。这条路径现在会**明确报错**而不是给出
错答案（见 §10.4）。

完整冻结流程见 `docs/FREEZE_SOP.md`。

## 10. 调用：`pair_style deepmd/kk <model>.pt2`

本树的产物长程**在图内**（`pme_mesh` 拓扑），所以**不需要 `kspace_style`，也不需要
`fix sog/response`**，而且 `full_lr`、`latent_charge_to_q` 这两个关键字在本树里不存在。

### 10.1 单卡

```lammps
units           metal
boundary        p p p
atom_style      charge/kk          # 主机插件用 charge
atom_modify     map yes            # 消息传递模型必需；全局部模型也建议加
neighbor        2.0 bin
neigh_modify    every 20 delay 0 check no
read_data       <system>.lmp
mass            1 16
mass            2 2
pair_style      deepmd/kk <model>.pt2      # 主机插件写 deepmd <model>.pt2
pair_coeff      * * O H
kspace_style    none
fix             1 all nve/kk                # 主机插件用 nve
run_style       verlet/kk                  # 主机插件可不写
velocity        all create 300.0 12345 mom yes rot no dist gaussian
timestep        0.0005
thermo_style    custom step temp pe press vol
run             100000
```

```bash
lmp -k on g 1 -in in.lammps
```

单卡时 `nloc == nall_nodes`，`dpa4-sog` 也**不**交换、也**会**被捕获重放，和 `dpa4c-sog`
走同一条路径。

### 10.2 多卡

```lammps
processors      1 1 2              # 或 1 1 4；z 向切分
atom_modify     map yes            # pair_style deepmd/kk 硬性要求（init_style 里检查）
pair_style      deepmd/kk <model>.pt2
kspace_style    none
```

```bash
mpirun --oversubscribe -np 2 ./lmp_rank_dev.sh lmp -k on g 1 -in in.lammps
```

`/tmp/lmp_rank_dev.sh` 见 §14.3——把 `CUDA_VISIBLE_DEVICES` 按 `LOCAL_RANK` 逐 rank 缩到一个
GPU。

`dpa4-sog` 多卡时 `pair deepmd/kk` 会自动把邻居表的 send/recv 元数据传给模型，模型在
forward 内交换 halo 特征；`dpa4c-sog` 不传，因为没有东西要换。**两者对输入的要求相同**，
区别只在性能（DPA4 每步多一次 host 集合通信，且因此放弃 CUDA graph 重放）。

### 10.3 实测（2026-11-11）

`master-merge-test/water_dpa4_sog` 的 `model.ckpt-30000.pt` 重新 `--compress`，
192 原子水盒子，`run 0`，`|F|rms = 0.81 eV/Å`：

| 运行 | PE (eV) | press (atm) | max\|ΔF\| vs host np1 |
|---|---|---|---|
| host np=1 | −29946.758 | −3151.59 | — |
| host np=2 / np=4 | −29946.758 | −3151.59 | 2.7e-6 / 3.0e-6 |
| **`deepmd/kk` np=1 / np=2 / np=4** | −29946.758 | −3151.59 | **7.0e-6 / 6.6e-6 / 6.7e-6** |

净力各次都在 ~1e-5。用同样的盒子加两个 NULL 型（虚拟）原子、`pair_coeff * * O H NULL`，
`host`/`kk` 的 np=2/4 同样到 2.3e-6 / 5.8e-6，虚拟原子受力恒为零。

对照：非消息传递的 `dpa4c-sog` 压缩产物（`lr_bs6.pt2`）np=1/2/4 无回归，`kk` 到 7e-6。

### 10.4 多卡的四种拒绝（都是刻意的）

| 现象 | 原因 |
|---|---|
| `... needs the swap metadata of the decomposition (comm_nlist) ...` | 有交换工件却没拿到 comm 载荷；只有宿主引擎不打这个参数时会触发 |
| `... cannot serve a rank without nodes.` | 消息传递 + 压缩 ABI 没有 phantom 原子的位置；某个 rank 被切空即拒绝，不会挂在别人已进入的集合通信里 |
| `... evaluates a periodic system with a long-range module only through a compressed DPA4C model ...` | `deepmd/kk` 的 device edge 路径不传晶胞；周期体系的长程只能走压缩 canonical |
| 未压缩的 graph 产物在多卡下报缺 with-comm 工件 | graph 的 with-comm 下界**没有**长程槽位（无位置/晶胞/communicator），所以**消息传递 + 长程的模型必须在冻结时压缩**才能多卡 |

## 11. 与第一部分的接口

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

## 12. 本树自己的文档

- `docs/FREEZE_SOP.md` —— 冻结 SOP（`--compress`、谱系锁、多卡）
- `docs/BUILD_AND_TEST.md` —— 两棵树的编译与测试全流程
- `docs/GRID_FROM_ACCURACY.md` —— 网格误差律与参考核网格表
- `docs/IK_MODE_ACCURACY.md`、`docs/AD_MODE_ACCURACY.md` —— 两种力微分的实测
- `docs/MIDTOWN_VS_BSPLINE_BENCH.md` —— 等精度下的后端选择
- `docs/SPLINE_MATRIX_BENCH.md` —— 极端大体系（526 k / 1.12 M 原子）下十种样条配置 ×
  复制/分布式网格的时间与显存全表
- `docs/SLAB_IK_FAILURE_DIAGNOSIS.md` —— 分布式网格 × ik 的崩溃诊断与修复记录
- `docs/PR_SUBMISSION.md` —— 提 PR 用的范围划分、环境、编译与验证步骤
- `docs/SLAB_DEPLOYMENT.md` —— z-slab 分布式网格（算子侧，与 LAMMPS 的 `sog/kk` 无关）

本树自己的验收：`pytest source/tests/pt_expt/kernels/test_mesh_backend.py`、
`pytest source/tests/common/dpmodel/test_grid_rule.py`、`test_long_range.py`。

## 13. 故障与修复（本树的代码）

这一节是本树对两个"看起来像 LAMMPS 崩溃"的问题的处理。完整诊断过程见
`docs/SLAB_IK_FAILURE_DIAGNOSIS.md`；这里给的是**结论与用法**。

### 13.1 ik × 分布式网格：已支持，但默认仍走 replicated

**历史**：早先 ik 模型（`force_mode: ik`）遇到 `SOG_MIDTOWN_SLAB=1` 会直接中止，
报 `Expected a proper Tensor but got None … 'self'`——分布式求解器的力只来自 AD gather，
没有 `i·k` 微分，算子用 `TORCH_CHECK` 硬拒，而这个 env 通常是进程级设的（bench 脚本、
容器 profile），于是"一个环境变量打死整条 MD"。中间版本改成"警告 + 回落到 replicated"。

**现在**：z-slab 求解器**实现了 ik 通路**，但**默认不用**——`SOG_MIDTOWN_SLAB=1` 单独出现
时仍然回落到 replicated（力定律相同，每 rank 持全网格），并打一次警告；要真的跑分布式 ik
必须再显式加一个开关：

```bash
export SOG_MIDTOWN_SLAB=1 SOG_SLAB_IK=1     # ik 走 z-slab 分布式网格
export SOG_SLAB_STRICT=1                    # 上两者缺任一个时硬失败，而不是回退
```

为什么默认不跟：ik 的每个笛卡尔分量都要走一遍"逆 z + 转置 + 逆 (x,y)"，即**转置次数 ×4**，
外加一个 pencil 和一个 brick 的工作集；这个代价不该由一个上游统一设的 `SOG_MIDTOWN_SLAB`
替你决定。

**实现**（`midtown_sum_slab`，[`midtown_sum.cu:1330`](../source/op/pt/long_range/midtown_sum.cu#L1330)
起）：green 乘完之后把势谱 φ̂ 留在 `s_pot2`，然后对 x/y/z 各做一次 `i·k_a·φ̂`
（**在 z-pencil 域**，那里 kx/ky/kz 齐全，自共轭判据与 replicated 的 `k_mul_ik` 完全一致），
各自逆 z、转置、逆 (x,y) 到同一个梯度 brick，再逐分量 gather。三条分量依次复用同一对
block/pencil 缓冲和同一个梯度 brick，所以内存只多一个 pencil + 一个 brick，不是四套。

**数值**（`lr_mid4ik.pt2` / `lr_mid6ik.pt2` / `lr_quads6ik.pt2`，192 原子水盒，`run 0`，
`|F|rms = 0.81 eV/Å`）：

| 扫描 | np=1 | np=2 | np=4 |
|---|---|---|---|
| CubeS₂-4 ik：分布式 vs 复制 | 6.7e-7 | 2.4e-6 | 1.8e-6 |
| CubeS₂-6 ik：分布式 vs 复制 | 6.8e-7 | 1.9e-6 | —（网格 18³，`1 1 4` 下 4 < 6 阶模板，沿用原有网格守卫） |
| QuadS-6 ik：分布式 vs 复制 | 5.8e-7 | 1.8e-6 | — |

PE 各次都是 −29946.558 eV，净力 ~1e-6。AD 分布式路径无回归（5.8e-7 / 1.6e-6 / 2.1e-6）。

单步代价（192 原子，40 步，小网格下开销主导）：复制 ik 0.102 s、分布式 ik np=1 0.116 s
（+14%）、np=2 0.097 s；作为对照分布式 AD np=1 比复制 AD 慢 +11%。也就是说三个额外转置
在整步里与分布式本身的固定开销同量级。

`SOG_MIDTOWN_DIST=1`（local-spread）仍然**没有** ik 通路：它的 stencil 是 CubeS₂ 6 阶硬编码
的，没有 Fourier 微分版 gather；设了它 + ik 仍回落到 replicated 并警告。

**网格守卫**：`ZL = nz/np` 必须 ≥ 模板宽度（CubeS₂/QuadS 4 → 4，6 → 6）。小盒子配高阶
ik 时 `1 1 4` 会报 `nz/np = … is thinner than the stencil` —— 这是网格划分的限制，不是
ik 的问题，换更细网格或更少 rank。

### 13.2 算子异常不再被 AOTI 吞成 "None / 'self'"

`.pt2` 里的自定义算子通过 AOTI proxy executor 调用；它**捕获**算子抛出的 C++ 异常，只往
stderr 打一行 `[E... shim_common.cpp]`，并把该算子的输出张量留成 undefined。图照常"返回"，
上层随后在第一次 `reshape` 上崩，报的却是 `Expected a proper Tensor … 'self'`——**显存不足、
力模式被拒、参数非法三种事故同一个字符串**。Python 路径更糟：不崩，返回 `nan`。

**修复**：`require_defined_outputs()`（`source/api_cc/include/commonPT.h`）在每次
`extract_outputs` 之后点名列出 undefined 的槽位，并指向 stderr 的 `[E...]` 行；
`flatten_canonical_atom_virial` 也不再对 None 调 `reshape`。

```
ERROR: … DeepPotPTExpt::compute: the exported graph returned undefined output(s)
'atom_energy', 'atom_virial', 'energy', 'force', 'mask', 'virial'. A custom operator
inside the graph threw and the AOTI proxy executor caught it instead of propagating
it, so the model still 'returned' its outputs with those slots unset. The operator's
own message is the '[E...] … shim_common.cpp' line on stderr -- look there for the
real cause (mesh too large for the device, an unsupported force mode, invalid
arguments, ...).
```

### 13.3 网格过大：建表之前就报，而不是裸 OOM

replicated 网格按 **20 B/格点**占用设备内存（`f_gp` 8 + `f_rho_r` 4 + `f_rho_fx` 8）。
旧路径先花约一分钟在 host 上造几十 GB 的 fp64 绿函数，再在 `cudaMalloc` 上失败。
现在先与 `cudaMemGetInfo` 比对：

```
midtown: the replicated mesh needs 26.52 GiB of device memory for a 1125x1125x1125 grid,
but only 18.14 GiB is free. Use a coarser mesh (a larger `pme_spacing` or a less
demanding `pme_accuracy`), a smaller box, or a distributed backend where the force mode
allows one.
```

同一个 526 848 原子 / 1125³ 的算例从"1 分钟后 OOM"变成 **10 秒**内报错。

### 13.4 相关开关一览

| 环境变量 | 作用 |
|---|---|
| `SOG_MIDTOWN_SLAB=1` | 走 z-slab 分布式网格。AD 力直接生效；ik 力**需同时设** `SOG_SLAB_IK=1`，否则回落到 replicated |
| `SOG_SLAB_IK=1` | 打开 z-slab 的 ik（Fourier 微分）通路：转置次数 ×4，多一个 pencil + 一个 brick |
| `SOG_MIDTOWN_DIST=1` | 走 local-spread 分布式网格（只有 AD 力；ik 一律回落） |
| `SOG_SLAB_STRICT=1` | 需要分布式却没开成（如 ik 未设 `SOG_SLAB_IK`）时**硬报错**而不是回落（CI / 批处理用） |
| `SOG_MIDTOWN_FORM_A=0/1` | CubeS₂ 的 Form A / Form B 去卷积 |
| `SOG_MIDTOWN_DEBUG=1` | 打印网格重建与 kernel 参数 |
| `SOG_SLAB_TIME=1` / `SOG_SLAB_TRACE=1` | 分阶段计时 / 阶段跟踪（调试分布式管线） |

## 14. 本树的标准编译与调用流程

前面 §11 说的是"产物怎么被第一部分消费"。这一节是**在这一棵树里从源码到跑起来**的完整流程。

### 14.1 环境

```bash
source /etc/profile.d/modules.sh
module load gcc/13 cmake/3.26.3 openmpi/4.1.4
conda activate lr212            # Python 3.11，torch 2.12.1+cu126
export PATH=/usr/local/cuda-12.4/bin:$PATH     # 系统默认 nvcc 是 11.8，不认 -std=c++20
```

| 组件 | 必须是 | 不对会怎样 |
|---|---|---|
| gcc / mpicc | 13.x | AOTI 冻结报 `-std=c++20: unrecognized`；插件 MPI 符号错 |
| cmake | 3.26.3 | — |
| CUDA toolkit | **12.4** | 11.8 编不过 `-std=c++20` |
| torch | **2.12.1+cu126** | 算子 `.so` 与运行环境 ABI 不符，`import deepmd` 直接报版本错 |
| LAMMPS 源码 | `$REPO/lammps_src/lammps-stable_22Jul2025_update2` | 编 `deepmd_lmp` 目标需要 |

> 仓库根 `AGENTS.md` 写的是**上游** `uv pip install -e .` 流程，不适用：这里要同时编 CUDA
> 算子、MPI 与 LAMMPS 插件。

### 14.2 编译

**① 核心五目标**（算子 / C++ API / pt_expt 后端 / C API / **主机版 LAMMPS 插件**）：

```bash
export REPO=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
sog_work/mdsuite/build_lrpt212_full.sh --build
# 等价于：cmake -S $REPO/source -B $REPO/build-lmp-mpi-212 <一堆开关>；
#         make deepmd_op_pt deepmd_cc deepmd_backend_ptexpt deepmd_c deepmd_lmp
#         装进 /data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib
#         主机插件软链 -> /tmp/lmpplugin_lrpt/dpplugin.so
#         并校验 canonical op 的 22 参数 schema 含 force_mode
```

**② Kokkos（device）插件** —— 同一份源码，`DEEPMD_LAMMPS_KOKKOS=ON`，但**不能**用普通
`nvcc_wrapper` 直接编（三个坑：CMake 发 `-std=c++2a`；CMake 先发 `CXX_INCLUDES` 再发
`CXX_FLAGS`，LAMMPS 的 `src/STUBS/mpi.h` 会先命中导致 MPI 符号 C++ 修饰；Kokkos 要
`-fopenmp`）。用一个 shim 抹平：

```bash
cat > /tmp/nvcc_wrapper_shim <<'EOF'
#!/bin/bash
OMPI=/data/app/openmpi-4.1.4/install/openmpi/4.1.4
args=("-I$OMPI/include")          # 必须排在所有 -I 之前
for a in "$@"; do
  case "$a" in
    -std=c++2a|-std=gnu++2a) args+=("-std=c++20") ;;
    *) args+=("$a") ;;
  esac
done
exec /data/home/public/jiangzhen/lammps/lammps_stable/lib/kokkos/bin/nvcc_wrapper "${args[@]}"
EOF
chmod +x /tmp/nvcc_wrapper_shim

cmake -S $REPO/source/lmp/plugin -B /tmp/kk_lrpt_build \
  -DCMAKE_CXX_COMPILER=/tmp/nvcc_wrapper_shim \
  -DDEEPMD_LAMMPS_KOKKOS=ON \
  -DLAMMPS_SOURCE_ROOT=/data/home/public/jiangzhen/lammps/lammps_stable \
  -DKokkos_DIR=/data/home/public/jiangzhen/lammps/lammps_stable/build_kokkos/lib/kokkos \
  -DCMAKE_CXX_FLAGS="-fopenmp -pthread -I$REPO/source/api_cc/include -I$REPO/source/lib/include"
cmake --build /tmp/kk_lrpt_build -j 16
mkdir -p /tmp/kk_lrpt_plugin
ln -sfn /tmp/kk_lrpt_build/libdeepmd_lmp.so /tmp/kk_lrpt_plugin/dpplugin.so
```

### 14.3 运行时环境

```bash
CONDA=/data/home/jiangzhen/.conda/envs/lr212
NVD=$CONDA/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$CONDA/lib:$CONDA/lib/python3.11/site-packages/torch/lib:\
$NVD/cuda_runtime/lib:$NVD/cuda_cupti/lib:$NVD/nvjitlink/lib:$NVD/cudnn/lib:\
$NVD/cublas/lib:$NVD/cusparse/lib:$NVD/cusparselt/lib:$NVD/curand/lib:\
$NVD/cufft/lib:$NVD/cufile/lib:$NVD/nccl/lib:$NVD/nvshmem/lib:\
/data/home/public/jiangzhen/dp/lmp_kokkos_install/lib:\
/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib
export DP_BACKEND_PLUGIN_PATH=/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib
export LAMMPS_PLUGIN_PATH=/tmp/kk_lrpt_plugin      # 或 /tmp/lmpplugin_lrpt（主机插件）
export OMP_NUM_THREADS=1
```

顺序有意义：conda 的 `libstdc++` 要在系统之前，torch 自带的 CUDA 运行时要在系统之前。

多卡时一个 rank 一张卡，用这个 shim 把 `CUDA_VISIBLE_DEVICES` 按 `LOCAL_RANK` 缩到一张：

```bash
#!/bin/bash
# /tmp/lmp_rank_dev.sh
IFS=',' read -r -a _devs <<< "${CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES="${_devs[${OMPI_COMM_WORLD_LOCAL_RANK:-0}]}"
exec "$@"
```

### 14.4 标准调用

输入文件、单卡/多卡命令、逐 rank 绑卡脚本都在 **§10**。这里只补环境侧的两条：

- 多卡启动时 `LAMMPS_PLUGIN_PATH` 指向 Kokkos 插件目录（§14.2 的 `/tmp/kk_lrpt_plugin`），
  主机插件是 `/tmp/lmpplugin_lrpt`；两者**不要同时**挂着。
- **产物 → 输入**：`.pt2` 里 `model/extra/metadata.json` 的
  `topology_params.pme_mesh.spacing` 决定网格；`pme_order` / `mesh_backend` / `force_mode`
  是**冻结时的计算设置**，改它们要重新冻结（`sog_work/tools/freeze_lr_variants.py
  --backend {bspline|midtown|quads} --order {4|6} --force-mode {ad|ik}`，冻结前
  `module load gcc/13`）。运行期没有能改样条类型的开关，见 §15。

### 14.5 自检

```bash
# 纯 Python（不需 GPU）
python -m pytest source/tests/common/dpmodel/test_grid_rule.py \
                 source/tests/common/dpmodel/test_long_range.py -q        # 82 passed
# 算子（需要一张 GPU，PYTHONPATH=/ LD_LIBRARY_PATH 指向 deepmd/lib）
python -m pytest source/tests/pt_expt/kernels/test_mesh_backend.py -q     # 20 passed

# 端到端：run 0 对拍 + 多卡一致
lmp -k on g 1 -in in.lammps -log log.np1
mpirun -np 2 ... -in in.lammps -log log.np2
# pe 应逐位相同；不同 mesh_backend 之间按等精度设计应到 ~1e-8 eV/atom
```

> **量单步时间时不要用 `run 6`**：LAMMPS 的 `Loop time` 含一次性 setup（邻居表、首次求值、
> AOTI 图捕获），在几十万原子上是好几秒。用 `run 5` + `run 20` 两段，取第二段
> （`docs/bench_spline_matrix.py` 就是这么做的）。

## 15. MD 时能不能改 SOG 用的样条类型？

**分两棵树的链回答。第一部分的链可以，第二部分的链不可以。**

### 15.1 第一部分：`kspace_style sog|sog/kk` 有 `spline` 关键字

`source/lmp/sog.cpp::settings()` 解析，**运行期生效，不需要重新编译或重新冻结**；
`SOGKSpaceKokkos` 继承同一个 `settings()`，所以 `sog` 和 `sog/kk` 用法一致。默认
**CubeS₂ 4 阶**（关键字总表见 §3.2）：

```lammps
# 追加在核参数之后、其它关键字之前或之后均可
kspace_style    sog/kk ${KERNEL} spline cubes2_6 use_gpu yes
```

`${KERNEL}` 就是 §4.1 里那组 `amp … bandwidth …`（或旧的 `accuracy b sigma M [n_dl]`）。

| `spline` 取值 | 含义 |
|---|---|
| `bspline` / `bspline_5` | 可分离 B-spline，5 阶（`bspline` 是 `bspline_5` 的别名） |
| `bspline_4` / `bspline_6` | 可分离 B-spline，4 / 6 阶 |
| `cubes2_4` | CubeS₂（midtown 非可分离窗），4 阶 —— **默认** |
| `cubes2_6` | CubeS₂，6 阶 |
| `quads_4` / `quads_6` | QuadS³，4 / 6 阶 |

同一处 `settings()` 还有 `grid_method`（`sog_bandwidth`／`pppm_legacy`）、
`mesh_oversample`、`mesh_alias_extent`、`remove_self_interaction`、`phi_max`、`n_dl`。
CUDA 侧另有只在运行期读的环境变量（`SOG_MIDTOWN_SLAB`、`SOG_MIDTOWN_DIST`、
`SOG_MIDTOWN_FORM_A`、`SOG_SLAB_STRICT`）——它们只改**同一个样条**的网格分布/Green 形式，
**不改样条族或阶数**。

### 15.2 第二部分：样条是**冻结进图**的常量，运行期无接口

`mesh_backend`、`pme_order`、`force_mode` 是 `latent_charge_sum` 算子的**参数**，在冻结时
由模型配置决定，随后作为常量写进 AOTInductor 程序。`.pt2` 的操作数表（`pair_style
deepmd/kk` 的输入）里**没有**这几个字段，`pair_deepmd*` 也不解析任何相关关键字。

要换样条，**必须重新冻结**（权重不动，只换求值设置）：

```bash
module load gcc/13
python /data/home/public/jiangzhen/dp/sog_work/tools/freeze_lr_variants.py \
    --ckpt model.ckpt-30000.pt --out model_mid4.pt2 \
    --backend midtown --order 4 --force-mode ad
```

| 参数 | 取值 | 说明 |
|---|---|---|
| `--backend` | `bspline` / `midtown` / `quads` | 对应 `mesh_backend`；默认 `bspline` |
| `--order` | `4` / `6` | midtown/Quads 只有这两个 |
| `--force-mode` | `ad` / `ik` | `ik` 需要 `midtown`/`quads` |
| `--accuracy` | 例如 `1e-4` | 反解网格间距的精度档（`pme_accuracy`） |
| `--spacing` | 数值 | 直接钉死网格间距，做同网格对照时用 |
| `--sr-only` | — | 丢掉长程模块，做短程基线 |
| `--no-compress` | — | 出 graph 下界而不是 compact canonical（多卡用不到） |

> 换 backend/阶数会改变**网格尺寸**，所以跨配置比较必须先对齐精度档（`--accuracy`）或直接
> 钉同一个 `--spacing`，否则比的是两套不同的离散误差——见 `docs/GRID_FROM_ACCURACY.md`、
> `docs/SPLINE_MATRIX_BENCH.md`。

为什么不做成运行期开关：样条的阶数/族决定**网格尺寸**与**算子内核的编译分支**，两者都在
冻结期固定；若运行期可切换，要么每次切换重编程序，要么在每个 kernel 里带多份实现，代价
都高于"改设置就重冻一次"（权重不重训，几分钟）。

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
4. 分布式网格（§13.1）：
   - **AD 模型**：`SOG_MIDTOWN_SLAB=1` 下 `run 0` 的 `pe` 与 replicated 一致（力到 ~1e-6）。
   - **ik 模型**：只设 `SOG_MIDTOWN_SLAB=1` 时必须看到警告并**跑完**（回落 replicated，
     `pe` 与不带 env 逐位相同）；再加 `SOG_SLAB_IK=1` 时不应再有警告，`pe` 仍相同、力到
     ~1e-6（分布式 ik 真的走了）。
   - 想让配置错误立刻暴露，加 `SOG_SLAB_STRICT=1`：应变成可读的 undefined-output 错误
     而不是 `'self'`。
5. 分布式 ik 的网格守卫：`nz/np` 必须 ≥ 模板宽度（4 阶 → 4，6 阶 → 6）；小盒子跑
   `processors 1 1 4` 配 6 阶会报 `is thinner than the stencil`，换细网格或减少 rank。
