# 编译与测试：可复现全流程

本文把本轮工作涉及的两棵树、两条链的**全套编译命令、安装布局、测试**和**文件/数据路径**固定下来。
配套文档：[网格引擎测试与复现](MESH_ENGINE_TEST_REPRO.md)（编译、冻结工件清单、四层测试、基准与论文表格的再生）、冻结 [`FREEZE_SOP.md`](FREEZE_SOP.md)、基准 [`MIDTOWN_VS_BSPLINE_BENCH.md`](MIDTOWN_VS_BSPLINE_BENCH.md)、
网格规则 [`GRID_FROM_ACCURACY.md`](GRID_FROM_ACCURACY.md)、ik 精度 [`IK_MODE_ACCURACY.md`](IK_MODE_ACCURACY.md)。

- **A 链**（图内长程、生产 MD）：`/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt`（lr-pt，`latent_charge`）。
- **B 链**（LAMMPS 插件 `kspace_style sog`）：`/data/home/public/jiangzhen/dp/deepmd-kit-LR`（sezm，`sog_energy`）。
- 谱系锁：两树不能互跑对方模型（见 `FREEZE_SOP.md`）。

---

## 0. 环境

```bash
source /etc/profile.d/modules.sh
module load gcc/13 cmake/3.26.3 openmpi/4.1.4
conda activate lr212          # /data/home/jiangzhen/.conda/envs/lr212
```

| 组件 | 路径 / 版本 |
|---|---|
| Python / torch | `/data/home/jiangzhen/.conda/envs/lr212/bin/python`，torch 2.12.1+cu126 |
| CUDA | `/usr/local/cuda-12.4`，`CMAKE_CUDA_ARCHITECTURES=89`（RTX 4090） |
| MPI C++ | `/data/app/openmpi-4.1.4/install/openmpi/4.1.4/bin/mpiCC` |
| LAMMPS（二进制） | `/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp`（Kokkos 版，预编译） |
| LAMMPS（源码） | `/data/home/public/jiangzhen/lammps/lammps_stable`（B 链 kk 插件用）；A 链插件用 `deepmd-kit-LR-lr-pt/lammps_src/lammps-stable_22Jul2025_update2` |
| C++ 标准 | C++20（插件 ABI 需要 `std::format`） |

---

## 1. A 链（lr-pt）全套编译

产物是一个 CMake 构建目录 `build-lmp-mpi-212`（**五个目标**）：
`deepmd_op_pt`（含 `force_mode` 的 canonical op，torch 扩展）、`deepmd_cc`、`deepmd_backend_ptexpt`、
`deepmd_c`、`deepmd_lmp`（LAMMPS 插件）。

> 说明：当前工作用的 `libdeepmd_op_pt.so` 实际在兄弟目录 `build-pt-mpi-gcc-212` 里单独编（同一 `source/`、
> 同一套选项），两者等价；`build-lmp-mpi-212` 里同样暴露 `deepmd_op_pt` 目标，可一并构建。

### 1.1 配置 + 编译（一条命令脚本）

**一键**（configure + build 5 目标 + 安装 + 软链 + 校验 schema）：

```bash
JOBS=20 /data/home/public/jiangzhen/dp/sog_work/mdsuite/build_lrpt212_full.sh
```

下面是等价的分解命令（`build_lrpt212_api.sh` 是仅配置/仅 3 目标的旧脚本，`--build` 不含 op_pt/插件）。
核心配置：

```bash
REPO=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
BUILD=$REPO/build-lmp-mpi-212
TORCH=/data/home/jiangzhen/.conda/envs/lr212/lib/python3.11/site-packages/torch
CUDA=/usr/local/cuda-12.4
MPICC=/data/app/openmpi-4.1.4/install/openmpi/4.1.4/bin/mpiCC
LMP_SRC=$REPO/lammps_src/lammps-stable_22Jul2025_update2

cmake -S "$REPO/source" -B "$BUILD" \
  -DENABLE_PYTORCH=ON -DENABLE_TENSORFLOW=OFF -DENABLE_JAX=OFF -DENABLE_PADDLE=OFF \
  -DUSE_CUDA_TOOLKIT=ON \
  -DCMAKE_PREFIX_PATH="$TORCH" \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA" -DCUDAToolkit_ROOT="$CUDA" \
  -DCUDAToolkit_NVCC_EXECUTABLE="$CUDA/bin/nvcc" -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DDEEPMD_BYPASS_TORCH_CUDA_CHECK=ON -DDEEPMD_CUDA_PORTABLE_PTX=ON \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DCMAKE_CXX_COMPILER="$MPICC" -DMPI_CXX_COMPILER="$MPICC" \
  -DPython_EXECUTABLE=/data/home/jiangzhen/.conda/envs/lr212/bin/python \
  -DUSE_PT_PYTHON_LIBS=ON -DBUILD_CPP_IF=ON -DBUILD_PY_IF=OFF \
  -DLAMMPS_SOURCE_ROOT="$LMP_SRC" -DDEEPMD_LAMMPS_KOKKOS=OFF

make -C "$BUILD" deepmd_op_pt deepmd_cc deepmd_backend_ptexpt deepmd_c deepmd_lmp -j 20
```

产物：

```
build-lmp-mpi-212/op/pt/libdeepmd_op_pt.so               # torch 扩展（canonical op，含 force_mode）
build-lmp-mpi-212/api_cc/libdeepmd_cc.so
build-lmp-mpi-212/api_cc/libdeepmd_backend_ptexpt.so
build-lmp-mpi-212/api_c/libdeepmd_c.so
build-lmp-mpi-212/lib/libdeepmd.so
build-lmp-mpi-212/lib/src/gpu/libdeepmd_op_cuda.so
build-lmp-mpi-212/lib/src/gpu/cudart/libdeepmd_dyn_cudart.so
build-lmp-mpi-212/lmp/plugin/libdeepmd_lmp.so            # LAMMPS 插件
```

### 1.2 安装到运行时目录 + 插件符号链接

运行时（bench/生产）从两个位置取库：
- **C++ API**：插件 RUNPATH 直接指向 `build-lmp-mpi-212`；
- **后端 + op_pt**：`DP_BACKEND_PLUGIN_PATH` / `LD_LIBRARY_PATH` 的 `lr212_install_lrpt/deepmd/lib`。

把（重新编出来的）库同步到 install，再挂插件软链。**重编后必须整组同步**，不要新旧混用（见 §1.3）：

```bash
INSTALL=/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib
B=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt/build-lmp-mpi-212
cp -a $B/op/pt/libdeepmd_op_pt.so $B/api_cc/libdeepmd_cc.so \
      $B/api_cc/libdeepmd_backend_ptexpt.so $B/api_c/libdeepmd_c.so \
      $B/lib/libdeepmd.so $B/lib/src/gpu/libdeepmd_op_cuda.so \
      $B/lib/src/gpu/cudart/libdeepmd_dyn_cudart.so "$INSTALL/"

# 插件软链
mkdir -p /tmp/lmpplugin_lrpt
ln -sfn $B/lmp/plugin/libdeepmd_lmp.so /tmp/lmpplugin_lrpt/dpplugin.so
```

### 1.3 关键坑：canonical op 的 schema 从 21→22 参数

`dpa4c_canonical_compress_long_range_energy_force` 新增了 `int force_mode` 参数（`source/op/pt/dpa4c/graph_compress.cu`，
由 `libdeepmd_op_pt.so` 注册）。**重编 op 源后，必须把新 `libdeepmd_op_pt.so` 覆盖进 install**，
否则新冻结的 `.pt2`（带 `force_mode`）会报：

```
flatten_canonical_atom_virial: Expected a proper Tensor but got None ... 'self'
```

自检（应看到 `..., int mesh_backend=0, int force_mode=0) ->`）：

```bash
strings /data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib/libdeepmd_op_pt.so \
  | grep -m1 "int mesh_backend=0, int force_mode=0)"
```

---

## 2. B 链（deepmd-kit-LR）插件编译（含 GPU B-spline）

B 链的 `kspace_style sog`（含本树新增的 GPU B-spline：`source/lmp/sog_gpu.cu`、`sog.cpp`）编进**Kokkos 插件**
`deepmd_lmp.so`。两个构建目录：

```bash
# (a) C++ API + op_pt（deepmd-kit-LR/source）
cmake -S /data/home/public/jiangzhen/dp/deepmd-kit-LR/source -B /tmp/build_sog212 \
  -DENABLE_PYTORCH=ON -DENABLE_TENSORFLOW=OFF -DUSE_CUDA_TOOLKIT=ON \
  -DCMAKE_PREFIX_PATH=/data/home/jiangzhen/.conda/envs/lr212/lib/python3.11/site-packages/torch \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/data/app/openmpi-4.1.4/install/openmpi/4.1.4/bin/mpiCC \
  -DCMAKE_INSTALL_PREFIX=/data/home/public/jiangzhen/dp/lr212_install
make -C /tmp/build_sog212 deepmd_op_pt deepmd_cc deepmd_backend_ptexpt deepmd_c -j 20

# (b) Kokkos 插件（含 sog_gpu.cu；CXX=nvcc_wrapper）
cmake -S /data/home/public/jiangzhen/dp/deepmd-kit-LR/source/lmp/plugin_kokkos -B /tmp/dplmp_kk212 \
  -DLAMMPS_SOURCE_ROOT=/data/home/public/jiangzhen/lammps/lammps_stable \
  -DKOKKOS_BUILD_DIR=/data/home/public/jiangzhen/lammps/lammps_stable/build_kokkos \
  -DCMAKE_CXX_COMPILER=/data/home/public/jiangzhen/lammps/lammps_stable/lib/kokkos/bin/nvcc_wrapper \
  -DCMAKE_INSTALL_PREFIX=/data/home/public/jiangzhen/dp/lr212_install
make -C /tmp/dplmp_kk212 -j 20          # -> /tmp/dplmp_kk212/deepmd_lmp.so
```

```bash
mkdir -p /tmp/lmpkk_plugin_212
ln -sfn /tmp/dplmp_kk212/deepmd_lmp.so /tmp/lmpkk_plugin_212/dpplugin.so
```

> `-k on g 1` 跑 Kokkos GPU；插件加载要求 `LAMMPS_PLUGIN_PATH` 指向**干净目录**（只放 `dpplugin.so` 软链），
> 不能直接指向 CMake 构建目录（否则 `Loaded 0 plugins`）。

---

## 3. 测试

| 测试 | 命令 | 结果 |
|---|---|---|
| 网格规则 / ik 规则单测 | `python -m pytest source/tests/common/dpmodel/test_grid_rule.py -q` | **21 passed** |
| GPU B-spline 逐位一致（B 链） | 见 §3.2 | **3 passed** |
| **多卡 np=1 vs np=2/4 力一致**（A 链） | 见 §3.4 | **6 passed**（含 ik） |
| 冻结 + `run 0` 冒烟 | 见 §3.3 | 见 `FREEZE_SOP.md` |
| 端到端基准 | 见 `MIDTOWN_VS_BSPLINE_BENCH.md` §9 | — |

### 3.1 网格规则

```bash
cd /data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
/data/home/jiangzhen/.conda/envs/lr212/bin/python -m pytest \
  source/tests/common/dpmodel/test_grid_rule.py -q
# 21 passed（含 IK_FORCE_RULE、ik 网格取 min(能量,力)、ik 比 AD 粗、参考核 291³/177³）
```

### 3.2 GPU B-spline 逐位一致

脚本：`/data/home/public/jiangzhen/dp/deepmd-kit-LR/source/lmp/tests/test_sog_gpu_bspline.py`
（CPU vs GPU，`spline bspline_{4,5,6} use_gpu yes|no`，力 `max|ΔF|`、能量 `|ΔE|`）。

```bash
cd /data/home/public/jiangzhen/dp/deepmd-kit-LR
ENV=/data/home/jiangzhen/.conda/envs/lr212; TORCH=$ENV/lib/python3.11/site-packages/torch
NVD=$ENV/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="/data/app/openmpi-4.1.4/install/openmpi/4.1.4/lib:$ENV/lib:$TORCH/lib:$NVD/cuda_runtime/lib:$NVD/cufft/lib:$NVD/curand/lib:$NVD/nvjitlink/lib:$NVD/cusparse/lib:/data/home/public/jiangzhen/dp/lmp_kokkos_install/lib:/data/home/public/jiangzhen/dp/lr212_install/deepmd/lib"
export LAMMPS_EXECUTABLE=/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp
export SOG_PLUGIN_DIR=/tmp/lmpkk_plugin_212
export SOG_TEST_LDLIB="$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
$ENV/bin/python -m pytest source/lmp/tests/test_sog_gpu_bspline.py -q
# 3 passed
```

### 3.3 冻结 + 冒烟
A 链冻结（canonical/`--compress`）用 `sog_work/tools/freeze_lr_variants.py`（详见 `FREEZE_SOP.md`）：

```bash
PY=/data/home/jiangzhen/.conda/envs/lr212/bin/python
TOOL=/data/home/public/jiangzhen/dp/sog_work/tools/freeze_lr_variants.py
OUT=/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/midtown_vs_bspline
CUDA_VISIBLE_DEVICES=0 $PY $TOOL --out $OUT/lr_mid4ik.pt2 --backend midtown --order 4 --force-mode ik
```

冒烟（`run 0` 出能量、无 ERROR）用 `sog_work/bench/bench_250k_lrpt.py` 的 `run_one`，或直接 LAMMPS：

```bash
INSTALL=/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib
LD_LIBRARY_PATH=<见 bench ENV_BASE>:$INSTALL \
DP_BACKEND_PLUGIN_PATH=$INSTALL \
LAMMPS_PLUGIN_PATH=/tmp/lmpplugin_lrpt \
/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp -in in.file -log log
```

### 3.4 多卡（np>1）一致性

**多卡无需额外编译**（replicated 网格路径已在标准构建里）。多卡只要求：
canonical/`--compress` 下界（np>1 唯一可用形态）、输入加 `processors 1 1 np`、每 rank 一张 GPU。
默认多卡是 **replicated 网格**（每个 rank 一份完整网格、原子做域分解），所以 **`ik` 在 np>1 也正确**；
显式的 `SOG_MIDTOWN_SLAB` / `SOG_MIDTOWN_DIST`（省网格显存）**只支持 AD**，设了这两个 env 时 `ik` 会报错。

测试文件：`sog_work/bench/test_multi_rank_parity.py`（每个冻结变体跑 np=1 与 np=2/4，要求
`max|ΔF|/⟨|F|⟩ < 1e-5`，实测 ~8e-7）：

```bash
cd /data/home/public/jiangzhen/dp/sog_work/bench
SOG_NP=2,4 /data/home/jiangzhen/.conda/envs/lr212/bin/python -m pytest test_multi_rank_parity.py -q
```

多卡整步/显存基准：`bench_lr_variants.py` 支持 `NP=2,4`（自动加 `processors 1 1 np`、每 rank 一张卡）。

---

## 4. 文件 / 数据路径清单

**代码 / 脚本**

| 用途 | 路径 |
|---|---|
| A 链树 | `/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt` |
| B 链树 | `/data/home/public/jiangzhen/dp/deepmd-kit-LR` |
| 网格规则（含 ik） | `<A>/deepmd/dpmodel/long_range/grid_rule.py` |
| ik/force_mode 贯穿 | `<A>/deepmd/pt_expt/kernels/long_range.py`、`pt_expt/long_range/latent_charge.py`、`pt_expt/kernels/dpa4c/canonical_long_range.py` |
| canonical op（force_mode） | `<A>/source/op/pt/dpa4c/graph_compress.cu` |
| GPU B-spline | `<B>/source/lmp/sog_gpu.cu`、`sog.cpp`、`sog_gpu.cuh` |
| A 链编译脚本 | `/data/home/public/jiangzhen/dp/sog_work/mdsuite/build_lrpt212_api.sh` |
| A 链全套编译+装脚本 | `/data/home/public/jiangzhen/dp/sog_work/mdsuite/build_lrpt212_full.sh` |
| 冻结变体工具 | `/data/home/public/jiangzhen/dp/sog_work/tools/freeze_lr_variants.py` |
| 基准（整步/显存） | `/data/home/public/jiangzhen/dp/sog_work/bench/bench_lr_variants.py`、`bench_250k_lrpt.py` |
| Fourier 隔离计时 | `/data/home/public/jiangzhen/dp/sog_work/bench/time_mesh_op.py`（`ORD=4|6`） |
| B 链 Fourier | `/data/home/public/jiangzhen/dp/sog_work/bench/bench_sog_spline.py` |
| 出图 | `/data/home/public/jiangzhen/dp/sog_work/bench/plot_midtown_vs_bspline.py` |

**模型 / 数据**

| 用途 | 路径 |
|---|---|
| A 链 250K checkpoint | `/data/home/public/jiangzhen/dp/master-merge-test/water/run_250k_librule/model.ckpt-250000.pt` |
| 参考核（α=2.0/1.13） | `.../run_150k_widthfloor_archive/...`、`.../run_200k_noramp_archive/...` |
| A 链冻结产物 | `/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/midtown_vs_bspline/lr_{bs4,mid4,mid4ik,bs6,mid6,mid6ik,sr}.pt2` |
| 基准数据（LAMMPS） | `/data/home/public/jiangzhen/dp/sog_work/bench/data_{192,1536,...,98304}.lmp` |
| 基准结果 CSV | `.../matrix_midtown/bench_variants.{o4ik,o6ik}.np1.csv` |
| 图 | `.../midtown_vs_bspline_bench.png` |

**运行时 install**

| 用途 | 路径 |
|---|---|
| A 链运行时库 | `/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib/` |
| B 链运行时库 | `/data/home/public/jiangzhen/dp/lr212_install/deepmd/lib/` |
| A 链插件软链 | `/tmp/lmpplugin_lrpt/dpplugin.so` |
| B 链插件软链 | `/tmp/lmpkk_plugin_212/dpplugin.so` |
| LAMMPS 二进制 | `/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp` |
