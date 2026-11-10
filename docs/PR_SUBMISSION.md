# 提 PR：范围、环境、编译命令、给用户的说明

本文给三样东西：**这次 PR 到底改了什么**、**管理员在自己的机器上怎么编**、**给用户的一段话**。
最后一节是可以直接粘贴的英文 PR 正文。

---

## 1. 范围

**仓库**：`/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt`，分支 `feat/long-range-pt`。

**一次性涉及两个主题，建议拆成两个 PR**：

| PR | 主题 | 提交 |
|---|---|---|
| **A** | **算子异常可读化 + ik×分布式回落 + 网格显存预检查**（纯 bugfix） | `1e08b6e92`、`314e7492b`、`5b0229ad6`、`76fa2cac9` |
| **B** | **长程网格/后端**：QuadS 后端、`ik` 力模式、实测网格误差律、SOG ladder 默认可空、长程核训练调度 | `f991734a5`、`708dc6aae`、`09b76ec29` |
| **C**（文档） | 网格误差模型/力模式基准、冻结 SOP、两棵树的分工、z-slab 部署结论、`pair deepmd/kk` 使用说明 | `fc7f43038`、`d30706c09`、`29a2efcd2`、`3686504e4`、`672af5c04`、`fa9d04da2` |
| **D**（分布式网格，独立） | z-slab 支持任意进程网格（`full_brick`） | `105e04c66` |

PR A 只碰四个文件，是纯收益：

| 文件 | 行数 | 作用 |
|---|---|---|
| `source/api_cc/include/commonPT.h` | +54/−3 | `require_defined_outputs()`；`flatten_canonical_atom_virial` 先判 `defined()` |
| `source/api_cc/src/DeepPotPTExpt.cc` | +12 | 四处 `extract_outputs` 后校验 |
| `source/op/pt/long_range/latent_charge_sum.cu` | +38/−8 | ik × 分布式 → 警告 + 回落；`SOG_SLAB_STRICT=1` 恢复硬错 |
| `source/op/pt/long_range/midtown_sum.cu` | +18 | 网格显存预检查 |

> **不要**把 `docs/SLAB_IK_FAILURE_DIAGNOSIS.md` 之类的内部台账塞进 PR A；PR 只需要
> `PAIR_DEEPMD_KK_SOG.md` 的 §12 作为说明（或直接写进 PR 正文）。

---

## 2. 环境（管理员复现用）

```bash
# 1) 编译器 / 构建系统 / MPI
source /etc/profile.d/modules.sh
module load gcc/13 cmake/3.26.3 openmpi/4.1.4

# 2) Python：3.11 + torch 2.12.1+cu126（本分支的算子 ABI 按它编）
conda activate lr212          # /data/home/jiangzhen/.conda/envs/lr212

# 3) CUDA toolkit：12.4（系统默认的 /usr/local/cuda 是 11.8，必须显式指定）
ls /usr/local/cuda-12.4/bin/nvcc

# 4) LAMMPS 源码（只为编 plugin 目标，不需要先编 LAMMPS）
ls $REPO/lammps_src/lammps-stable_22Jul2025_update2/src
```

版本对照（复现时对不上会编不过或运行期崩）：

| 组件 | 版本 |
|---|---|
| torch | 2.12.1+cu126 |
| CUDA toolkit | 12.4.131 |
| gcc | 13.x |
| cmake | 3.26.3 |
| openmpi | 4.1.4 |
| CUDA arch | `89`（RTX 4090）；换卡改 `-DCMAKE_CUDA_ARCHITECTURES` |

> 仓库根目录的 `AGENTS.md` 描述的是**上游 DeePMD-kit** 的 `uv pip install -e .` 流程，
> 不适用于本分支：这里要同时编 CUDA 算子、MPI、以及 LAMMPS 插件，走下面的 cmake 直编。

---

## 3. 编译命令（从零到可跑）

```bash
export REPO=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
export BUILD=$REPO/build-lmp-mpi-212
export TORCH=/data/home/jiangzhen/.conda/envs/lr212/lib/python3.11/site-packages/torch
export CUDA=/usr/local/cuda-12.4
export MPICC=/data/app/openmpi-4.1.4/install/openmpi/4.1.4/bin/mpiCC
export PYBIN=/data/home/jiangzhen/.conda/envs/lr212/bin/python
export LMP_SRC=$REPO/lammps_src/lammps-stable_22Jul2025_update2
export INSTALL=/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib

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
  -DPython_EXECUTABLE="$PYBIN" \
  -DUSE_PT_PYTHON_LIBS=ON -DBUILD_CPP_IF=ON -DBUILD_PY_IF=OFF \
  -DLAMMPS_SOURCE_ROOT="$LMP_SRC" -DDEEPMD_LAMMPS_KOKKOS=OFF

# 只验 PR A 的话，前三个目标就够；deepmd_lmp 是 LAMMPS 插件
make -C "$BUILD" deepmd_op_pt deepmd_cc deepmd_backend_ptexpt deepmd_c deepmd_lmp -j 24

mkdir -p "$INSTALL"
cp -a "$BUILD/op/pt/libdeepmd_op_pt.so" \
      "$BUILD/api_cc/libdeepmd_cc.so" \
      "$BUILD/api_cc/libdeepmd_backend_ptexpt.so" \
      "$BUILD/api_c/libdeepmd_c.so" \
      "$BUILD/lib/libdeepmd.so" \
      "$BUILD/lib/src/gpu/libdeepmd_op_cuda.so" \
      "$BUILD/lib/src/gpu/cudart/libdeepmd_dyn_cudart.so" \
      "$INSTALL/"
```

一键包装（同一条命令 + schema 校验 + plugin 软链）：
`sog_work/mdsuite/build_lrpt212_full.sh --build`。

**必须知道的三点**：

1. **`PATH` 里的 `nvcc` 默认是 11.8**，会遇到 `nvcc fatal: Value 'c++20' is not
   defined for option 'std'`——用上面的 `-DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc` 或
   `export PATH=/usr/local/cuda-12.4/bin:$PATH`。
2. **`nvcc` 与 `nvcc_wrapper` 不能同时出现**：`LAMMPS_SOURCE_ROOT` 存在时插件单独编；
   本 PR 的改动不涉及 Kokkos，`-DDEEPMD_LAMMPS_KOKKOS=OFF` 即可。
3. **仓库里的 `deepmd/lib/libdeepmd_op_pt.so` 必须与运行时环境同 torch 版本**
   （这里 = torch 2.12）。用 `lr` 环境（torch 2.11）跑同一个文件会报
   `RuntimeError: The version of …`。

---

## 4. 验证（管理员跑这四条即可）

### 4.1 纯 Python（不需 GPU）

```bash
cd $REPO && $PYBIN -m pytest \
  source/tests/common/dpmodel/test_grid_rule.py \
  source/tests/common/dpmodel/test_long_range.py -q
# 期望：82 passed
```

### 4.2 算子（需要一张 4090）

```bash
cd $REPO
export PYTHONPATH=$REPO
export LD_LIBRARY_PATH=$REPO/deepmd/lib:/usr/local/cuda-12.4/lib64:\
$TORCH/lib:/data/home/jiangzhen/.conda/envs/lr212/lib/python3.11/site-packages/nvidia/nccl/lib:\
/data/app/openmpi-4.1.4/install/openmpi/4.1.4/lib:/data/home/jiangzhen/.conda/envs/lr212/lib
$PYBIN -m pytest source/tests/pt_expt/kernels/test_mesh_backend.py -q
# 期望：20 passed
```

### 4.3 端到端：ik × 分布式网格（PR A 的主场景）

```bash
# 运行时环境（$TORCH / $BUILD / $INSTALL 沿用 §3 的 export）
export PATH=/data/app/openmpi-4.1.4/install/openmpi/4.1.4/bin:$PATH
NVD=/data/home/jiangzhen/.conda/envs/lr212/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=/data/app/openmpi-4.1.4/install/openmpi/4.1.4/lib:\
/data/home/jiangzhen/.conda/envs/lr212/lib:$TORCH/lib:\
$NVD/cuda_runtime/lib:$NVD/cuda_cupti/lib:$NVD/nvjitlink/lib:$NVD/cudnn/lib:\
$NVD/cublas/lib:$NVD/cusparse/lib:$NVD/cusparselt/lib:$NVD/curand/lib:\
$NVD/cufft/lib:$NVD/cufile/lib:$NVD/nccl/lib:$NVD/nvshmem/lib:\
/data/home/public/jiangzhen/dp/lmp_kokkos_install/lib:$INSTALL
export DP_BACKEND_PLUGIN_PATH=$INSTALL
export LAMMPS_PLUGIN_PATH=/tmp/lmpplugin_lrpt
ln -sfn "$BUILD/lmp/plugin/libdeepmd_lmp.so" /tmp/lmpplugin_lrpt/dpplugin.so

SOG_MIDTOWN_SLAB=1 lmp -in in.mid4ik.lammps -log log.a
# 期望：stderr 一行 [W..] "… runs the replicated solve …"，且 pe 与不带 env 的 run 相同
```

（`in.mid4ik.lammps` 只需 `units metal / atom_style charge / read_data <任意 192 原子 data> /
pair_style deepmd <lr_mid4ik.pt2> / pair_coeff * * O H / fix 1 all nve / run 6`。
完整的运行脚本见 `examples/lammps/deepmd_kk_sog/run.sh` 与
`sog_work/bench/matrix_slab_vs_rep/`。）

### 4.4 三条断言（写进 PR 正文的验收标准）

| 断言 | 现象 |
|---|---|
| ik + `SOG_MIDTOWN_SLAB=1` 不再中止 | `pe` 与不带 env 逐位相同 |
| `SOG_SLAB_STRICT=1` 时错误可读 | 报 undefined 输出**名称** + 指向 stderr 的 `[E...]`，不再是 `'self'` |
| 超大网格快速失败 | 1125³ / 24 GiB 卡上 **10 秒**内报 `needs 26.52 GiB … free`（以前约 1 分钟后 OOM） |

---

## 5. 给用户的一段话（release note / 群公告）

> **长程模型的两处使用变化**
>
> 1. **在 LAMMPS 里给 ik 模型设 `SOG_MIDTOWN_SLAB=1`（或 `SOG_MIDTOWN_DIST=1`）不再报错中止。**
>    分布式网格只实现 AD 力，所以这类调用会自动回落到 replicated 求解——**力定律完全相同**，
>    只是每个 rank 持有全网格。日志里会打一次 `[W…] runs the replicated solve`。
>    批处理若不想悄悄换成更吃显存的路径，设 `SOG_SLAB_STRICT=1` 保留硬报错。
> 2. **`.pt2` 里的自定义算子出错时，报错会点名。** 以前任何算子异常都会显示成
>    `Expected a proper Tensor … 'self'`；现在会列出未定义的输出槽位，并提示真因在 stderr 的
>    `[E... shim_common.cpp]` 行。Python 路径也不再静默返回 `nan`。
>
> 详见 `docs/PAIR_DEEPMD_KK_SOG.md` §12。

---

## 6. 英文 PR 正文（可直接粘贴）

```markdown
### Motivation

Two failure modes reported from MD runs on exported `.pt2` models produced the
same unhelpful message:

    ERROR: ... Expected a proper Tensor but got None ... for argument #0 'self'

1. A custom operator inside the graph threw, the AOTI proxy executor caught the
   exception (printing only one `[E...] shim_common.cpp]` line to stderr) and
   left that operator's outputs undefined.  The failure then surfaced as the
   first `reshape` on a None, naming neither the operator nor the cause.  Under
   `SOG_MIDTOWN_SLAB=1` with `force_mode=ik` this aborted an entire MD run
   from an environment variable that is usually set process-wide; the same
   message also hid a mesh that did not fit the card -- two unrelated causes,
   one string.  On the Python path the call did not even raise: it returned NaN.

2. A distributed mesh (z-slab / local-spread) has no `i*k` differentiation.  It
   used to refuse ik with `TORCH_CHECK`, which is correct physics but a bad
   failure mode for a parallelism hint.

### Changes

* `require_defined_outputs()` in `source/api_cc/include/commonPT.h` reports the
  undefined output slots and points at the operator's own `[E...]` line.  It runs
  after every `extract_outputs` (main compute, its second branch, and both
  charge-response paths), and `flatten_canonical_atom_virial` no longer reshapes
  a None.
* ik against a distributed mesh now warns once and falls back to the replicated
  solve -- a parallelism change, not a physics change: the replicated solve
  evaluates the same force law.  `SOG_SLAB_STRICT=1` restores the hard error.
* The replicated mesh is checked against `cudaMemGetInfo` (`20 B` per grid point)
  before the fp64 green table is built, so an oversized grid fails in seconds
  with the requirement and the free memory instead of after tens of GB of host
  allocation.

### Backward compatibility

No numerical change.  The fallback produces bit-identical energies to the run
without `SOG_MIDTOWN_SLAB` (verified np=1 and np=2 on a 192-atom cell).  The
strict env var preserves the previous abort-on-misconfiguration behaviour for
batch runs.

### How to build

```bash
module load gcc/13 cmake/3.26.3 openmpi/4.1.4
conda activate lr212          # python 3.11, torch 2.12.1+cu126
cmake -S source -B build \
  -DENABLE_PYTORCH=ON -DENABLE_TENSORFLOW=OFF -DENABLE_JAX=OFF -DENABLE_PADDLE=OFF \
  -DUSE_CUDA_TOOLKIT=ON \
  -DCMAKE_PREFIX_PATH=$TORCH \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DDEEPMD_BYPASS_TORCH_CUDA_CHECK=ON -DDEEPMD_CUDA_PORTABLE_PTX=ON \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DCMAKE_CXX_COMPILER=mpiCC -DMPI_CXX_COMPILER=mpiCC \
  -DUSE_PT_PYTHON_LIBS=ON -DBUILD_CPP_IF=ON -DBUILD_PY_IF=OFF \
  -DLAMMPS_SOURCE_ROOT=$LAMMPS_SRC -DDEEPMD_LAMMPS_KOKKOS=OFF
make deepmd_op_pt deepmd_cc deepmd_backend_ptexpt deepmd_c deepmd_lmp -j 24
```

Note: the default `nvcc` on this image is 11.8, which rejects `-std=c++20`; point
`CMAKE_CUDA_COMPILER` at CUDA 12.4 explicitly.

### Tests

* `pytest source/tests/common/dpmodel/test_grid_rule.py test_long_range.py` (82)
* `pytest source/tests/pt_expt/kernels/test_mesh_backend.py` (20, needs one GPU)
* End to end: an ik artifact under `SOG_MIDTOWN_SLAB=1` now completes with the
  same energy as without it; with `SOG_SLAB_STRICT=1` it fails with the
  undefined-output message instead of `'self'`.
```
