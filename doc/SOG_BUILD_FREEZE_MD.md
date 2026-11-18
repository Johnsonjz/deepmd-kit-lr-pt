# SOG / DPA4C 编译、冻结与 LAMMPS MD 运行手册

更新：2026-11-15  
适用分支：`main`（`4d515ad63` 之后，含 B4/6 阶 slab/可选 VJP）  
生产设置：**CubeS₂-6 + slab + compressed canonical .pt2**

本文替代旧文档中 `full_lr` / `kspace_style sog` / `fix sog` 的部署说明；旧报告
（`MD_DEPLOYMENT_PARITY.md`、`MD_P0_REPORT.md`）保留为历史结果，最新流程以本文为准。

---

## 1. 环境

本机参考环境：

```bash
module load gcc/13 cmake/3.26.3 openmpi/4.1.4
CONDA=/data/home/jiangzhen/.conda/envs/lr212
REPO=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
INSTALL=/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib
LMP=/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp
MPI=/data/app/openmpi-4.1.4/install/openmpi/4.1.4/bin/mpirun
```

Python 运行时需要：

```bash
NVD=$CONDA/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$CONDA/lib:$CONDA/lib/python3.11/site-packages/torch/lib:\
$NVD/cuda_runtime/lib:$NVD/cuda_cupti/lib:$NVD/cudnn/lib:$NVD/cublas/lib:\
$NVD/cusparse/lib:$NVD/cusparselt/lib:$NVD/nvjitlink/lib:$NVD/curand/lib:\
$NVD/cufft/lib:$NVD/cufile/lib:$NVD/nccl/lib:$NVD/nvshmem/lib:$INSTALL
export DP_BACKEND_PLUGIN_PATH=$INSTALL
export PYTHONPATH=$REPO:$PYTHONPATH
export OMP_NUM_THREADS=1
```

LAMMPS 侧还需要：

```bash
export LAMMPS_PLUGIN_PATH=/tmp/kk_lrpt_plugin
```

`/tmp/kk_lrpt_plugin/dpplugin.so` 是 Kokkos `pair deepmd/kk` 插件；它由 LAMMPS
Kokkos 构建流程单独生成。换机器时重建插件或把它放到稳定路径，不要依赖 `/tmp`。

---

## 2. 编译 lr-pt 的算子

只改 C++/CUDA 算子时：

```bash
cd $REPO
make -C build-lmp-mpi-212 deepmd_op_pt -j 20
```

产物：

```
build-lmp-mpi-212/op/pt/libdeepmd_op_pt.so
```

安装到运行时和仓库内：

```bash
cp build-lmp-mpi-212/op/pt/libdeepmd_op_pt.so $INSTALL/
cp build-lmp-mpi-212/op/pt/libdeepmd_op_pt.so deepmd/lib/
```

验证：

```bash
cd $REPO
$CONDA/bin/python -m pytest -q \
  source/tests/pt_expt/kernels/test_mesh_backend.py \
  source/tests/pt_expt/kernels/test_long_range_sum.py
```

预期当前 `main`：**70 passed**；加上 `test_dpa4c_graph_lower.py` 为 **101 passed**。

> 只改 Python（模型/训练/文档）不需要重编 `.so`；只改 `source/op/pt/*` 才需要。

---

## 3. 冻结为 compressed canonical `.pt2`

### 3.1 直接冻结

```bash
cd <run_dir>
CC=gcc CXX=g++ $CONDA/bin/python - <<'PY'
from deepmd.main import main
import sys
sys.argv = ["dp", "--pt-expt", "freeze",
            "-c", "model.ckpt-200000.pt",
            "-o", "frozen",
            "--compress"]
main()
PY
```

如果 `dp` 已在 PATH：

```bash
CC=gcc CXX=g++ dp --pt-expt freeze -c model.ckpt-200000.pt -o frozen --compress
```

关键点：

- **必须加 `--compress`**：Kokkos `pair deepmd/kk` 的周期长程只接受
  compressed canonical DPA4C。未压缩的 `.pt2` 会报：
  `only through a compressed DPA4C model frozen for a CUDA device`。
- **`CC=gcc CXX=g++`**：如果环境里 load 了 OpenMPI，`CXX` 可能是 `mpiCC`，
  PyTorch Inductor 预编译头会报 `undefined reference to main`；显式用 `g++`。
- 产物：`frozen.pt2`（及同目录中间件；最终给 LAMMPS 的是 `.pt2`）。

### 3.2 不重训，切换 PME → midtown 6 阶

训练常用 PME `pme_order=8`；LAMMPS slab 只支持 4/5/6。**不用重训**，改
checkpoint 的 `_extra_state` 再冻：

```python
import torch
src = "model.ckpt-20000.pt"
dst = "model.ckpt-20000-mid6.pt"
ck = torch.load(src, map_location="cpu", weights_only=False)
lr = ck["model"]["_extra_state"]["model_params"]["long_range"]
lr["pme_order"] = 6
lr["mesh_backend"] = "midtown"
lr["force_mode"] = "ad"
torch.save(ck, dst)
```

然后用 §3.1 冻结 `model.ckpt-20000-mid6.pt`。权重不变，只换倒空间求解器；
20k 模型实测可正常加载/运行。

生产 6 阶工件：

```
/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/dpa4_campaign_20261114/dpa4c-sog-mid6ad.pt2
```

### 3.3 冻结校验

最小校验：同一构型，checkpoint 前向 vs LAMMPS `run 0` 的 energy/force 应一致到
数值精度（训练盒；跨盒差异来自 k 网格，见历史文档 §4）。

```bash
# LAMMPS run 0
$MPI --oversubscribe -np 1 $LMP -k on g 1 -in in.run0.lammps
```

`in.run0.lammps` 里至少要有：

```
units metal
boundary p p p
atom_style atomic
read_data <data>
pair_style deepmd/kk <frozen.pt2>
pair_coeff * * O H
run 0
```

模型侧用 `deepmd.infer.DeepPot(<checkpoint>)` 或 `DeepEval.eval_descriptor`
对同一坐标/盒前向。

---

## 4. LAMMPS MD 运行

### 4.1 最小周期输入

```
units           metal
boundary        p p p
atom_style      atomic
atom_modify     map yes

read_data       water.lmp
mass            1 16
mass            2 2

pair_style      deepmd/kk /path/to/frozen.pt2
pair_coeff      * * O H

velocity        all create 330.0 23456789
fix             1 all nvt temp 330.0 330.0 100.0
timestep        0.0005
thermo          100
run             10000
```

注意：

- 当前 compressed canonical 已把长程算在 `pair_style deepmd/kk` 内，
  **不要再写 `kspace_style sog` 或 `fix sog`**；写了会重复/冲突。
- `pair_coeff` 的顺序按模型 `type_map`（本生产模型是 `O H`）。
- `pair_style deepmd/kk` 需要 CUDA + Kokkos + 压缩 `.pt2`。

### 4.2 启动命令

```bash
# 2 卡 slab
CUDA_VISIBLE_DEVICES=0,1 \
SOG_MIDTOWN_SLAB=1 \
$MPI --oversubscribe -np 2 \
  $LMP -k on g 1 -in in.slab.np2.lammps -log log.np2

# 4 卡 slab
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SOG_MIDTOWN_SLAB=1 \
$MPI --oversubscribe -np 4 \
  $LMP -k on g 1 -in in.slab.np4.lammps -log log.np4
```

- **slab（生产，≥2 卡）**：设 `SOG_MIDTOWN_SLAB=1`。它把 mesh/FFT 按 z 分片，
  避免每个 rank 做 full-mesh FFT。
- **replicated**：不设 `SOG_MIDTOWN_SLAB`（或显式 `=0`）。只建议 1 卡/小体系；
  4 卡 replicated 比 slab 慢约 25%（1.12M 实测 48.72 vs 36.28 ms）。
- `SOG_SLAB_TIME=1` 可选，打印长程分项时间。
- `SOG_CANON_FUSE_J=1` 可选，开启 B4（J 融合进 energy fitting，1.12M np2
  约 −1.5 ms）；默认关，因为 B4 改变 fitting-route 的逐位累加顺序。
- `SOG_CANON_VJP=1` 可选，走 VJP 显存模式（时间 +4.2%，显存 −468 MiB）；
  生产默认不用。

### 4.3 生产/基准输入

| 用途 | deck |
|---|---|
| 1.12M slab 2 卡 | `sog_work/bench/matrix_slab1m/in.slab1m.1119744.np2.lammps` |
| 1.12M slab 4 卡 | `.../in.slab1m.1119744.np4.lammps` |
| 1M dpa4c 基线 | `.../in.dpa4c1m.1119744.np{1,2,4}.lammps` |
| 2M max-size | `sog_work/bench/matrix_maxsize/` |

当前生产数据：1.12M / 2M 水；生产模型 `dpa4c-sog-mid6ad.pt2`。

### 4.4 期望数字（1.12M，6 阶）

| np | dpa4c | SOG slab | SOG replicated |
|---:|---:|---:|---:|
| 1 | 84.89 ms | 102.57 ms | — |
| 2 | 51.57 ms | **64.64 ms** | 71.72 ms |
| 4 | 28.51 ms | **36.28 ms** | 48.72 ms |

显存（B4 后）：np2 5,370 MiB/GPU；np4 3,454 MiB/GPU。  
2M 6 阶：13,890 / 8,770 / 5,554 MiB（1/2/4 卡）。

---

## 5. 常见错误

| 报错 | 原因/处理 |
|---|---|
| `only through a compressed DPA4C model frozen for a CUDA device` | 冻结时漏了 `--compress`，或用了未压缩 `.pt2`；重新 `freeze --compress` |
| `slab B-spline backend supports order 4, 5 or 6, got 8` | checkpoint 仍是 PME order8；按 §3.2 改成 `pme_order=6, mesh_backend=midtown` 再冻（不用重训） |
| `undefined reference to main`（freeze 时） | `CXX=mpiCC` 被 Inductor 当链接器；`CC=gcc CXX=g++` |
| `undefined symbol: cudaGetDriverEntryPointByVersion` | `LD_LIBRARY_PATH`/CUDA runtime 版本不匹配；按 §1 补全 conda/torch/nvidia 库 |
| `CUDA error: invalid resource handle` | graph capture 与流/计时冲突；生产不要开 `SOG_CANON_TIME`；确认当前流补丁 |
| 能量/力与 checkpoint 差很多 | 检查 `pme_order/mesh_backend`、`type_map` 顺序、`pair_coeff` 映射、是否误加 `kspace_style sog` |

---

## 6. 测试

常用：

```bash
cd $REPO
$CONDA/bin/python -m pytest -q \
  source/tests/pt_expt/model/test_dpa4c_graph_lower.py \
  source/tests/pt_expt/kernels/test_mesh_backend.py \
  source/tests/pt_expt/kernels/test_long_range_sum.py
# 101 passed
```

完整相关 suite（PME/freeze/frozen SOG）：

```bash
$CONDA/bin/python -m pytest -q \
  source/tests/common/dpmodel/test_long_range.py \
  source/tests/common/dpmodel/test_grid_rule.py \
  source/tests/pt_expt/kernels/test_mesh_backend.py \
  source/tests/pt_expt/kernels/test_long_range_sum.py \
  source/tests/pt_expt/model/test_long_range.py \
  source/tests/pt_expt/test_long_range_kernel_schedule.py \
  source/tests/pt_expt/utils/test_canonical_graph.py
# 305 passed, 8 skipped
```

---

## 7. 相关文档

- 有效优化与收益：[`doc/SOG_EFFECTIVE_OPTIMIZATIONS.md`](SOG_EFFECTIVE_OPTIMIZATIONS.md)
- 大体系瓶颈与分阶段计时：`sog_work/docs/SOG_LARGE_SCALE_BOTTLENECKS.md`
- charge net 拆解与负结果：`sog_work/docs/SOG_CHARGE_NET_BREAKDOWN.md`
- 历史部署一致性：`sog_work/docs/MD_DEPLOYMENT_PARITY.md`（旧 full_lr 口径，仅供对照）
- 历史 P0 报告：`sog_work/docs/MD_P0_REPORT.md`（旧 full_lr 口径）
