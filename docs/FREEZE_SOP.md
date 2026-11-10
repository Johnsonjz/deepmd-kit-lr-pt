# 生产冻结 SOP：`--compress`（canonical 下界）

本文是 lr-pt 谱系（`latent_charge` 长程）模型**生产冻结**的标准流程。核心结论：

> **生产一律用 `--compress`（compact canonical 下界）。**
> 实测同一权重：canonical 比默认 graph 下界**快 ~3×、显存省 ~40×、能量逐位一致**，
> 而且是**唯一能跑多卡（np>1）**的形态。默认 graph 下界只能单卡。

配套：网格/精度规则见 [`GRID_FROM_ACCURACY.md`](GRID_FROM_ACCURACY.md)，长程移植台账见
[`LONG_RANGE_PORT.md`](LONG_RANGE_PORT.md)，**编译与测试见 [`BUILD_AND_TEST.md`](BUILD_AND_TEST.md)**。

---

## 0. 谱系锁（谁能用本文）

| 树 | 谱系 | 冻结命令 |
|---|---|---|
| **`deepmd-kit-LR-lr-pt`** | `fitting=ener` + `long_range=latent_charge` | 本文 |
| `deepmd-kit-LR` | `fitting=sog_energy`（LR 在拟合头内） | 另一套，见其文档 |

两棵树**不能互相跑对方的模型**（`latent_charge` 只在 lr-pt，`sog_energy` 只在 LR）。
本文只覆盖 lr-pt 的 `latent_charge` 模型。

---

## 1. 环境

```bash
source /etc/profile.d/modules.sh
module load gcc/13                       # AOTI wrapper 用冻结时的 g++ 编译
export PYTHONPATH=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
export CUDA_HOME=/usr/local/cuda-12.4
PY=/data/home/jiangzhen/.conda/envs/lr212/bin/python   # torch 2.12.1+cu126
```

- **torch ≥ 2.12 是硬要求**：2.11 的 AOTInductor 在 mesh/FFT 图上死在
  `torch/_inductor/tiling_utils.py` 的 `get_pw_red_splits`（断言失败）。
  代码里有 `check_mesh_freeze_torch_version` 守卫会直接报错。**不要用 2.11 绕过守卫**。
- `lr212` env = clone `lr` + `pip install torch==2.12.1 --index-url .../cu126`（triton 3.7.1）。
- 冻结时 `deepmd/lib/libdeepmd_op_pt.so` 必须是用同一 torch 2.12 编的（见
  `sog_work/mdsuite/build_lrpt212_op.sh`），且 `deepmd/lib/run_config.ini` 的
  `PT_VERSIOn` 与运行时 torch 一致。

---

## 2. 生产命令

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$PY" -m deepmd --pt-expt freeze \
  -c /path/to/model.ckpt-<step>.pt \
  -o /path/to/model.pt2 \
  --compress
```

- `--compress` 触发 canonical/compact 下界（`lower_input_kind=dpa4c_canonical`，
  `graph_edge_dtype=float32`，`canonical_index_storage=byte_words`）。**多卡必需**
  （`source/api_cc/src/DeepPotPTExpt.cc` 对非 canonical 下界在多 rank 下直接抛错）。
- **不加 `--compress`** → 默认 graph 下界（`lower_input_kind=graph`，`float64` 边），
  **仅单卡**。用于对照/调试，不用于生产。
- `--compress` 只对 `--pt-expt` 有效（`deepmd/main.py` 的守卫）。
- `--lower-kind` 不必写；`--compress` 会以 `lower_kind="auto"` 重导出并选 `dpa4c_canonical`。

---

## 3. 输入与默认

- checkpoint：`state_dict["model"]["_extra_state"]["model_params"]`。
- 长程计算设置在 `model_params.long_range`（**不是权重**，可改）：
  `mesh_backend`（默认 `bspline`）、`pme_order`（默认 `None`→8）、
  `pme_accuracy`（默认 `None`→1e-4）、`pme_spacing`（默认 `None`→按精度算）、
  `k_grid_mode`、`k_grid_bounds`、`kspace_tolerance`。
- 训练的核参数（`log_width`、`amp`）在权重里，冻结原样带入。

要改 mesh backend / 阶数做对比，用
`sog_work/tools/freeze_lr_variants.py`（patch `model_params.long_range` 后调用 freeze），
**不改权重**。

---

## 4. 冻结前自检

```python
import torch
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
mp = sd["_extra_state"]["model_params"]
lr = mp["long_range"]
assert lr["type"] == "latent_charge" and lr["periodic"]
# 预测网格（Δ 与 σ_min、k_max 有关，见 GRID_FROM_ACCURACY.md §1.2）
```

预期网格（参考核 σ_min=2.0 Å、k_max=3.0349、盒 122.31 Å、1e-4，**算子取整后**）：

| backend | order | Δ (Å) | 部署网格 |
|---|---|---|---|
| bspline | 8 | 0.776378 | 160³ |
| bspline | 6 | 0.759286 | 162³ |
| midtown | 6 | 0.481222 | 256³ |

（`bspline` 走 `even_fft_size`（偶、2-3-5-7）；`midtown` 走 `next_factorable`（2-3-5）。
原始 `ceil` 是 158/162/255，**部署网格按算子取整后是 160/162/256**。）

---

## 5. 冻结后验证

```python
import json, zipfile
z = zipfile.ZipFile(out_pt2)
for n in z.namelist():
    if n.endswith("metadata.json"):
        d = json.loads(z.read(n))
        if "topologies" in d:
            assert "pme_mesh" in d["topologies"]
            print("spacing:", d["topology_params"]["pme_mesh"]["spacing"])
            print("lower :", d.get("lower_input_kind"), d.get("graph_edge_dtype"))
            break
```

canonical 生产件必须满足：
- `topologies` 含 `pme_mesh`，且 `topology_params.pme_mesh.spacing` == 预期 Δ；
- `lower_input_kind == "dpa4c_canonical"`；
- `graph_edge_dtype == "float32"`。

---

## 6. 冒烟

```lammps
pair_style   deepmd  /path/model.pt2
pair_coeff   * * O H
run 0
```
- 单卡：`lmp -in deck`；
- 多卡：deck 加 `processors 1 1 <np>`，`mpirun -np <np> lmp -in deck`；
  midtown 分布式网格只认 z-slab（`processors 1 1 np`），否则回退复制网格。
- 打开 `SOG_MIDTOWN_DEBUG=1` 可确认 midtown 网格被采用。

---

## 7. 环境变量与排错

| 现象 | 原因 / 处理 |
|---|---|
| `requires torch >= 2.12` / `get_pw_red_splits` 断言 | torch 太旧 → 用 `lr212`（2.12.1） |
| AOTI wrapper 加载时 `GLIBCXX_3.4.30` 缺失 / 段错误 | 冻结时未 `module load gcc/13` |
| `run on a single rank`（np>1） | 用了默认 graph 下界 → 加 `--compress` |
| `pme_mesh.spacing` 与预期不符 | `mesh_backend`/`pme_order` 未生效；确认是 `pt_expt`（native `pt` 会丢弃这两项） |
| op 加载 ABI 报错 | `libdeepmd_op_pt.so` 与运行时 torch 版本不一致 → 重编 op |

运行时改后端/阶数（不改权重）用 `sog_work/tools/freeze_lr_variants.py`；
网格/精度规则与常数表见 [`GRID_FROM_ACCURACY.md`](GRID_FROM_ACCURACY.md)。
