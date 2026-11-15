# SOG 有效优化与合并说明

日期：2026-11-15  
主分支：`main`（合并提交 `944d09e52`）  
实验分支：`charge-net-exp-20261115`（保留完整实验/负结果）  
基线：`1f9251fac`（checkpoint，305 passed）

本文只记录**已合并到 `main` 的有效项**；未合并的 A1/计时/负结果在实验分支和
`/data/home/public/jiangzhen/dp/sog_work/experiments/charge_net_20261115/REPORT.md`。

## 1. 合并内容

主分支 `944d09e52` 包含：

1. **slab AD 路径 device 常驻**（P0a）
   - `midtown_sum.cu`：slab 的力/势不再 D2H/H2D；`sog_gpu_gather_brick_dev_ptr` 等
     直接写 device 张量，去掉 host force/vpot 与 `.to(device).clone()`。
2. **6 阶 packed brick spread**（P0b）
   - `midtown/sog_gpu.cu`：`k_spread_cubes2_brick_packed6`，6 阶 CubeS2 走
     packed 6th brick spread。
3. **6 阶 brick term gather**（P0c）
   - `k_gather_cubes2_term_brick6`、`sog_gpu_spread/gather_brick_dev_ptr`。
4. **B4：J 响应融进 energy fitting epilogue**
   - `graph_fitting.cu`：fused kernel 写 descriptor cotangent 时可选加 `J*v`
     （`add_jacobian`/`add_v`）。
   - `graph_compress.cu`：`charge_jacobian`/`charge_gradient` 按 tile 偏移传入；
     per-layer charge 的 `backward()` 先于 energy fitting；删除独立
     `descriptor.addcmul_(J, v)`。
   - `graph_ops.h`：`fitting_energy_gradient_range` 增加可选参数。
   - **力与旧 J 路径逐位一致（max=0，rms=0）**。
5. **可选 VJP 显存模式**（默认关）
   - `SOG_CANON_VJP=1`：tile VJP，不再物化 N×208 J；1.12M np2
     +4.2% 时间、−468 MiB。默认仍走 J 路径。
6. **正确性修复**
   - `LatentChargeFitting`：反序列化后 `bias_atom_e` 仍遵循输出层
     `trainable=False`；fix `test_parameter_gradients_and_freezing`。
   - PME model：B-spline order≥8 在 `mesh_spacing()` 加
     `0.75*pi/k_max` Nyquist cap，修复 PME 与直接 k-space 参考、
     帧批处理、frozen SOG 的一致性；LAMMPS midtown 路径不受影响。
   - 相关测试更新在 `source/tests/common/dpmodel/test_long_range.py`。

## 2. 实测收益（1.12M，6 阶，slab）

| 阶段 | np2 | np4 |
|---|---:|---:|
| P0a/b/c 前 | 74.52 ms | 41.84 ms |
| P0a/b/c 后 | 65.80 ms | 36.85 ms |
| **B4 后（当前 main）** | **64.64 ms** | **36.28 ms** |
| dpa4c 同参数 | 51.57 ms | 28.51 ms |

- B4：np2 **−1.50 ms（−2.3%）**，np4 **−0.45 ms（−1.2%）**。
- SOG 相对 dpa4c：np2 +25.3%，np4 +27.3%。
- 2M 6 阶显存：**13,890 / 8,770 / 5,554 MiB**（1/2/4 卡），
  0.227 / 0.334 / 0.490 ns/day；B4 不改变峰值。

### 98k / 331k（AD）

| 配置 | 98k LR / mem | 331k LR / mem |
|---|---:|---:|
| CubeS2-4 | 1.12 / +334 | 4.80 / +1194 |
| CubeS2-6 | 1.18 / +310 | 4.86 / +1102 |
| B-spline-6 | 1.27 / +326 | 5.32 / +1152 |
| QuadS-6 | 1.19 / +310 | 5.06 / +1086 |

SR 98k 7.96 ms / 1402 MiB；331k 26.18 ms / 2044 MiB。LR 只占 12–16%。

## 3. 分阶段卡点（B4 后，1.12M 6 阶）

| phase | np2 | np4 |
|---|---:|---:|
| step1 descriptor/state | 14.3–14.7 | 6.65–6.82 |
| step2 charge fwd + J | 2.74 | 1.53 |
| step3 LR | 5.7–6.4 | 3.5–3.65 |
| step4 energy fitting + J | 19.4–19.7 | 11.1–11.3 |
| step5 descriptor backward/force | 15.2–15.8 | 7.6–7.7 |
| step6 epilogue | 0.15 | 0.08 |
| op total | 58.2–59.0 | 30.6–31.0 |
| LAMMPS step | 64.64 | 36.28 |

- **LR = mesh + FFT**：mesh 65%（spread 1.07 + gather 2.27），
  FFT/all-to-all 约 32%，halo/reduce 4%。
- **SOG 专属增量** np2 13.07 ms：LR 5.8（44%）、charge 2.7（21%）、
  图/workspace 约 4.6（35%）；np4 7.77 ms：LR 3.5、charge 1.5、
  图/workspace 2.8。
- **整步最大项**是 energy fitting（19.5/11.1）和 descriptor/edge
  （step1+5，29.4/14.4），两者 dpa4c 共用，不是 SOG 专属。
- graph/workspace 仍有 **1–2 ms** 空间：J fp16、预分配/复用
  charge/J buffer、减少 dtype 转换、AOTI workspace 规划；不是 4.6 ms 全部。

## 4. 未合并（实验分支保留）

- **A1 shared-weight**：20k 精度不损，但仍两个前向，1.12M np2
  68.65 vs 68.90 ms，无性能收益。
- **A1 shared-trunk op**：`SOG_CANON_A1=1` 原型，力 max 1e-5，
  但 1.12M np2 **85.5 ms**（比 68.65 慢）；未合并。
- **`SOG_CANON_TIME`**：分阶段计时仪器，未合并生产。
- **fp16 J / 线性/低秩 charge head**：只做了分析/微基准，未实现。

## 5. 运行方式

```bash
# 构建
make -C build-lmp-mpi-212 deepmd_op_pt -j 20
cp build-lmp-mpi-212/op/pt/libdeepmd_op_pt.so \
   /data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib/

# LAMMPS slab（生产）
SOG_MIDTOWN_SLAB=1 <mpirun> lmp -k on g 1 -in in.slab1m.1119744.np2.lammps

# 可选：VJP 显存模式
SOG_MIDTOWN_SLAB=1 SOG_CANON_VJP=1 <mpirun> lmp ...
```

- 生产工件：`dpa4c-sog-mid6ad.pt2`（6 阶）。
- 测试：
  - `source/tests/pt_expt/model/test_dpa4c_graph_lower.py`
  - `source/tests/pt_expt/kernels/test_mesh_backend.py`
  - `source/tests/pt_expt/kernels/test_long_range_sum.py`
  - 合并前 **101 passed**；完整相关 suite 在 checkpoint 上 **305 passed / 8 skipped**。

## 6. 代码位置

- `source/op/pt/long_range/midtown_sum.cu` / `midtown/sog_gpu.cu` / `sog_gpu.cuh`
  — slab device、6 阶 packed spread/gather。
- `source/op/pt/dpa4c/graph_compress.cu` — B4、optional VJP。
- `source/op/pt/graph_fitting.cu` / `graph_ops.h` — fused J add。
- `deepmd/pt_expt/fitting/latent_charge.py` — freeze bias。
- `deepmd/dpmodel/long_range/latent_charge.py` — PME order-8 cap。
- `source/tests/common/dpmodel/test_long_range.py` — PME 测试。

## 7. 结论

- 已合并的有效项：slab device、6 阶 packed spread/gather、B4 fused J add、
  PME/freeze 修复、可选 VJP 显存模式。
- 6 阶 + slab 仍是生产设置；1.12M np2 64.64 ms，2M 13,890 MiB（1 卡）。
- 后续优化优先级：energy fitting/descriptor（整步大头），LR gather，
  charge/J（1–2 ms），graph/workspace（1–2 ms）；A1 当前负结果，不合并。
