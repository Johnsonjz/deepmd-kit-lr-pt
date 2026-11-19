# P0 port to pr/long-range-complete（2026-11-19）

## 做了什么

把 main/checkpoint 的 P0 long-range 核心（fixed-point float brick、6 阶 packed
spread、6 阶 term gather、device brick 路径）移植到 c560/PR 树：

- `source/op/pt/long_range/{charge_sum.h,domain_exchange.cc,latent_charge_sum.cu,mesh_sum.cu,ops.cc}`
- `source/op/pt/long_range/midtown/{sog_gpu.cu,sog_gpu.cuh,sog_green.h}`
- `source/op/pt/long_range/midtown_sum.cu`

commit：`254611ad`，已 push：

```
3f7db81a..254611ad  c560-test -> pr/long-range-complete
```

## 正确性

- 同一 c560 mid6 工件，1.12M `run 0`：
  - 旧 PR/c560 实现 vs P0 移植：**max|ΔF| = 2.78e-4，rms = 8.47e-5**
  - 相对 |F|max：**8.1e-5**；这是 P0 fixed-point/float brick 与旧 double
    路径的精度差，不是 P0 逻辑错误。
- c560 测试：
  - `test_mesh_backend.py` + `test_long_range_sum.py`：**54 passed**
  - `test_long_range_multi_rank.py`：2 skipped（本机跳过）
- 2/4 rank slab run0 均正常。

## 重跑样条对比（1.12M，6 阶，slab）

| np | B-spline-6 | CubeS₂-6 | 结论 |
|---:|---:|---:|---|
| 2 | 72.21 ms | **67.28 ms** | CubeS₂ 快 6.8% |
| 4 | 39.38 ms | **36.81 ms** | CubeS₂ 快 6.5% |

P0 前 c560 np2 CubeS₂ 为 111.9 ms；真正重编 P0 后 **67.20 ms**（−40%）。
np2/np4 都已回到 main 同量级（main np2 67.4、np4 36.3）。

## 备注

- main 上同工件为 B-spline 75.5 / CubeS₂ 67.4（np2）；c560/PR 的 P0 移植
  np2 仍偏慢，np4 已恢复 CubeS₂ 优于 B-spline。剩余差异需要检查 c560 的
  grid/freeze 工件与 slab 分块，而不是继续改 P0 kernel。
- 本次移植用的是 P0/full long-range 文件级替换；c560 的 with-comm 路径已在
  2/4 rank slab run0 验证，但多 rank 单测在本机 skipped，未做完整 MPI 回归。
- 之前一次 80.8/81.1 ms 的数字是**旧 .so 未重编**造成的；本次 `strings`
  确认 `k_spread_cubes2_brick_packed6` 已在二进制中，并强制重新编译。


## 最终回归（2026-11-19）

### PR CI 等价本地回归

在 c560/PR 树上、`CXX=gcc`、单卡 CUDA 运行：

```
235 passed, 8 skipped, 98 warnings in 732.03 s
```

覆盖：

- `source/tests/pt_expt/model/test_long_range.py`
- `source/tests/pt_expt/model/test_dpa4c_graph_lower.py`
- `source/tests/pt_expt/kernels/test_mesh_backend.py`
- `source/tests/pt_expt/kernels/test_long_range_sum.py`
- `source/tests/pt_expt/utils/test_canonical_graph.py`

另外 with-comm export：

```
15 passed in 450.95 s
```

（`test_export_with_comm` + `test_graph_export_with_comm` +
`test_spin_export_with_comm`。）

### with-comm MPI 回归

2 rank 和 4 rank：

```
mpirun -np 2 pytest test_long_range_multi_rank.py
# 2 passed

mpirun -np 4 pytest test_long_range_multi_rank.py
# 每 rank 2 passed（4 rank 全部通过）
```

修复了 P0 port 丢失的 **midtown frame-virial 分摊**：

- replicated 路径：按 owner 节点求和 per-node share；
- local/slab 路径：本 rank 的 atom_virial 求和；
- 2/4 rank 的 per-rank shares 之和等于单 domain 的 frame virial。

### PME 测试修复

c560 上 order-8 B-spline PME 网格偏粗导致 5 个 model 测试失败；
已在 `mesh_spacing()` 对 `bspline` 且 order≥8 加 Nyquist cap，与 main 一致；
相关 PME 测试 7 passed。
