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
  - 旧 PR/c560 实现 vs P0 移植：**max|ΔF| = 0，rms = 0**
- c560 测试：
  - `test_mesh_backend.py` + `test_long_range_sum.py`：**54 passed**
  - `test_long_range_multi_rank.py`：2 skipped（本机跳过）
- 2/4 rank slab run0 均正常。

## 重跑样条对比（1.12M，6 阶，slab）

| np | B-spline-6 | CubeS₂-6 | 结论 |
|---:|---:|---:|---|
| 2 | 74.26 ms | 80.82 ms | CubeS₂ 仍慢 8.8% |
| 4 | 49.68 ms | 44.11 ms | CubeS₂ 快 11.2% |

P0 前 c560 np2 CubeS₂ 为 111.9 ms，P0 后 80.8 ms（**−28%**）；B-spline 不变。

## 备注

- main 上同工件为 B-spline 75.5 / CubeS₂ 67.4（np2）；c560/PR 的 P0 移植
  np2 仍偏慢，np4 已恢复 CubeS₂ 优于 B-spline。剩余差异需要检查 c560 的
  grid/freeze 工件与 slab 分块，而不是继续改 P0 kernel。
- 本次移植用的是 P0/full long-range 文件级替换；c560 的 with-comm 路径已在
  2/4 rank slab run0 验证，但多 rank 单测在本机 skipped，未做完整 MPI 回归。
