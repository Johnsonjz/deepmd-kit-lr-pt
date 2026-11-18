# B4 / QuadS 补测报告（2026-11-18）

环境：`CXX=gcc`、`CC=gcc`、GPU 空闲；`main` = `f3d4b77e4`。

## 默认路径（B4 off、QuadS stencil off）

```
305 passed, 8 skipped in 528.80 s
```

覆盖：

- `source/tests/common/dpmodel/test_long_range.py`
- `source/tests/common/dpmodel/test_grid_rule.py`
- `source/tests/pt_expt/kernels/test_mesh_backend.py`
- `source/tests/pt_expt/kernels/test_long_range_sum.py`
- `source/tests/pt_expt/model/test_long_range.py`
- `source/tests/pt_expt/test_long_range_kernel_schedule.py`
- `source/tests/pt_expt/utils/test_canonical_graph.py`

## B4（`SOG_CANON_FUSE_J=1`）

| 项 | 结果 |
|---|---|
| `test_dpa4c_graph_lower.py` | **31 passed** |
| 1.12M run0 力 vs 默认 J | **max=0, rms=0** |
| 1.12M 6th slab np2 | 默认 65.64 ms → B4 **64.73 ms（−1.4%）** |

**已知限制**：B4 改变 fitting-route 的逐位累加顺序，
`test_every_network_route_returns_the_bits_of_the_rules` 会失败；因此 B4
默认关闭，生产显式 `SOG_CANON_FUSE_J=1`。兼容开关 `SOG_CANON_NO_B4=1`。

## QuadS stencil（`SOG_QUADS_STENCIL=1`）

| 项 | 结果 |
|---|---|
| `test_mesh_backend` + `test_long_range_sum` | **70 passed** |
| 1.12M QuadS-6 run0 力 vs 旧 QNodeS | **max=1.0e-5, rms=5.2e-7** |
| 1.12M 1 GPU | 0.8524/0.8531 s → 0.8486/0.8489 s（约 **0.4%**） |

结论：stencil 正确，但无性能量级收益；默认关闭。

## 未合入

- `charge-zero-init`：初始化时把 charge 输出层置零（不是改 `bias_atom_q`），
  保留在本地分支 `charge-zero-init`，未合入 main。
