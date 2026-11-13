# 与上游 `deepmd-kit` 同步：合并冲突地图

> 状态：2026-11-13。`main` 相对 `upstream/master` 是 **42 ahead / 10 behind**，
> merge base 为 `0edeaef56`。直接 `git merge upstream/master` 会在 8 个文件、
> 16 个 hunk 上冲突。本文记录冲突位置、根因和建议的解法，供真正做同步时对照。

## 1. 复现

```bash
git worktree add -b sync/upstream ../deepmd-kit-lr-pt-sync main
cd ../deepmd-kit-lr-pt-sync
git merge upstream/master
git diff --name-only --diff-filter=U
```

试合并只用于看冲突，不要提交；看完 `git merge --abort` 并
`git worktree remove ../deepmd-kit-lr-pt-sync --force`。

## 2. 落后的 10 个上游提交

```
ba95134a1 fix(dpmodel): avoid deprecated torch size check (#6136)
b0d8ac860 fix(pt_expt): reuse the stored min_nbor_dist and batch the neighbor statistics (#5956)
3b2555c46 perf(train): batch bound EMA updates with foreach (#6129)
7477c9aea feat(dpa4c): add multi-block attention and compressed deployment (#6127)   <- 主冲突源
5485b3abc ci: cache generated C++ model fixtures (#6030)
4e3264d44 fix(build): honor disabled TensorFlow for native interfaces (#6051)
f4c007382 docs(pt): sync DPA4 compile compatibility docs (#6046)
501c43cb6 fix(pt_expt): preserve numb_dos for exported DOS models (#6041)
32759294f fix(pt): omit absent charge-spin inputs for legacy scripted models (#6052)
5d4ae8072 Rewrite of int->int function (#6040)
```

主冲突来自 **#6127**：上游把 compact canonical 下界的跨 rank 通信改成了
**border/halo 计划**（`halo_send` / `halo_swaps`、`ptexpt::HaloTensors`、
`descriptor.block_rcut`、`compress_blocks`），而本分支在同一层用的是
**`comm_dict` + `build_comm_tensors_positional*` + with-comm 兄弟入口**。
两边都改了 `forward_lower_canonical_graph` 的签名和算子实参列表，所以冲突
集中在这一条 ABI 上，而不是长程物理本身。

## 3. 冲突清单

| 文件 | hunk | 上游改动 | 本分支改动 |
|---|---:|---|---|
| `deepmd/pt_expt/model/ener_model.py` | 1 | 尾部参数 `halo_send` / `halo_swaps` | 尾部参数 `comm_dict` |
| `deepmd/pt_expt/kernels/dpa4c/canonical_long_range.py` | 1 | 新增 `compress_blocks`、border 计划、`block_rcut[1:]` | 新增 `mesh_backend` / `force_mode` 实参 |
| `deepmd/pt_expt/utils/serialization.py` | 4 | 合成输入新增 `want_border` / `border_sizes`；canonical 资格新增 `dpa4c_canonical_unsupported` | 合成输入新增 `nframes`；SeZM 资格从 `dpa4` 取 |
| `source/api_cc/src/DeepPotPTExpt.cc` | 4 | 调 `ptexpt::halo_tensors()`，canonical lower 传 `halo` | 调 `build_comm_tensors_positional[_with_virtual_atoms]()`，走 `run_model_canonical_graph_with_comm` |
| `source/api_cc/src/DeepPot.cc` | 1 | 输出/参数校验的上下文 | 输出 dump 与 `has_message_passing` 分支 |
| `source/lmp/pair_deepmd_kokkos.cpp` | 2 | halo/border 计划 | comm payload / 输出 dump |
| `source/op/pt/dpa4c/graph_compress.cu` | 2 | 算子签名加 border/block 参数 | 算子签名加 long-range 参数 |
| `source/op/pt/dpa4c/ops.cc` | 1 | 注册/校验新参数 | 注册/校验长程参数 |

## 4. 建议的解法（需要设计确认，不要盲合）

1. **消息传递用上游的 halo 计划，长程用本分支的 comm**：两者语义不同——halo
   刷新的是多 block 描述符的 ghost 特征，长程 comm 传的是电荷/网格跨 rank 求和。
   合并后的 `forward_lower_canonical_graph` 应同时接受 `halo_send/halo_swaps`
   和 `comm_dict`（或把长程 payload 也编码进同一个 border 计划，但那是 ABI 变更，
   需要重新生成 `.pt2` fixture）。
2. **算子实参顺序统一**：`canonical_long_range` 的调用点要同时带
   `compress_blocks/border/block_rcut` 与 `mesh_backend/force_mode`。建议把新参数
   全部放在尾部并给默认值，避免旧图失效。
3. **资格判断合并**：保留上游的 `dpa4c_canonical_unsupported` 拒绝原因，再把
   SeZM 的 `sezm_canonical_eligible` 并进去；顺序是 unsupported → dpa4c → sezm → dpa1。
4. **ABI 版本**：如果最终改了 canonical lower 的输入顺序/数量，必须同步更新
   `source/api_c/tests`、`source/api_cc/tests` 与 `source/tests/infer` 的 fixture，
   并在 PR 里单独说明。

## 5. 合并后的验证

```bash
# 纯 Python（无 GPU）
python -m pytest source/tests/common/dpmodel/test_grid_rule.py \
                 source/tests/common/dpmodel/test_long_range.py -q

# 单卡算子
python -m pytest source/tests/pt_expt/kernels/test_mesh_backend.py -q
python -m pytest source/tests/pt/model/test_sezm_long_range.py -q
python -m pytest source/tests/pt_expt/test_long_range_kernel_schedule.py -q

# 编译五个目标后，跑 examples/lammps/deepmd_kk_sog 的 run 0
# 以及 host 链 / Kokkos 链的 pe 对齐（np=1 与 np=2）
```

## 6. 在合并之前就能做的 PR 材料

- [PR_SUBMISSION.md](PR_SUBMISSION.md)：A（bugfix）/ B（网格后端）/ C（文档）/
  D（分布式网格）的拆分、编译命令、验收断言、英文正文。
- [LONG_RANGE_PORT.md](LONG_RANGE_PORT.md)：上游 `long_range` → pt backend 的
  接口映射与有意偏差，可直接作为设计说明的骨架。
- [BUILD_AND_TEST.md](BUILD_AND_TEST.md) 与
  [MESH_ENGINE_TEST_REPRO.md](MESH_ENGINE_TEST_REPRO.md)：复现与测试矩阵。
