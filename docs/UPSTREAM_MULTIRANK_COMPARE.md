# 上游多卡策略 vs 本分支（2026-11-18）

> 结论先行：上游最近这轮（#6127）改的是 **compressed canonical 的 multi-block halo ABI**，
> 不是把 DPA4/SeZM 的图多卡策略重写。`doc/model/dpa4.md` 里那段“multi-rank depends
> on the model”来自 7 月的 #5884，**已经在我们的 base 里**；这次只改了一行
> `DP_COMPILE_INFER`。真正的差异在 canonical 这一层：上游用
> `BorderPlan{send_nodes, swaps}`，我们用 8 元组 `comm_dict`。
>
> 我们的生产路径（CubeS₂-6 + slab + compressed canonical）是 **单 block**，多卡走
> **slab 分布式网格**，不需要消息传递 halo。所以物理路径不冲突，冲突只在 ABI；
> 先把 canonical 多 block 让给上游的 halo，再把 mesh/ik/grid/slab 拆成独立 PR。

## 1. 当前状态

| | 值 |
|---|---|
| merge base | `0edeaef56`（2026-11-06） |
| 我们的 `main` | `fadb995dd`（2026-11-18，含 B4/QuadS 测试报告），**50 ahead / 14 behind** |
| 上游 tip | `511b5c836`（2026-11-18） |
| 关键上游提交 | `7477c9aea` **#6127** DPA4C multi-block + compressed deployment |
| 其他新提交 | #6138、#6136、#6129、#6056、#6052、#6046、#6041、#6030、#5980、#5956、#5834、#6051、#6040 |

上次（11-13）我们只落后 10 个，之后上游又进了 4 个：
`511b5c836`（LAMMPS Kokkos toggle）、`fcf301429`（virtual-atom neighbor stats）、
`5fac80e6c`（`is_inference` compile）、`175f6d18a`（actions/cache）。

## 2. 上游的两套多卡机制

上游现在同时存在两套，服务的层不同：

### 2.1 图 / edge_vec 路径：`comm_dict`

来自 #5884，和本分支的图多卡实现**同源**：8 个张量
（`send_list`, `send_proc`, `recv_proc`, `send_num`, `recv_num`, `communicator`,
`nlocal`, `nghost`），在 descriptor 的 `border_op` 里逐 block 交换 ghost 特征。
产物是 `model/extra/forward_lower_with_comm.pt2`，C++ 用
`build_comm_tensors_positional()` 装配。

`doc/model/dpa4.md` 的 multi-GPU 说明就是这一套：

> A multi-rank archive must embed a with-comm AOTInductor artifact
> (`model/extra/forward_lower_with_comm.pt2`) for the cross-rank exchange.
> … the graph route … runs through the `border_op` MPI path once per interaction
> block … a rank with zero atoms aborts collectively.

### 2.2 compressed canonical（#6127 新增）：`BorderPlan`

多 block DPA4C 的 fused canonical 不能调 Python 的 `border_op`，所以 #6127 在
CUDA 算子里另做了一套 **halo plan**：

- `deepmd/pt_expt/utils/canonical_graph.py::BorderPlan`：
  `send_nodes`（`(M,)` int32，device）+ `swaps`（`(nswap, 5)` int32，CPU）+
  `communicator`；`swaps` 每行是 `[send to, recv from, send count, recv count,
  first halo row]`。
- C++ `source/api_cc/src/commonPTExpt.h::HaloTensors` /
  `halo_tensors(lmp_list, fwd_map, nloc, device)`，只发 halo 节点，不发整张
  neighbor sendlist。
- `source/op/pt/border_exchange.cc` 在 fused op 内逐 block 交换，**pinned host
  staging**（`torch::empty(..., pinned_memory(true))`）。
- 能力查询从 `has_message_passing()` 改为
  `canonical_graph_requires_halo()`；canonical lower 的尾部参数是
  `halo_send` / `halo_swaps`（外加共享的 `communicator`）。
- 算子签名（`ops.cc`）尾部：
  `... fitting_routes=None, blocks=None, halo_send=None, halo_swaps=None,
  communicator=None, block_rcut=None`。

上游文档（#6127 改的 `doc/model/dpa4c.md` / `doc/model/long-range.md`）新增：

> A model with several message-passing blocks compresses on the same terms and
> runs compressed on one GPU or several …
> A compressed DPA4C model with several message-passing blocks requires such a
> module on a CUDA device, on one MPI rank or on several.

以及 `doc/install/install-lammps.md` 的 UCX 启动选项（`--mca pml ucx`,
`UCX_MAX_RNDV_RAILS=1`, `UCX_RNDV_SCHEME=put_zcopy`）。

## 3. 本分支的实现

| 路径 | 机制 | 多卡靠什么 |
|---|---|---|
| 生产：`dp --pt-expt freeze --compress` → `pair deepmd/kk` | 单 block DPA4C fused canonical + long-range op | **slab 分布式网格**（`SOG_MIDTOWN_SLAB=1`）+ `communicator`；不需要 halo |
| SeZM（DPA4）compact canonical（`SEZM_CANONICAL_STATUS.md`） | with-comm 兄弟入口，8 元组 `comm_dict` | `comm_dict` + `border_op` |
| 图 / edge_vec | 与上游 #5884 相同的 `comm_dict` | `border_op` |
| 能力查询 | `has_message_passing()`（C++ / C API） | 自建 |

关键点：**生产路径没有消息传递 halo**。DPA4C 是单 block，canonical 只需要
`communicator` 给长程算子做 slab 的 FFT/NCCL 交换；`comm_dict` 那条是给 SeZM
图描述符用的，和上游 #6127 的 `BorderPlan` 是两套东西。

### 3.1 与上游的直接差异

| | 上游 #6127 canonical | 本分支 canonical |
|---|---|---|
| plan | `send_nodes` + `swaps`（2 张量） | 8 张量 `send_list/send_proc/...` |
| 交换位置 | fused CUDA op 内（`border_exchange`） | Python `border_op`（SeZM with-comm） |
| 数据量 | 只发 halo 节点 | 整张 neighbor sendlist |
| 能力查询 | `canonical_graph_requires_halo()` | `has_message_passing()` |
| 覆盖 | 多 block DPA4C | 单 block DPA4C / SeZM |
| 零原子 rank | collective abort | 空 subdomain 早退，可能 hang（已知限制） |

## 4. 合并冲突与解法

`git merge upstream/master` 仍是 **8 个文件、16 个 hunk**（新提交没有增加文件）：

| 文件 | 解法 |
|---|---|
| `deepmd/pt_expt/kernels/dpa4c/canonical_long_range.py` | 取上游的 `border` 参数与尾部 `blocks/halo_send/halo_swaps/block_rcut`，把我们的 `mesh_backend/force_mode` **接在 `block_rcut` 之后**；单 block 时 `border=None` |
| `deepmd/pt_expt/model/ener_model.py` | canonical 签名改用 `halo_send/halo_swaps`；要保留 SeZM with-comm 就再加一路分派，否则删掉 |
| `deepmd/pt_expt/utils/serialization.py` | 合并 `nframes`（我们）与 `want_border/border_sizes`（上游）；资格判断用上游 `dpa4c_canonical_unsupported` + 我们的 SeZM |
| `source/api_cc/src/DeepPotPTExpt.cc` | canonical 走 `halo_tensors()`；图/SeZM with-comm 保留 `comm_tensors()` |
| `source/api_cc/src/DeepPot.cc` | 保留上游的 `canonical_graph_requires_halo()`，删我们的 `has_message_passing()` 重复 API |
| `source/lmp/pair_deepmd_kokkos.cpp` | 传 halo plan |
| `source/op/pt/dpa4c/graph_compress.cu` | 算子尾部同时接受 `blocks/halo/block_rcut` 与 `mesh_backend/force_mode` |
| `source/op/pt/dpa4c/ops.cc` | 同上，schema 尾部追加两个 int（带默认值） |

## 5. PR 应该怎么提

### 第 0 步：同步（不是 PR）

在 `sync/upstream` 分支解 8 个冲突，**先不 push 到 main**。同步后的验收：

```bash
python -m pytest source/tests/common/dpmodel/test_grid_rule.py \
                 source/tests/common/dpmodel/test_long_range.py -q
python -m pytest source/tests/pt_expt/kernels/test_mesh_backend.py \
                 source/tests/pt_expt/kernels/test_long_range_sum.py -q
# 冻结 → run 0：checkpoint vs .pt2
# 2/4 卡 slab：SOG_MIDTOWN_SLAB=1，能量/力与 np=1 对齐
```

### 第 1 步：拆 PR

| PR | 范围 | 规模 | 依赖 |
|---|---|---|---|
| **1** | 算子报错可读化 + ik×distributed 回落 + 网格显存预检查 | 4 文件 | 无 |
| **2** | CubeS₂/QuadS mesh + `force_mode=ik` + 能量律网格（**核心**） | ~12 文件 | #6116/#6127 |
| **3** | z-slab 分布式网格（任意进程网格、NCCL、任意 rank） | CUDA 大块 | PR 2 |
| **4** | slab device 常驻 + 6 阶 packed brick + B4 fused J（性能） | CUDA | PR 3 |
| **5** | LAMMPS `deepmd/kk` + `kspace sog/kk` 链 | 中 | 仅当还需要旧链 |

**建议先提 PR 1 + PR 2**。PR 2 是纯增量：只改 `grid_rule.py`、`midtown_*`、
`canonical_long_range.py`、`ops.cc`/`graph_compress.cu` 的尾部参数和测试，
不碰上游的 halo/border 逻辑，冲突面最小。

### 第 2 步：哪些先不要进 PR

- **`deepmd/pt/long_range/*` 的 pt backend 移植**：当前生产是
  `dp --pt-expt train/freeze`，pt 后端没有进入生产链。除非维护者明确要，
  否则这 30+ 文件会把 PR 撑得很大。
- **SeZM compact canonical + with-comm**：与上游 #6127 的 `BorderPlan` 直接
  重复。生产不用它；先留在本地分支，等上游问起再单独提。
- **`doc/SOG_BUILD_FREEZE_MD.md`、`doc/SOG_EFFECTIVE_OPTIMIZATIONS.md`、
  `doc/QUADS_STENCIL.md`、`doc/TEST_REPORT_20261118.md`**：内部 runbook/报告，
  不属于上游 `doc/` 的 toctree。
  移到 `docs-notes`，PR 里只保留对 `doc/model/long-range.md` 的用户向补充。
- **`kspace sog/kk` 旧链**：新 runbook 已用 compressed canonical 替代。

### 第 3 步：PR 2 的正文骨架

```markdown
feat(long-range): CubeS2/QuadS mesh backends, Fourier force mode and a
measured energy-law grid rule

- Add `mesh_backend ∈ {bspline, midtown, quads}` and `force_mode ∈ {ad, ik}` to
  the latent-charge module. `midtown` is the CubeS2 family, `quads` the QuadS
  family; both use a fixed 4/6-node stencil and share the same long-range
  operator tail.
- Size the mesh from the measured energy law `eps_E = C (Delta/sigma_min)^p`
  under the k-space normalisation `|dE|/|E_k|`; `pme_accuracy` becomes an
  energy-relative target. AD keeps its one-order-lower force exponent, so a
  force-relative guarantee is documented as `FORCE_RULE`.
- Extend the fused DPA4C canonical op with the two new integer arguments after
  the multi-block halo arguments of #6127, defaults preserving old graphs.
- Tests: grid-rule table (CPU), mesh backend parity vs replicated (GPU),
  freeze/run-0 parity on a 192-atom water cell.
```

## 6. 待确认

1. 训练到底走 `dp --pt` 还是 `dp --pt-expt`？前者才需要 pt backend 移植。
2. SeZM compact canonical 还在生产里吗？如果不在，PR 里直接删，冲突少一半。
3. `kspace sog/kk` 旧链还有用户吗？没有就不提。
4. 三份 `doc/SOG_*.md` 是否同意移到 `docs-notes`。
