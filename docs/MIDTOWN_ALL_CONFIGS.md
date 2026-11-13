# CubeS₂ / QuadS 全配置复核：4/6 阶 × AD/ik，98k 与 331k

> 2026-11-13，RTX 4090（GPU 3 空闲），A 链 `dpa4c-sog` 核，等精度
> `pme_accuracy = 1e-4`，replicated，`fix nve`，0.5 fs，50 步取第二段
> `Loop time`。数据：`sog_work/bench/matrix_midtown/backend_10_{98304,331776}.csv`
>（由 `all8_*.csv` 复制而来；`backend_9_*` 是 pre-fix，勿用）。
> 构建：`source/op/pt/long_range/` 的 stream/capture 修复（见
> [`MIDTOWN_PERF_REDESIGN.md`](MIDTOWN_PERF_REDESIGN.md) §0.1）。
> 论文：`papers/sog-lib-paper/sections/06-results.tex` 的 Table 6/7 与
> Fig. 9 已按本表更新。

## 1. 等精度主表

| backend | order | mode | 98k grid | step (ms) | LR (ms) | LR mem (MiB) | 331k LR (ms) | 331k mem (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| B-spline | 4 | AD | 216³ | 9.74 | 1.66 | 488 | 7.82 | 1294 |
| B-spline | 6 | AD | 108³ | 9.39 | 1.30 | 324 | 5.37 | 1156 |
| CubeS₂ | 4 | AD | 125³ | 9.27 | 1.18 | 334 | **4.79** | 1200 |
| CubeS₂ | 6 | AD | 96³ | 9.31 | 1.23 | **310** | 5.00 | 1102 |
| CubeS₂ | 4 | ik | 125³ | 9.35 | 1.26 | 366 | 5.12 | 1312 |
| CubeS₂ | 6 | ik | 96³ | 9.79 | 1.70 | 326 | 6.48 | 1150 |
| QuadS | 4 | AD | 125³ | 9.24 | **1.16** | 342 | 4.87 | 1222 |
| QuadS | 6 | AD | 81³ | 9.31 | 1.23 | **310** | 5.09 | **1086** |
| QuadS | 4 | ik | 125³ | 9.32 | 1.23 | 374 | 5.18 | 1334 |
| QuadS | 6 | ik | 81³ | 9.35 | 1.27 | 324 | 5.23 | 1124 |

> 论文 AD-only 表用的是 **3 次中位数**（CubeS₂-6 1.207/4.933、
> CubeS₂-4 1.162/4.841 等，见 §5），与下表单次值差 ~0.02 ms；
> 下表保留 10 配置全貌，ik 行仅作历史记录。

SR baseline：98k 8.09 ms / 1402 MiB；331k 26.58 ms / 2044 MiB。
`LR = step − SR`，`LR mem = peak − SR peak`，均为 0.1 s 整卡采样；
GPU 3 空闲，重复运行 ±2 MiB / ±1%。

## 2. 逐配置结论

### B-spline 4/6（基准）
- 4 阶：216³/320³，1.66/7.82 ms，+488/+1294 MiB。网格最大、最慢、最费显存；
  4 阶 B-spline 是这次矩阵里唯一"明显不值得"的配置。
- 6 阶：108³/160³，1.30/5.37 ms，+324/+1156 MiB。仍是 6 阶的可靠基准，
  但比 CubeS₂-6/QuadS-6 慢约 5%、多约 5% 显存。

### CubeS₂-4 AD（4 阶首选之一）
- 125³/192³，1.18/4.79 ms，+334/+1200 MiB。
- 等精度下比 B-spline-4 少 5.2× 格点，LR **−29%/−39%**，显存 **−32%/−7%**。
- 98k 比 QuadS-4 AD 慢 0.02 ms（1.18 vs 1.16），331k 反超 0.08 ms（4.79 vs 4.87）；
  两者在 2% 内，4 阶可以按显存/实现偏好二选一。
- 4 阶 32 节点、6 阶 88 节点的 ball-supported stencil，无独立 stencil pass；
  4 阶实现是这次矩阵里最稳的。

### CubeS₂-6 AD（6 阶首选之一）
- 96³/144³，1.23/5.00 ms，+310/+1102 MiB。
- 比 B-spline-6 快 5%、省 4–5% 显存；88 节点 vs 216 节点，但每节点成本高，
  净收益只有 5%——这就是"节点数优势被每节点成本吃掉"后的真实剩余。
- 与 QuadS-6 AD 基本并列（1.23/1.23；331k 5.00/5.09）。

### CubeS₂-4 ik
- 125³/192³，1.26/5.12 ms，+366/+1312 MiB。
- 比 AD 慢 7%/7%，但换到"力的阶数"；在 4 阶、需要 ik 力时可用。
- 显存比 AD 多 32/112 MiB（4 阶网格大，3 张梯度网格的代价被放大）。

### CubeS₂-6 ik（唯一明确不推荐）
- 96³/144³，1.70/6.48 ms，+326/+1150 MiB。
- 比同网格 AD 慢 0.47/1.48 ms；108³ 相位表里 gather 0.620 ms，是 AD 0.189 的 3.3×。
- 原因：value-only gather 仍要读 4 个独立网格（pot/gx/gy/gz），88 节点 × 4 路；
  不是"多 3 次逆变换"本身（fft_inv 0.142 vs 0.045，只多 0.10 ms）。
- 独立微基准（96³/98k）：baseline 4 路读 0.216 ms、去掉表填 0.199、
  只读 pot 0.062、float4 单读 0.107；即"4 路读"值 ~0.15 ms，
  但 op 内同一 kernel 是 0.62 ms（profiler 0.625）。**op 上下文里还有
  ~0.4 ms 的、ncu 无法归因的开销**（本机 `ncu` 报 `ERR_NVGPUCTRPERM`）；
  所以本轮没有把它压到 AD 水平。工程结论不变：6 阶用 QuadS-ik/AD。
- **结论：6 阶 ik 用 QuadS-6 ik，退役 CubeS₂-6 ik。**
- **重要口径**：本 campaign 的能量规则给 ik 和 AD **同一个网格**
  （6 阶 96³；4 阶 125³）。**ik 不买网格，只买力的阶数**；在 6 阶它没有
  网格收益且慢 0.47 ms，所以退役；4 阶若需要 ik 力，选 QuadS-4 或 CubeS₂-4。
  “ik 的网格比 AD 粗”是**力口径**的结论，不是本表能量口径的结论。

### QuadS-4 AD（4 阶首选之一）
- 125³/192³，1.16/4.87 ms，+342/+1222 MiB。
- 98k 最快（1.16），331k 第二（4.87）；与 CubeS₂-4 AD 在 2% 内。
- 表在 spread/gather 各填一次（216 B/atom，shared），所以 spread 0.213 ms
  比 B-spline 的 0.199 还高；但 gather 0.127 ms 只有 CubeS₂ 的 2/3，
  总 mesh 0.414 ms，比 B-spline 低 9%。

### QuadS-6 AD
- 81³/125³，1.23/5.09 ms，+310/1086 MiB（331k 显存最低）。
- 与 CubeS₂-6 AD 并列；6 阶网格 81³ 比 CubeS₂ 的 96³ 小 40%，
  但 per-atom 表填把时间拉回同一水平。
- 若显存优先，331k 选 QuadS-6 AD（1086 MiB）。

### QuadS-4 ik
- 125³/192³，1.23/5.18 ms，+374/+1334 MiB。
- 比 AD 慢 6%/6%，与 CubeS₂-4 ik 基本并列；4 阶 ik 可用。

### QuadS-6 ik（6 阶 ik 首选）
- 81³/125³，1.27/5.23 ms，+324/+1124 MiB。
- 只比 AD 慢 0.04/0.14 ms；108³ gather 0.166 ms（AD 0.127，CubeS₂-ik 0.620）。
- **若必须用 ik，选 QuadS-6 ik。**

## 3. 逐配置优化：做了什么、没做什么

### 已落地（全配置共享）
- **CUDA stream + graph capture 修复**：所有 mesh kernel/memset/cuFFT/event 走
  `at::cuda::getCurrentCUDAStream()`；CUDA 系数改 pinned host mirror。
  这是本轮唯一"全局"优化，直接把 pre-fix 的 +724/+2298 MiB 降到 +310/+1102，
  并让 AOTI graph 真正捕获 mesh。没有它，下面所有 per-config 数字都不成立。

### 试过并回退（负结果）
- **ik 的 pot/gx/gy/gz float4 交织**：pack 成 `float4`（16 B/pt），gather 从
  4 路 4 B 读变 1 路 16 B 读。结果：mid6ik 98k 1.70→1.74 ms，显存 +14 MiB；
  qs6ik 1.27→1.28。**没有收益，回退。**
  解释：pack 自身要一次 16 B/pt 的全网格读写，而 ik gather 不是
  "transaction 数"受限——4 张网格的 footprint/表填才是主因；88 节点的
  value-only gather 已经能把这 4 路读藏在延迟里。

### 仍可做、但优先级下调
- **CubeS₂ AD 的 4 项无分支 gather**（`MIDTOWN_PERF_REDESIGN.md` P1）：
  SASS 模型 1737→483 instr/atom、regs 63→39、occupancy 4→6 block；
  但 gather 只占 AD mesh 的 ~50%，预计 mesh −13~20%、LR −1~3%。
  不影响本表的结论，故未在本轮落地。
- **QuadS 稠密 stencil pass**（P2）：+216 B/atom。98k +21 MiB、331k +72 MiB，
  会把 QuadS-6 的 1086 MiB 显存优势吃掉，换 5–15% 时间。
  **在"显存也要赢"的目标下不做**；如果只追时间，可单独 A/B。
- **compact green**（P3）：`f_gp` full-nx 8 B/pt → 半谱 4 B/pt，
  331k/144³ 省 11.4 MiB，384³ 省 226 MiB；与 per-config 无关，仍建议做。
- **float 力/势 + 单 workspace**（P4）：省 ~10 MiB、少 3 个 launch；
  不影响本表，仍建议做。

## 4. 对论文的修正

`papers/sog-lib-paper/sections/06-results.tex` 的 Performance 小节已重写为
**AD-only**（用户要求：只区分样条，不区分微分方法）：
- Table 6 为 6 行：B-spline 4/6、CubeS₂ 4/6、QuadS 4/6，无 mode 列；
- Table 7 为 3 行：B-spline、CubeS₂-6、QuadS-6；
- Fig. 9 用 `backend_ad_*.csv` 重绘为 6 根柱 + 331k 三角；
- Fig. 8 的 DPA4C-SOG 曲线用 post-fix 的 **AD** 2M 数据重绘；
- 正文结论：4 阶 CubeS₂/QuadS 两强；6 阶 CubeS₂/QuadS 比 B-spline
  快/省约 6%；不再出现 ik 的比较、推荐或"ik 网格更粗"的错误口径；
- Table 5（max-size）的 DPA4C-SOG np=1 从 21,876 改为 **14,574 MiB**，
  np=2 从 10,184 改为 **9,452**；正文 "14.7 GiB / 7 kB per atom" 改为
  "7,426 MiB / ~3.8 kB per atom"；np=4 AD 点未测（GPU 占用），已删行。


## 5. 4-term AD gather 落地（2026-11-13）

- 实现：`sog_gpu.cu` 的 `k_gather_cubes2_term<ORD>` + `Cs2Basis` 共享基表 +
  `c_cs2_terms[2][96]`/`c_poly`/`c_dpoly`；host `cs2gen::build_and_check`
  在 `sog_gpu_create` 时用 `sog_spline.h` 的参考函数自检（300 个随机点/节点，
  阈值 1e-9），失败则 `term_ok=0` 自动回退旧 gather。
- 语义：4 项无分支收缩，节点顺序 `c_nodes4s/6s`；每 block 8 个原子，
  basis 表 8×42 floats；每节点 4 次 uint4 查表 + 12 次 shared 读。
- A/B（op graph replay，98k）：
  - mid4ad 0.4058 → 0.3863 ms（−4.8%）
  - mid6ad 0.3780 → 0.3569 ms（−5.6%）
  - mid6ad@108 0.3843 → 0.3636 ms（−5.4%）
- 108³ 相位：CubeS₂-6 gather 0.189→**0.160**，mesh sum 0.380→**0.344**
  （−9.5%）；全步 3 次中位数：CubeS₂-6 1.207/4.933 ms（旧 1.225/4.999）；
  CubeS₂-4 的 term 版全步略差（1.203/4.849 vs 1.181/4.794），
  因此 **默认只对 6 阶启用**，4 阶走旧 gather；`SOG_TERM_GATHER=0` 可关，
`SOG_TERM_GATHER=2` 可强制 4/6 阶都走 term 做 A/B。
- 数值：相对旧 gather，energy 1.8e-7、force 9.4e-7、virial 9.7e-7；
  `test_mesh_backend` 20 + `test_long_range_sum` 50 全过。
- 限制：slab/brick 仍走旧 gather，replicated 与 slab 的 AD 差 ~1e-6；
  若需 Tier-3 bitwise parity，需把 term 收缩移植到 brick 路径。
