# QuadS 的 stencil 实现与结论

2026-11-18。实验开关：`SOG_QUADS_STENCIL=1`（默认关，旧路径保留）。

## 1. 背景

QuadS 本身是**可分离**的：

\[
W_{\rm QuadS}(\theta,i)=\prod_{d=x,y,z} S_{\xi^\circ}(\theta_d,i_d),
\]

但原有 brick 路径 (`k_spread_quads_brick` / `k_gather_quads_brick`)
使用的是排序好的 3D 节点表 `QNodeS`：

- 每个节点读 `dx,dy,dz,a,b,c`；
- 每个节点做 3 次 `brick_coord`；
- 每个节点再读 1D 权重表。

这不是 B-spline `AxisStencil` 那种**逐轴 stencil + lane 走 z** 的调度。

## 2. 新实现

新增：

- `QuadsBrickTab<NORD>`：每轴 1D 值/斜率 + 裁剪后的 `cx/cy/cz`；
- `quads_fill_brick_tab()`：在 warp 内一次填好每轴 stencil；
- `k_spread_quads_brick_stencil<NORD>`；
- `k_gather_quads_brick_stencil<NORD>`。

调度：

- 一个 lane 负责一个 `(x,y)` pair；
- 在 z 方向走 `P` 个节点；
- `(x,y)` 权重和坐标在 lane 内 hoist；
- 内层只做：z 权重/斜率 shared read + mesh read + FMA。

与 B-spline `AxisStencil` 的 lane 调度一致，但 QuadS 的节点数仍是
`(2ν)^3` = 64/216（不是 CubeS₂ 的 32/88）。

## 3. 正确性

1.12M `run 0`，QuadS-6，slab：

```
old: /tmp/f_qs_old.dump
new: /tmp/f_qs_st.dump
max|ΔF| = 1.0e-5
rms     = 5.2e-7
rel     = 2.9e-6
```

与旧路径在 fp32 数值精度内一致。

## 4. 性能

1.12M、QuadS-6、1 GPU（其它 GPU 被占，单卡干净）：

| path | step |
|---|---:|
| QNodeS table | 0.8524 / 0.8531 s |
| axis stencil | 0.8486 / 0.8489 s |

约 **0.4%**，在噪声边缘；2 卡受其它训练任务污染，也只是 0.855→0.853 s。
**没有量级收益。**

## 5. 为什么没有收益

1. 原来的 `QNodeS` 表已经是“排序 + 每 lane 一个节点”的 stencil-like 路径；
2. QuadS 的 1D 权重表 (`QuadsTab`) 早就存在，新路径只是把节点 decode 换成
   逐轴坐标；
3. QuadS 仍有 **216 个节点**（order 6），mesh read/atomic 数不变；
4. 真正省节点的是 CubeS₂（88 节点），不是 stencil 调度。

## 6. 结论

- **QuadS 的 stencil 路径已实现并验证**，默认关闭；
- 性能提升约 0.4%，不值得替换生产默认；
- 需要 stencil 语义/代码统一时可开 `SOG_QUADS_STENCIL=1`；
- 生产性能仍应使用 CubeS₂-6（88 节点），而不是把 QuadS 做成 stencil 来追 CubeS₂。
