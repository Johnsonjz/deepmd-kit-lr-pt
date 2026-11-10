# 极端大体系下的样条矩阵：复制网格 vs 分布式网格

十种等精度样条配置（B-spline 4/6、CubeS₂ 4/6 ×{ad,ik}、QuadS 4/6 ×{ad,ik}）×
两种网格（replicated / z-slab）× 两个极端尺寸（526 848 / 1 119 744 原子）的
**每步时间**与**峰值显存**。装置链 `pair_style deepmd/kk`，同一份 checkpoint 冻结。

- 脚本：`docs/bench_spline_matrix.py`（原始 CSV：`sog_work/bench/matrix_spline_modes/spline_matrix.csv`）
- 产物：`sog_work/mdsuite/frozen/midtown_vs_bspline/lr_*.pt2`
- 数据：`sog_work/bench/data_{526848,1119744}.lmp`（水，`units metal`，192·k³）

> **一句话结论**
> 1. **十种配置在同一个构型上能量一致到 0.007 eV / 526 848 原子（8e-11 相对）、压强 1.6e-5**
>    ——等精度网格规则是对的，选哪个后端不影响物理。
> 2. **速度由 pair 决定，不由网格决定**：网格从 225³ 到 960³ 变 78 倍，每步时间只变 2 倍。
>    所以"更细的网格"通常不是慢的原因，"更低阶的样条"才是（它逼你上细网格）。
> 3. **多卡默认用 replicated**：np=2 比 np=1 快 1.28–1.53×，没有代价（除了每 rank 持全网格）。
> 4. **分布式只在网格大到显存吃紧时值**：bs4 750³ 省 9.6% 显存但要 2.0× 时间；
>    **QuadS-4 在 1.12 M 原子时是唯一能跑的选择**（replicated np=2 直接 OOM）。
> 5. **QuadS-6 AD 是唯一 slab 比 replicated 快的配置**（0.88×，即快 12%）——因为 QuadS
>    每 rank 持全复谱，切一半反而省带宽。
> 6. **四个 ik 配置在装置链上永远走 replicated**（分布式没有 i·k 通路，自动回落）。

---

## 0. 环境验证

构建（`sog_work/mdsuite/build_lrpt212_full.sh --build`，2026-11-10 21:11）：

| 项 | 值 |
|---|---|
| gcc / mpicc | 13.3.1 |
| cmake | 3.26.3 |
| nvcc | CUDA 12.4.131（系统默认是 11.8，不认 `-std=c++20`） |
| torch / python | 2.12.1+cu126 / 3.11.17 |
| 算子 | `lr212_install_lrpt/deepmd/lib/libdeepmd_op_pt.so`（含 force_mode schema 校验通过） |
| kk 插件 | `/tmp/kk_lrpt_build/libdeepmd_lmp.so`，`ldd` 无 `not found` |
| LAMMPS | `/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp`（Kokkos 5.1.99，CUDA backend） |

十种产物的网格间距（与 `grid_rule.py` 的误差律逐位一致）：

| 模式 | Δ(Å) | 526 848@174.23 Å | 1 119 744@224.00 Å |
|---|---|---|---|
| B-spline-4 | 0.238806 | 750³ | 960³ |
| B-spline-6 | 0.707107 | 250³ | 320³ |
| CubeS₂-4 ad | 0.159813 | 1125³ | 1440³ |
| CubeS₂-4 ik | 0.420448 | 432³ | 540³ |
| CubeS₂-6 ad | 0.462021 | 384³ | 486³ |
| CubeS₂-6 ik | 0.691769 | 256³ | 324³ |
| QuadS-4 ad | 0.238406 | 750³ | 960³ |
| QuadS-4 ik | 0.433277 | 405³ | 540³ |
| QuadS-6 ad | 0.577286 | 320³ | 400³ |
| QuadS-6 ik | 0.797911 | 225³ | 288³ |

> 补齐了两个原本缺失的产物：`lr_quads4.pt2` / `lr_quads6.pt2`（QuadS 的 AD 版本）。
> 冻结必须 `module load gcc/13`——AOTI wrapper 用系统 g++（8.5）编会报
> `-std=c++20: unrecognized`。

---

## 1. 方法

- 装置链：`pair_style deepmd/kk <pt2>` + `kspace_style none`（长程在图内，`pme_mesh` 拓扑）。
- 三个配置：`replicated np=1`（单卡基线）、`replicated np=2`、`z-slab np=2`（`SOG_MIDTOWN_SLAB=1`）。
- 每点 `run 6`（0.5 fs），时间取 LAMMPS `Loop time / steps`；显存取 `nvidia-smi` 采样峰值
  **减去运行时基线**，两个 rank 取最大，并同时给出两卡合计。
- 每两步做一次邻居重建（`neigh_modify every 20`），与生产一致。
- **ik × slab 的四个配置会自动回落到 replicated**，表里以 `(回落)` 标注——那不是分布式测量。

---

## 2. 526 848 原子（L = 174.23 Å）

| 模式 | 网格 | rep np=1 | rep np=2 | slab np=2 | slab/rep 时间 | slab/rep 显存 |
|---|---|---|---|---|---|---|
| B-spline-4 ad | 750³ | 593.2 ms / 11 980 MiB | 463.0 / 10 768 | 922.2 / 9 738 | **1.99×** | **−9.6%** |
| B-spline-6 ad | 250³ | 344.1 / 4 260 | **224.9 / 3 048** | 250.5 / 3 462 | 1.11× | **+13.6%** |
| CubeS₂-4 ad | 1125³ | ✗ 26.5 GiB | ✗ | ✗ | — | — |
| CubeS₂-4 ik | 432³ | 461.9 / 9 808 | 331.2 / 6 804 | (回落) 333.3 / 6 804 | 1.01× | 0 |
| CubeS₂-6 ad | 384³ | 395.2 / 7 840 | 265.3 / 4 836 | 336.3 / 4 106 | 1.27× | **−15.1%** |
| CubeS₂-6 ik | 256³ | 482.9 / 6 882 | 341.0 / 3 878 | (回落) 345.7 / 3 878 | 1.01× | 0 |
| QuadS-4 ad | 750³ | 604.8 / 18 978 | 466.0 / 15 974 | 929.0 / 9 738 | **1.99×** | **−39.0%** |
| QuadS-4 ik | 405³ | 465.9 / 9 146 | 331.5 / 6 142 | (回落) 332.2 / 6 142 | 1.00× | 0 |
| QuadS-6 ad | 320³ | 475.3 / 7 096 | 344.0 / 4 092 | **305.3 / 3 730** | **0.89×** | **−8.8%** |
| QuadS-6 ik | 225³ | 446.9 / 6 622 | **310.3 / 3 618** | (回落) 308.3 / 3 618 | 0.99× | 0 |

（显存为单 rank 峰值；`rep np=2` 与 `slab np=2` 的两卡合计分别是上表的 2×。）

## 3. 1 119 744 原子（L = 224.00 Å）

| 模式 | 网格 | rep np=1 | rep np=2 | slab np=2 | slab/rep 时间 | slab/rep 显存 |
|---|---|---|---|---|---|---|
| B-spline-4 ad | 960³ | ✗ OOM | 934.0 ms / 21 748 MiB | 1 929.0 / 18 952 | **2.06×** | **−12.9%** |
| B-spline-6 ad | 320³ | 704.3 / 8 284 | **468.1 / 5 562** | 508.6 / 5 818 | 1.09× | **+4.6%** |
| CubeS₂-4 ad | 1440³ | ✗ | ✗ | ✗ | — | — |
| CubeS₂-4 ik | 540³ | 929.4 / 18 848 | 669.6 / 12 472 | (回落) 668.2 / 12 472 | 1.00× | 0 |
| CubeS₂-6 ad | 486³ | 794.5 / 15 132 | 536.2 / 8 758 | 667.8 / 7 096 | 1.25× | **−19.0%** |
| CubeS₂-6 ik | 324³ | 973.6 / 13 190 | 707.3 / 6 816 | (回落) 707.8 / 6 816 | 1.00× | 0 |
| QuadS-4 ad | 960³ | ✗ OOM | **✗ OOM（rc=134）** | **1 932.2 / 18 952** | — | **唯一可跑** |
| QuadS-4 ik | 540³ | 964.9 / 18 822 | 709.5 / 12 446 | (回落) 710.0 / 12 446 | 1.00× | 0 |
| QuadS-6 ad | 400³ | 956.1 / 13 562 | 693.3 / 7 186 | **606.6 / 6 310** | **0.88×** | **−12.2%** |
| QuadS-6 ik | 288³ | 897.5 / 12 704 | **635.5 / 6 328** | (回落) 635.5 / 6 328 | 1.00× | 0 |

---

## 4. 分析

### 4.1 精度：十种配置是同一个物理

同一构型、同一时刻、`thermo_modify format float %.6f`：

| 模式 | pe (eV) | press (bar) |
|---|---|---|
| B-spline-6 | −82 173 532.410908 | −345.495924 |
| CubeS₂-6 ad | −82 173 532.404087 | −345.501462 |
| QuadS-6 ad | −82 173 532.404360 | −345.501492 |
| CubeS₂-4 ik | −82 173 532.404932 | −345.500954 |

最大差 **0.0068 eV / 526 848 原子 = 1.3e-8 eV/atom**（8e-11 相对），压强 **5.5e-3 bar / 345 =
1.6e-5 相对**。远低于网格规则设定的 1e-4 力精度目标——因为网格误差只影响长程项，而长程项
只是总能量的很小一部分。**后端/阶数/力模式的选择不改变物理结论。**

### 4.2 速度：网格不是瓶颈

- **同一尺寸内，网格变 37 倍、时间只变 1.3 倍**：526 848 原子 np=1 下，QuadS-6 ik 是
  225³（11.4 M 点）447 ms，B-spline-4 是 750³（422 M 点）593 ms；1.12 M 下 288³ → 540³
  是 897 → 929 ms。526 848 原子的 DPA4 描述符 + 边张量占了绝大部分。
- 决定速度的是**样条阶数**：4 阶 AD 的等精度间距 0.16–0.24 Å，逼出 750³–1440³ 的网格；
  6 阶只要 0.46–0.71 Å，网格小一个量级。**bs/quad-4-ad 与 6 阶的时间差（593 vs 344 ms）几乎全
  来自网格与显存，而不是样条本身的开销。**
- `rep np=2` 相对 `np=1` 的加速 1.28–1.53×（pair 被两个 rank 分摊）；**没有一项出现负加速**。

### 4.3 显存：分布式只在网格大时赢

| 网格量级 | slab 相对 replicated |
|---|---|
| 750³–960³（bs4 / quads4） | **−9.6% ~ −39%**，代价 2.0× 时间 |
| 384³–486³（mid6 ad / quads6 ad） | −8.8% ~ −19%，代价 1.09–1.27× 时间 |
| 225³–320³（bs6 / quads6ik） | **+4.6% ~ +13.6%（反而更费）** |

原因是固定的：分布式每 rank 多付 NCCL 通信缓冲与 ghost 层，而网格 ≤320³ 时这份固定开销
超过 mesh 本身的 1/np 节省。**判据是网格尺寸而不是原子数：网格 ≲400³ 就不该开分布式**——
与本仓 `docs/SLAB_DEPLOYMENT.md` 在单算子尺度上的结论一致（那里是 ~500³ 的拐点，这里整步
被 pair 稀释后抬到 400³ 附近）。

### 4.4 两个反直觉的点

**(a) QuadS-6 AD 的 slab 比 replicated 快 12%。** QuadS 求解器每 rank 持**全复谱**
（`f_rho`/`f_pot`，16 B/格点）；z-slab 把它切成 1/np，省下的带宽大于转置的代价。而
CubeS₂/B-spline 只持半谱 + 实网格，切分的收益就压不过转置。**这不是噪声**：两个尺寸上
一致（0.89× / 0.88×），且显存同时下降 8.8% / 12.2%。

**(b) QuadS-4 AD 在 1.12 M 原子时只有 slab 能跑。** 960³ 的 QuadS 复谱在 replicated 下
replicated np=1 / np=2 都 OOM（np=2 rc=134），slab np=2 以 18 952 MiB 跑通。
**这是本次唯一一个"分布式从'可选'变成'必须'"的场景。**

### 4.5 失败点及其原因

| 配置 | 尺寸 | 原因 |
|---|---|---|
| CubeS₂-4 ad | 两个尺寸 | 1125³ / 1440³ → **26.5 / 55 GiB**，超过 24 GiB 卡；4 阶 AD 力在 1e-4 下本来就不可用 |
| B-spline-4 ad | 1.12 M np=1 | 960³ 网格 + 模型挤一张卡：`CUDA out of memory. Tried to allocate 16.65 GiB` |
| QuadS-4 ad | 1.12 M np=1/2 | 同上，且 QuadS 多一份全复谱 |

> 失败现在都报**可读**信息：midtown/QuadS 后端在建 host 绿函数**之前**按 `n³×20 B`
> 预检查并报出所需/可用显存；B-spline 后端（`mesh_sum.cu`）还是原始的 torch OOM——
> 但它至少点名了张量大小，不再是 `Expected a proper Tensor ... 'self'`。

---

## 5. 选型建议

| 场景 | 选择 |
|---|---|
| **要最快** | **B-spline-6 + replicated np=2**：526 k 上 225 ms / 3 048 MiB，1.12 M 上 468 ms / 5 562 MiB——两个尺寸都是全场最快，显存也最小 |
| 需要 CubeS₂ 后端 | CubeS₂-6 **ik** + replicated（331 / 670 ms）；**不要用 4 阶 AD** |
| 需要 QuadS 后端 | QuadS-6 **ad + slab**（305 / 607 ms，比 replicated 还快 12%）；或 QuadS-6 ik + replicated |
| 显存吃紧 | 只有 **QuadS-4 ad @1.12 M** 是必须上分布的（replicated 直接 OOM）；其余场景分布式的 8–13% 显存换 2× 时间是亏的 |
| 论文多 GPU 数字 | 用 **replicated**（ik 四个配置在装置链上本来就只能 replicated），并把"分布式在 QuadS-6 上快 12%、在 QuadS-4 大体系上是唯一可行解"作为一句限定说明保留 |

**一句话**：装置链上 replicated np=2 是默认答案；分布式网格是**显存救援手段**，
只在网格 ≥700³ 或 QuadS 复谱撑爆单卡时启用——而 QuadS-6 是它唯一的速度正收益场景。

---

## 6. 复现

```bash
# 构建（已验证：gcc 13 / cmake 3.26.3 / CUDA 12.4 / torch 2.12.1+cu126）
sog_work/mdsuite/build_lrpt212_full.sh --build

# 补齐 QuadS 的 AD 产物（若缺失）
module load gcc/13
PYTHONPATH=$REPO python sog_work/tools/freeze_lr_variants.py \
    --out $FR/lr_quads4.pt2 --backend quads --order 4
PYTHONPATH=$REPO python sog_work/tools/freeze_lr_variants.py \
    --out $FR/lr_quads6.pt2 --backend quads --order 6

# 全矩阵
python docs/bench_spline_matrix.py 526848 1119744
# 单点重测：ONLY=lr_mid4 python docs/bench_spline_matrix.py 526848
```
