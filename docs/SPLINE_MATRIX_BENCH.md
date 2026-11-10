# 极端大体系下的样条矩阵：复制网格 vs 分布式网格

十种等精度样条配置（B-spline 4/6、CubeS₂ 4/6 ×{ad,ik}、QuadS 4/6 ×{ad,ik}）×
两种网格（replicated / z-slab）× 两个极端尺寸（526 848 / 1 119 744 原子）的
**每步时间**与**峰值显存**。装置链 `pair_style deepmd/kk`，同一份 checkpoint 冻结。

- 脚本：`docs/bench_spline_matrix.py`（CSV：`sog_work/bench/matrix_spline_modes/spline_matrix.csv`）
- 产物：`sog_work/mdsuite/frozen/midtown_vs_bspline/lr_*.pt2`
- 数据：`sog_work/bench/data_{526848,1119744}.lmp`（水，`units metal`）

> **一句话结论**
> 1. **复制网格整体更划算**：四种 ik 配置根本走不了分布式（自动回落 replicated），
>    五种 AD 配置里四种 slab 更慢（1.09–2.06×），换来的显存只有 8.8–39%，而网格 ≤320³ 时
>    slab 反而**多费 4.6–13.6% 显存**。唯一速度正收益是 QuadS-6 AD（0.87–0.89×）。
> 2. **速度由"每原子赋值"（spread/gather）决定，不由网格决定**：网格从 225³ 到 960³ 变 78 倍，
>    谱部分只占每步 2–5 ms；而 **B-spline 的赋值 17 ns/atom、QuadS 是 49 ns/atom**，
>    52 万原子下这一项就差 16 ms。
> 3. 这就是 **QuadS-6 ik 网格更小却更慢**的原因：它省下的是网格（−27%，≈0.2 ms），
>    多付的是每原子赋值（+16 ms）；ik 本身在 QuadS 上反而**省 3 ms**。

---

## 0. 环境验证

构建（`sog_work/mdsuite/build_lrpt212_full.sh --build`）：gcc 13.3.1 / cmake 3.26.3 /
**CUDA 12.4**（系统默认 11.8 不认 `-std=c++20`）/ torch 2.12.1+cu126 / Python 3.11.17；
五个目标全编、canonical op 的 `force_mode` schema 校验通过；Kokkos 插件
`/tmp/kk_lrpt_build/libdeepmd_lmp.so` 增量重编、`ldd` 无 `not found`。

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

> 补齐了原本缺失的 `lr_quads4.pt2` / `lr_quads6.pt2`（QuadS 的 AD 版本）。冻结必须
> `module load gcc/13`——AOTI wrapper 用系统 g++ 8.5 编会报 `-std=c++20: unrecognized`。

## 1. 方法

- 装置链：`pair_style deepmd/kk <pt2>` + `kspace_style none`（长程在图内，`pme_mesh` 拓扑）。
- 三种布置：`replicated np=1`、`replicated np=2`、`z-slab np=2`（`SOG_MIDTOWN_SLAB=1`）。
- **每步时间**取 LAMMPS `Loop time / steps`；显存取 `nvidia-smi` 采样峰值减去运行前基线。
- 主表用 `run 6`；**§4.5 的分解用 `run 5` + `run 15` 取第二段**（剔除首次邻居表/图捕获），
  两法在重叠配置上相差 2–9%（主表偏高）。
- **ik × slab 的四个配置会自动回落到 replicated**，表里标 `(回落)`——那不是分布式测量。

## 2. 526 848 原子（L = 174.23 Å）

| 模式 | 网格 | rep np=1 | rep np=2 | slab np=2 | slab/rep 时间 | slab/rep 显存 |
|---|---|---|---|---|---|---|
| B-spline-4 ad | 750³ | 98.9 ms / 11 980 MiB | 77.2 / 10 768 | 153.7 / 9 738 | **1.99×** | **−9.6%** |
| B-spline-6 ad | 250³ | 57.4 / 4 260 | **37.5 / 3 048** | 41.7 / 3 462 | 1.11× | **+13.6%** |
| CubeS₂-4 ad | 1125³ | ✗ 26.5 GiB | ✗ | ✗ | — | — |
| CubeS₂-4 ik | 432³ | 77.0 / 9 808 | 55.2 / 6 804 | (回落) 55.6 / 6 804 | 1.01× | 0 |
| CubeS₂-6 ad | 384³ | 65.9 / 7 840 | 44.2 / 4 836 | 56.0 / 4 106 | 1.27× | **−15.1%** |
| CubeS₂-6 ik | 256³ | 80.5 / 6 882 | 56.8 / 3 878 | (回落) 57.6 / 3 878 | 1.01× | 0 |
| QuadS-4 ad | 750³ | 100.8 / 18 978 | 77.7 / 15 974 | 154.8 / 9 738 | **1.99×** | **−39.0%** |
| QuadS-4 ik | 405³ | 77.6 / 9 146 | 55.3 / 6 142 | (回落) 55.4 / 6 142 | 1.00× | 0 |
| QuadS-6 ad | 320³ | 79.2 / 7 096 | 57.3 / 4 092 | **50.9 / 3 730** | **0.89×** | **−8.8%** |
| QuadS-6 ik | 225³ | 74.5 / 6 622 | **51.7 / 3 618** | (回落) 51.4 / 3 618 | 0.99× | 0 |

## 3. 1 119 744 原子（L = 224.00 Å）

| 模式 | 网格 | rep np=1 | rep np=2 | slab np=2 | slab/rep 时间 | slab/rep 显存 |
|---|---|---|---|---|---|---|
| B-spline-4 ad | 960³ | ✗ OOM | 155.7 / 21 748 | 321.5 / 18 952 | **2.06×** | **−12.9%** |
| B-spline-6 ad | 320³ | 117.4 / 8 284 | **78.0 / 5 562** | 84.8 / 5 818 | 1.09× | **+4.6%** |
| CubeS₂-4 ad | 1440³ | ✗ | ✗ | ✗ | — | — |
| CubeS₂-4 ik | 540³ | 154.9 / 18 848 | 111.6 / 12 472 | (回落) 111.4 / 12 472 | 1.00× | 0 |
| CubeS₂-6 ad | 486³ | 132.4 / 15 132 | 89.4 / 8 758 | 111.3 / 7 096 | 1.25× | **−19.0%** |
| CubeS₂-6 ik | 324³ | 162.3 / 13 190 | 117.9 / 6 816 | (回落) 118.0 / 6 816 | 1.00× | 0 |
| QuadS-4 ad | 960³ | ✗ OOM | **✗ OOM (rc=134)** | **322.0 / 18 952** | — | **唯一可跑** |
| QuadS-4 ik | 540³ | 160.8 / 18 822 | 118.3 / 12 446 | (回落) 118.3 / 12 446 | 1.00× | 0 |
| QuadS-6 ad | 400³ | 159.4 / 13 562 | 115.6 / 7 186 | **101.1 / 6 310** | **0.87×** | **−12.2%** |
| QuadS-6 ik | 288³ | 149.6 / 12 704 | **105.9 / 6 328** | (回落) 105.9 / 6 328 | 1.00× | 0 |

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

最大差 **0.0068 eV / 526 848 原子 = 1.3e-8 eV/atom**（8e-11 相对），压强 **1.6e-5 相对**。
远低于网格规则设定的 1e-4 力精度目标——网格误差只影响长程项，而长程只是总能量的一小部分。
**后端/阶数/力模式的选择不改变物理。**

### 4.2 时间：网格不是瓶颈

同一尺寸内，网格从 225³（11.4 M 点）到 750³（422 M 点）变 **37 倍**，每步时间只从 74.5 到
98.9 ms（**1.33 倍**）；1.12 M 上 288³→540³ 也是 106 → 112 ms。**谱部分每步只有 2–5 ms**，
其余全是 pair + k-space 的每原子赋值。决定速度的是**样条阶数**（4 阶 AD 逼出 750³–1440³ 的
网格、显存跟着涨），以及**每原子赋值的实现**（§4.5）。

`rep np=2` 相对 `np=1` 加速 1.28–1.53×，没有一项负加速。

### 4.3 显存：分布式只在网格大时赢

| 网格量级 | slab 相对 replicated |
|---|---|
| 750³–960³（bs4 / quads4） | **−9.6% ~ −39%**，代价 1.99–2.06× 时间 |
| 384³–486³（mid6 ad / quads6 ad） | −8.8% ~ −19%，代价 0.87–1.27× 时间 |
| 225³–320³（bs6 / quads6ik） | **+4.6% ~ +13.6%（反而更费）** |

分布式每 rank 多付 NCCL 缓冲与 ghost 层；网格 ≲400³ 时这份固定开销超过 mesh 的 1/np 节省。
判据是**网格尺寸**，不是原子数。与 `docs/SLAB_DEPLOYMENT.md`（单算子尺度 ~500³ 的拐点）一致。

> **显存的一个额外事实**：邻表重建（`neigh_modify every 20`）会瞬时多占一大块。6 步的 run
> 从不触发重建；25 步的 run 会。实测 526 848 原子 bs4 np=2：**10.8 GiB（未重建）→ 18.1 GiB
> （重建时）**。做容量规划要按后者。

### 4.4 两个反直觉的点

**(a) QuadS-6 AD 的 slab 比 replicated 快 12–13%**（0.89× / 0.87×，两个尺寸一致）。QuadS 求解器
每 rank 持**全复谱**（`f_rho`/`f_pot`，16 B/格点）；z-slab 把它切成 1/np，省下的带宽大于转置代价。
CubeS₂/B-spline 只持半谱 + 实网格，切分的收益压不过转置。

**(b) QuadS-4 AD 在 1.12 M 原子时只有 slab 能跑**：960³ 的复谱在 replicated 下 np=1/np=2 双双
OOM（np=2 rc=134），slab np=2 以 18 952 MiB 跑通。**本次唯一"分布式从可选变必须"的场景。**

### 4.5 为什么 QuadS-6 ik 网格更少却更慢

先看事实（526 848 原子、np=2、`run 5`+`run 15` 的稳态值，减去 SR-only 基线的 25.7 ms）：

| 配置 | 网格 | 每步 (ms) | k-space 净成本 (ms) |
|---|---|---|---|
| **SR only**（无长程） | — | **25.7** | 0 |
| B-spline-6 ad | 250³ (15.6 M) | 36.7 | **11.0** |
| B-spline-6 ad | 375³ (52.7 M) | 41.8 | 16.1 |
| CubeS₂-6 ad | 225³ (11.4 M) | 36.8 | 11.1 |
| CubeS₂-6 ad | 384³ (56.6 M) | 42.5 | 16.8 |
| CubeS₂-6 ik | 225³ (11.4 M) | 54.3 | 28.6 |
| CubeS₂-6 ik | 256³ (16.8 M) | 55.5 | 29.8 |
| CubeS₂-6 ik | 384³ (56.6 M) | 65.2 | 39.5 |
| QuadS-6 ad | 225³ (11.4 M) | 52.8 | 27.1 |
| QuadS-6 ad | 320³ (32.8 M) | 54.8 | 29.1 |
| QuadS-6 ik | 225³ (11.4 M) | **49.9** | **24.2** |
| QuadS-6 ik | 320³ (32.8 M) | 54.2 | 28.5 |

> 同一 backends/力模式、**只改间距**的对照：`lr_mid6_sp160`(225³)、`lr_mid6ik_sp462`(384³)、
> `lr_quads6_ad_sp798`(225³)、`lr_quads6ik_sp577`(320³)，由
> `freeze_lr_variants.py --spacing` 冻结。

**拟合 `k-space = A + B·N_百万点`**（每组两条以上曲线，残差 <0.2 ms）：

| 配置 | `A`（每步固定+每原子，ms） | `B`（每百万格点，ms） |
|---|---|---|
| B-spline-6 ad | **8.9** | 0.137 |
| CubeS₂-6 ad | 9.7 | 0.126 |
| CubeS₂-6 ik | **25.9** | 0.241 |
| QuadS-6 ad | **26.0** | **0.093** |
| QuadS-6 ik | 21.9 | 0.201 |

`A` 主要是**每原子赋值**（12 288 原子上 k-space 净成本只有 0.6–1.0 ms，与按原子线性外推一致）：

| 配置 | 每原子赋值 |
|---|---|
| B-spline-6 ad | ≈ **17 ns/atom** |
| CubeS₂-6 ad | ≈ 18 ns/atom |
| QuadS-6 ad | ≈ **49 ns/atom** |
| CubeS₂-6 ik | ≈ 49 ns/atom |
| QuadS-6 ik | ≈ 42 ns/atom |

**对照实验（12 288 原子、np=2，每步 ms）**：SR 1.3 / B-spline-6 1.9 / CubeS₂-6 1.9 /
CubeS₂-6 ik 2.3 / QuadS-6 2.3 / QuadS-6 ik 2.2。网格在这里小 40 倍，`B·N` 可以忽略，
所以差值就是每原子项：
- QuadS-6 − B-spline-6 = **+0.4 ms / 12 288 = +31 ns/atom**（52 万原子 → +16 ms，与上表吻合）；
- CubeS₂-6 ik − CubeS₂-6 ad = **+0.4 ms / 12 288 = +33 ns/atom**（52 万原子 → +17 ms，吻合）。

**于是 QuadS-6 ik 的 49.9 ms 可以逐项拆开**（vs B-spline-6 ad 的 36.7）：

| 项 | 影响 |
|---|---|
| QuadS 的每原子赋值比 B-spline 贵 ~31 ns/atom × 526 848 | **+16.3 ms** |
| QuadS 里 ik 比 ad 省的一部分（见下） | **−2.9 ms** |
| 网格 225³ vs 250³（少 27% 格点，`B`≈0.2） | −0.6 ms |
| 合计 | **+12.8 ms ≈ 实测 +13.2 ms** |

**结论：QuadS-6 ik 输在"每原子赋值"，不是"网格"，也不是"ik"。**

- QuadS 的可分离窗口赋值比 B-spline 的三次 B-样条贵 **≈3 倍**（49 vs 17 ns/atom）；
  52 万原子下这一项就是 16 ms，而两者**全部**谱面开销只有 2–3 ms（`B·N`）。
  所以"网格少 27%"最多值 0.6 ms。
- **ik 在 QuadS 上反而快 2.9 ms**：ik 的力用 3 张梯度网格做"值插值"（每原子 3 次、无权重导数），
  而 QuadS 的 ad 需要**带权重导数**的 gather；后者的额外开销大于 3 次值插值。
- 对 **CubeS₂** 则相反：它的 ad gather 很便宜（18 ns/atom），ik 换成 3 次梯图插值要
  +33 ns/atom（52 万原子 +17 ms），再叠加 3 次额外逆变换的 `B`（0.126→0.241）。
  **所以跑 ik 要选 QuadS，跑 ad 要选 B-spline/CubeS₂。**
- 附带一条普适结论：**在这个尺度上 k-space 是"赋值受限"而不是"FFT 受限"**。想加速应该优化
  spread/gather，而不是缩网格——缩网格得到的收益被每原子项淹没。

### 4.6 失败点

| 配置 | 尺寸 | 原因 |
|---|---|---|
| CubeS₂-4 ad | 两个尺寸 | 1125³ / 1440³ → **26.5 / 55 GiB**，超 24 GiB 卡；4 阶 AD 力在 1e-4 下本就不可用 |
| B-spline-4 ad | 1.12 M np=1 | 960³ 网格 + 模型挤一张卡：`CUDA out of memory. Tried to allocate 16.65 GiB` |
| QuadS-4 ad | 1.12 M np=1/2 | 同上，且 QuadS 多一份全复谱 |

失败都报**可读**信息：midtown/QuadS 后端在建 host 绿函数前按 `n³×20 B` 预检查；B-spline 后端
（`mesh_sum.cu`）仍是 torch 原生 OOM（但点名了张量大小）。两者都不再是
`Expected a proper Tensor ... 'self'`。

---

## 5. 结论与选型

**① 当前复制网格整体更划算。** 十个配置里：4 个 ik 配置的"分布式"其实回落到了 replicated
（分布式求解器没有 i·k 通路），5 个 AD 配置里 **4 个 slab 更慢**（1.09–2.06×）换 8.8–39% 显存，
网格 ≤320³ 时更差（多费 4.6–13.6%）。**只有两个例外**：
- QuadS-6 AD 的 slab 快 12–13%（唯一速度正收益）；
- QuadS-4 AD @1.12 M 只有 slab 能跑（显存救援）。

**② 最快配置**：**B-spline-6 + replicated np=2**——526 k 上 37.5 ms / 3 048 MiB，
1.12 M 上 78.0 ms / 5 562 MiB，两个尺寸都是全场最快、显存最小。

**③ 后端与力模式的搭配有讲究**（§4.5 的量化结论）：

| 力模式 | 最优后端 | 理由 |
|---|---|---|
| `ad` | B-spline / CubeS₂ | 赋值便宜（17–18 ns/atom），QuadS 的 49 ns/atom 是纯亏 |
| `ik` | **QuadS** | ik 的值插值 gather 比 QuadS-ad 的权重导数 gather 便宜；CubeS₂-ik 要多付 33 ns/atom |

**④ 需要 CubeS₂ 后端时**用 6 阶（`ad` 或用 `ik`），**绝不要 4 阶 AD**（1125³ / 26.5 GiB 起）。

**⑤ 分布式**是显存救援手段：判据是**网格 ≥700³**（或 QuadS 全复谱撑爆单卡）；
其余场景 8–13% 显存换 2× 时间是亏的。论文多 GPU 数字用 replicated，把
"QuadS-6 AD 分布式快 12%""QuadS-4 大体系只能分布式"作为限定说明保留。

---

## 6. 复现

```bash
# 构建（gcc 13 / cmake 3.26.3 / CUDA 12.4 / torch 2.12.1+cu126）
sog_work/mdsuite/build_lrpt212_full.sh --build

# 补齐产物（冻结前必须 module load gcc/13）
PYTHONPATH=$REPO python sog_work/tools/freeze_lr_variants.py --out $FR/lr_quads4.pt2 --backend quads --order 4
PYTHONPATH=$REPO python sog_work/tools/freeze_lr_variants.py --out $FR/lr_quads6.pt2 --backend quads --order 6
# §4.5 的对照产物
PYTHONPATH=$REPO python sog_work/tools/freeze_lr_variants.py --out $FR/lr_mid6_sp160.pt2    --backend midtown --order 6 --spacing 0.776378136732931
PYTHONPATH=$REPO python sog_work/tools/freeze_lr_variants.py --out $FR/lr_mid6ik_sp462.pt2   --backend midtown --order 6 --force-mode ik --spacing 0.46202053524615094
PYTHONPATH=$REPO python sog_work/tools/freeze_lr_variants.py --out $FR/lr_quads6_ad_sp798.pt2 --backend quads --order 6 --spacing 0.7979114851390419
PYTHONPATH=$REPO python sog_work/tools/freeze_lr_variants.py --out $FR/lr_quads6ik_sp577.pt2 --backend quads --order 6 --force-mode ik --spacing 0.5772856682094816

# 主表（§2–§3）
python docs/bench_spline_matrix.py 526848 1119744
# 稳态/分解（§4.5）：两段 run，取第二段
STEPS=15 WARM=5 ONLY=lr_sr,lr_bs6,lr_mid6,lr_mid6ik,lr_quads6,lr_quads6ik \
  python docs/bench_spline_matrix.py 526848
```
