# 分布式网格（slab）与 ik 力模式的 LAMMPS 崩溃：报错诊断记录

> **2026-11-12 更新**：z-slab 求解器**已实现 ik 通路**，由 `SOG_SLAB_IK=1` 显式打开
> （默认仍走 replicated，默认行为不变）。实现与验证见
> `docs/PAIR_DEEPMD_KK_SOG.md` §13.1。本文保留为**诊断记录**：下面第 0 节说
> "设计如此、不是 bug" 的那一条描述的是 2026-11-10 的状态。
>
> 记录 2026-11-10 验证"所有样条模式都支持分布式网格"时撞到的三类失败，
> 以及它们**看起来相同、实际完全不同**的真因。
>
> 一句话结论：`Expected a proper Tensor but got None ... 'self'` 这条报错
> **不是** 任何一类问题的真因，而是 AOTI proxy executor 吞掉算子异常后的
> **二次症状**。真因分散在三个互不相关的地方（§3、§4、§5）。
>
> 修复与其验证见 §10；**设备链 `pair_style deepmd/kk` 的复测见 §11**
> ——同一体系比主机版快 23×，而分布式网格在设备链上慢了整整一倍。

---

## 0. 结论速览

| 观测到的现象 | 真因（stderr 里的原话） | 性质 |
|---|---|---|
| `mid4ik` / `quads4ik` + `SOG_MIDTOWN_SLAB=1` 崩溃（串行、np=1 也崩） | `latent_charge_sum: force_mode != 0 (ik) is implemented on the replicated midtown solve only; unset SOG_MIDTOWN_SLAB`（`latent_charge_sum.cu:279`）——slab 求解器只有 AD gather，没有 i·k 微分 | **设计如此**，不是 bug。**2026-11-10 19:50 起改为警告 + 回落到 replicated**（§10） |
| `mid4`（CubeS₂-4 AD，replicated）在 526 848 原子崩溃，≤98 304 正常 | `sog_gpu.cu:2696 out of memory` → `midtown: Form-A window symbol failed rc=-1`（`midtown_sum.cu:620`）——1091³ 网格的设备缓冲约 26 GiB，超过 24 GiB 卡 | **资源上限**，不是 bug |
| 上面两类都只报 `Expected a proper Tensor but got None ... 'self'` | AOTI proxy executor 捕获算子异常、只往 stderr 打一行 `[E...]` → 该输出变成 undefined → `flatten_canonical_atom_virial` 先撞上 | **误报掩盖真因**，是真 bug（诊断性） |

另外一条**结论性的工程判定**（见 §7）：论文的多 GPU 图应当用 **replicated**，
不要用分布式 slab。

---

## 1. 环境与可复现路径

| 项 | 值 |
|---|---|
| 代码树 | `/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt`（HEAD `fc7f43038`） |
| python / torch | `/data/home/jiangzhen/.conda/envs/lr212/bin/python`，torch 2.12.1+cu126 |
| 算子 `.so` | `lr212_install_lrpt/deepmd/lib/libdeepmd_op_pt.so`（2026-11-10 18:00，从 HEAD 重编） |
| 后端 `.so` | 同目录 `libdeepmd_backend_ptexpt.so`（2026-11-09 23:48） |
| LAMMPS | `/data/home/public/jiangzhen/dp/lmp_kokkos_install/bin/lmp`，插件 `/tmp/lmpplugin_lrpt/dpplugin.so` |
| 冻结产物 | `/data/home/public/jiangzhen/dp/sog_work/mdsuite/frozen/midtown_vs_bspline/lr_{bs4,mid4,mid4ik,quads4ik,bs6,mid6,mid6ik}.pt2` |
| 运行目录 | `/data/home/public/jiangzhen/dp/sog_work/bench/matrix_slab_vs_rep/`（`slab_vs_rep.np2.csv` 与全部 `log.*` 都在这里） |
| 驱动脚本 | `/data/home/public/jiangzhen/dp/sog_work/bench/bench_250k_lrpt.py`（`write_input` / `parse_log` / `ENV_BASE`） |

算子层（不经 LAMMPS）的等价性脚本：本目录 `check_all_spline_modes_slab.sh` +
`check_spline_parity_across_ranks.py` + `compare_spline_parity.py`（§9.3）。
只给真因用的最小探针：本目录 `probe_mesh_operator_error.py`（§9.1）。

---

## 2. 报错原文

所有失败场景在 LAMMPS 里都长这样（`log.mid4ik_slab.526848.np2`）：

```
ERROR on proc 0: DeePMD-kit C API Error: DeePMD-kit Error: DeePMD-kit PyTorch
Exportable backend error: Expected a proper Tensor but got None (or an undefined
Tensor in C++) for argument #0 'self'
Exception raised from checked_cast_variable at .../VariableTypeManual.cpp:66

frame #5: at::_ops::view::call(...)
frame #6: at::native::reshape_symint(...)
frame #8: at::_ops::reshape::call(...)
frame #9: deepmd::flatten_canonical_atom_virial(...)          <-- 第一现场
frame #10: deepmd::DeepPotPTExpt::compute<double, ...>(...)
frame #13: deepmd::DeepPotPTExpt::computew(...)
frame #14: deepmd::DeepPot::compute<double>(...)
frame #18: LAMMPS_NS::PairDeepMD::compute(int, int)
```

堆栈最内层是 **`flatten_canonical_atom_virial`**，函数体是
（`source/api_cc/include/commonPT.h:988`）：

```cpp
inline void flatten_canonical_atom_virial(
    std::map<std::string, torch::Tensor>& output_map) {
  const auto entry = output_map.find("atom_virial");
  if (entry == output_map.end()) {
    return;                       // 键不存在 -> 安全返回
  }
  entry->second = entry->second.reshape({entry->second.size(0), 9});  // 键在、张量未定义 -> 崩
}
```

即 `.pt2` 的 `output_keys` 里**有** `atom_virial` 这一项（六个产物都有，
`metadata.json` 中 `do_atomic_virial=True`），但这一次运行该槽位拿到的是
**undefined tensor**。报错因此落在"reshape 一个 None"上，与真正的失败原因无关。

---

## 3. 真因一（诊断性 bug）：AOTI 把算子异常吞掉了

### 3.1 机制

`.pt2` 的自定义算子通过 AOTI proxy executor 调用（`aoti_torch_proxy_executor_call_function`）。
算子内部抛出的 C++ 异常在那里被捕获，**只往 stderr 打一行 `[E...]`**，
然后让该算子的输出张量保持 undefined，模型照常"返回" 6 个输出：

```
[E1110 19:29:44.501769719 shim_common.cpp:1343] Exception in aoti_torch:
    latent_charge_sum: force_mode != 0 (ik) is implemented on the replicated
    midtown solve only; unset SOG_MIDTOWN_SLAB
```

异常**不再抛出**，于是上层看到的是"某个输出是未定义张量"，
而不是算子报的错。

### 3.2 证据 A：Python 路径能看到真消息

用 pt_expt 的 Python 接口跑同一个 `lr_mid4ik.pt2`（L=12 Å、64 原子、`SOG_MIDTOWN_SLAB=1`）：

```
$ SOG_MIDTOWN_SLAB=1 python probe_mesh_operator_error.py lr_mid4ik
[E1110 ...] Exception in aoti_torch: latent_charge_sum: force_mode != 0 (ik) is
implemented on the replicated midtown solve only; unset SOG_MIDTOWN_SLAB
Exception raised from latent_charge_sum at
    .../source/op/pt/long_range/latent_charge_sum.cu:279
OK nan                      <-- 注意：Python 路径不报错，返回 NaN
```

同一条命令把 `SOG_MIDTOWN_SLAB` 去掉就正常（`E = -8658.217698`）。
**这就是 slab + ik 的全部真因**：一个主动的 `TORCH_CHECK`。

### 3.3 证据 B：Python 路径会静默返回 NaN

`OK nan` 那一行更值得注意——异常被吞后，AOTI 用未初始化的输出继续算，
Python 侧拿到 `energy = nan` 而**不抛异常**。也就是说：

- 走 C++（LAMMPS）→ 因为 `flatten_canonical_atom_virial` 先撞上 undefined 而崩（至少崩了）；
- 走 Python `.pt2` 路径 → 拿到 NaN，**静默错误**，比崩更危险。

### 3.4 证据 C：同一个误报掩盖了两种毫无关系的真因

`mid4`（AD）在 526 848 原子崩溃时报的是**一模一样**的字符串，
但那里的算子其实是在给 1091³ 网格申请设备缓冲时 `cudaMalloc` 失败（§5.1）。
一个字符串对应两个毫不相干的真因（一条设计拒绝、一条显存不足），
说明它不携带任何诊断信息。

### 3.5 影响面

任何 `.pt2` 里任何自定义算子抛异常，都会退化成这条误导性消息。
包括但不限于：网格过大 OOM、形参/盒子不合法、kernel 参数越界。
排查时第一小时基本都浪费在这里。

### 3.6 修法（2026-11-10 晚已落地）

`require_defined_outputs()`（`source/api_cc/include/commonPT.h`）在每次
`extract_outputs` 之后点名列出 undefined 的槽位并指向 stderr 的 `[E...]` 行；
`flatten_canonical_atom_virial` 不再对 None 调 `reshape`。
用法与现状见 `PAIR_DEEPMD_KK_SOG.md` §12.2，实测见本文 §10.2。

---

## 4. 真因二（设计如此）：ik 力模式没有分布式实现

当时的代码（`source/op/pt/long_range/latent_charge_sum.cu:273-290`）：

```cpp
const char* slab_env = std::getenv("SOG_MIDTOWN_SLAB");
if (slab_env != nullptr && std::atoi(slab_env) != 0 &&
    (mesh_backend == 0 || mesh_backend == 1 || mesh_backend == 2)) {
  TORCH_CHECK(periodic, "the slab backend needs periodic frames");
  // The slab and distributed solves carry the AD gather only; running them
  // under an ik request would silently return a different force law.
  TORCH_CHECK(force_mode == 0,
              "latent_charge_sum: force_mode != 0 (ik) is implemented on the "
              "replicated midtown solve only; unset SOG_MIDTOWN_SLAB");
  ...
  return deepmd::long_range::midtown_sum_slab(...);
}
```

`SOG_MIDTOWN_DIST` 分支（`midtown_sum_local`）有同样的一条（`:304`）。
**2026-11-10 晚已改为"警告 + 回落 replicated"**（`SOG_SLAB_STRICT=1` 恢复硬报错），
见 §10。

要点：

- **三种样条（B-spline / CubeS₂ / QuadS）确实都支持 slab**：dispatch 里
  `mesh_backend == 0 || == 1 || == 2` 全部进 `midtown_sum_slab`，`quads=` /
  `bspline=` 标志区分。这条 dispatch（`10792742f`、`645ab3215`）是对的，
  算子层等价性也验证过（§6.1）。
- **卡住的不是样条，是 `force_mode`**。slab 求解器的力只来自 AD gather；
  没有实现 `i·k` 的 Fourier 微分，若强行走 slab 会**悄悄换掉力定律**，
  所以这里必须拒绝。
- 这不是回归；要支持就是给 slab 求解器补一条 ik 通路（新功能）。
- **2026-11-10 19:50 的处理**：不补通路，改为**警告 + 回落到 replicated**——力定律完全相同，
  只是每 rank 持全网格。理由与验证见 §10。

---

## 5. 真因三（资源上限）：CubeS₂-4 AD 的网格在大盒子里放不下

4 阶 AD 的等精度间距是 `Δ = 0.1598 Å`（`grid_rule.py` 的 `FORCE_RULE`），
而 4 阶 ik 是 `0.4204 Å`。在 bench 用的盒子里：

| 盒子 | L (Å) | bs4 `Δ=0.2388` | **mid4 `Δ=0.1598`** | mid4ik `Δ=0.4204` | quads4ik `Δ=0.4333` |
|---|---|---|---|---|---|
| 526 848 原子 | 174.23 | 730³ / 1.45 GiB | **1091³ / 4.84 GiB** | 415³ / 0.27 GiB | 403³ / 0.24 GiB |
| 1 119 744 原子 | 219.56 | 920³ / 2.90 GiB | **1374³ / 9.66 GiB** | 523³ / 0.53 GiB | 507³ / 0.49 GiB |

（"每 array"= `n³ × 4 B` 的 fp32 网格。实跑的设备缓冲是
`f_gp` 8 B + `f_rho_r` 4 B + `f_rho_fx` 8 B ≈ **5×** 这个数，见 §5.1；
另外 host 侧还要一张 56 B/格点的 fp64 绿函数表。）

实测（`mid4`、np=1、replicated，GPU2 空卡）：

| N | 192 | 1 536 | 12 288 | 24 000 | 41 472 | 98 304 | 526 848 |
|---|---|---|---|---|---|---|---|
| 结果 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ 显存不足 |
| `loop` (s/step) | 0.00147 | 0.00732 | 0.0447 | 0.0867 | 0.142 | 0.326 | — |

（139 968 / 192 000 两个点没跑完，被显式终止。）

### 5.1 526 848 那一次的完整链路（stderr 落盘后才看得到）

```
sog_gpu CUDA .../source/op/pt/long_range/midtown/sog_gpu.cu:2696 out of memory
[E1110 19:34:23 shim_common.cpp:1343] Exception in aoti_torch:
    midtown: Form-A window symbol failed rc=-1
Exception raised from midtown_sum at .../source/op/pt/long_range/midtown_sum.cu:620
```

对应代码链：

1. `midtown_sum.cu:591` → `sog_green::build_green(nx, ny, nz, …)` 在 **host**
   上逐格点构造 fp64 绿函数表（注释写明 56 B/格点）。1091³ = 1.30e9 格点
   ⇒ 数十 GB host 内存；实测 `RSS = 36 GB`、单核 100%、GPU 0%，
   这一段就烧掉约一分钟。
2. `midtown_sum.cu:596` → `sog_gpu_setup_f32` → `setup_replicated`
   （`sog_gpu.cu:2671`）按 `ng = nx·ny·nz` 申请设备缓冲：
   `f_gp`（float2，8 B）= **10.4 GiB**、`f_rho_r`（float，4 B）= **5.2 GiB**、
   `f_rho_fx`（uint64，8 B）= **10.4 GiB**，合计 ≈ **26 GiB > 24 GiB 卡**。
3. 前两个分配成功（已占 ~15.6 GiB），第三个
   `CK(cudaMalloc(&s->f_rho_fx, …))` 在 `sog_gpu.cu:2696` 报 `out of memory`；
   错误码上传后 `sog_gpu_green_form_a` 返回 −1，`midtown_sum.cu:620` 的
   `TORCH_CHECK(fa_rc == 0, "midtown: Form-A window symbol failed rc=", fa_rc)` 抛出。
4. 异常被 AOTI proxy executor 捕获（§3）→ 输出 undefined →
   LAMMPS 侧只剩 `Expected a proper Tensor`。

所以 `mid4` 不是"坏了"：≤98 304 原子一直是对的，只是网格随盒子**三次方**涨，
到 526 848 原子时单卡就放不下（而且 host 侧建表本身还要几十 GB 内存、一分钟）。
**它在这条基准里本来就不是可用配置**——这正是"4 阶 AD 在几十万原子下不可用、
ik 把它救回来"的量化版。

---

## 6. 验证矩阵：本来要验证的两层

### 6.1 算子层：三种样条 × 4/6 阶，np=1 replicated vs np=2 slab — **全部通过**

400 原子、`SPACING=0.5`、AD，直接调算子（不经 LAMMPS）；
数字由 `docs/check_all_spline_modes_slab.sh` 原样产出：

| 配置 | dE (eV) | max dF (eV/Å) | max dE/atom | rmsF | 相对 dF |
|---|---|---|---|---|---|
| B-spline-4 | +9.7e-04 | 1.43e-04 | 4.5e-05 | 5.709 | 5.5e-06 |
| B-spline-6 | +1.0e-03 | 1.39e-04 | 5.9e-05 | 5.728 | 5.4e-06 |
| CubeS₂-4 | −1.0e-04 | 1.14e-05 | 4.0e-06 | 5.752 | 4.8e-07 |
| CubeS₂-6 | +1.1e-05 | 7.21e-06 | 2.4e-06 | 5.733 | 2.8e-07 |
| QuadS-4 | +1.4e-05 | 8.64e-06 | 3.2e-06 | 5.734 | 3.3e-07 |
| QuadS-6 | +1.1e-05 | 1.04e-05 | 2.6e-06 | 5.731 | 4.2e-07 |

B-spline 的 ΔE/ΔVirial 比另外两家大两个量级（dVir −2.2e-2 vs −2e-4 eV）：
replicated 走 B-spline 的 `mesh_sum`、slab 走独立实现的 `midtown_sum_slab`，
这就是两个独立 fp32 实现的地板。**分布式网格对三种样条都是对的。**

### 6.2 MD 全链路（LAMMPS + canonical 融合 op，np=2，**主机版** `pair deepmd`）

`slab_vs_rep.np2.csv`（2026-11-10 18:0x，修复前）：

| 模型 | 力模式 | replicated | slab |
|---|---|---|---|
| `lr_bs4` | ad | ✓ 1219.8 ms / 11 610 MiB | ✓ 1299.7 ms / 10 596 MiB，`pe` 一致 |
| `lr_mid4ik` | ik | ✓ 1184.0 ms / 7 654 MiB | ✗（§4 的 guard，被 §3 掩盖）；§10 修复后回落 replicated |
| `lr_quads4ik` | ik | ✓ 1187.3 ms / 6 992 MiB | ✗（同上） |
| `lr_mid4` | ad | ✗ 526 848 原子（§5.1 显存）；≤98 304 正常 | 未测（同 §5） |

1 119 744 原子：bs4 replicated 2593.5 ms / 23 614 MiB → slab 2735.5 ms / 20 804 MiB，
`pe = -174649690.0` 两边一致；mid4ik replicated 2529.5 ms / 14 334 MiB，
quads4ik 2533.7 ms / 14 308 MiB。

> 这一节是**主机版 pair** 的数字。设备链 `pair_style deepmd/kk` 上同一体系快 23×，
> 而且分布式的代价从 +6% 变成 +102%——见 §11。

---

## 7. 该选哪个：论文的多 GPU 数据用 **replicated**

四条理由，按重要性排序：

1. **设备链上分布式直接慢一倍**（§11.3）：526 848 原子 np=2，`deepmd/kk` 的
   rep 77.8 ms/step vs slab 157.4 ms/step，只换来 10 768 → 9 738 MiB（−9.6%）显存。
2. **可覆盖性**：论文 §6.5 的 4 阶后端里两个 LR 后端是 ik，而 ik 没有分布式实现
   （§10 之后是静默回落 replicated，多设一个环境变量只会浪费一次警告）。
3. **收益本来就不大**：主机链上分布式只省 8.7%（526 848）/ 11.9%（1 119 744）显存，
   代价 +6.6% / +5.5% 时间——整步显存的大头是 descriptor 与边张量，不是网格。
4. **规模下限**：分布式网格每 rank 需要约 1 500 原子以上才划算——小尺寸下
   交换与同步开销压不住（maxsize 扫描里 `dpa4c-sog` 在 np=4、192 原子直接失败）。
   多 GPU 图的最小点正好在这个区间以下。

所以：**多 GPU 图取 replicated**。绝对数字用哪条链要统一：论文现在的 1 184 / 1 187 ms
（@526 848）和 2 530 / 2 534 ms（@1 119 744）来自**主机版** `pair deepmd`；
按文档主线（`deepmd/kk`）同一体系是 **99.6 ms**（bs4）/ **78.6 ms**（mid4ik）。
分布式网格最多作为一句"已实现并通过算子层等价性验证，但整步无收益、且与 ik 互斥"保留。

---

## 8. 修复清单（按优先级）

> **2026-11-10 19:50：第 1–5 条全部已实现并验证，见 §10。**

| # | 位置 | 事项 | 类型 |
|---|---|---|---|
| 1 | `source/api_cc/include/commonPT.h:988` `flatten_canonical_atom_virial` | undefined 时给出可读错误（指明"AOTI 吞掉了算子异常，见 stderr 的 `[E...]`"） | 诊断性 bug |
| 2 | `source/api_cc/src/DeepPotPTExpt.cc:1599 / 1992 / 2761` | 取到 `flat_outputs` 后统一校验每个槽位 `defined()` | 同上 |
| 3 | `sog_work/bench/bench_250k_lrpt.py:153` | `err = (p.stderr or "")[-300:]` 只留 stderr 尾部：`[E...]` 行在堆栈之前，正好被截掉。改为 grep 保留 `[E` / `ERROR` 行（本次就是靠改法才拿到真因） | 工具 |
| 4 | 文档 | 在 slab 相关文档里写明"ik × slab 被拒绝且分布式求解器无 i·k 通路" | 文档 |
| 5 | （可选）`latent_charge_sum.cu` | 网格预检查：`n_grid³` 超过某阈值时提前给出可读错误，而不是等分配失败 | 健壮性 |

第 1–2 条值得单独提 PR：它们把"任何算子异常"从误导信息变成可定位信息。

---

## 10. 修复与验证（2026-11-10 19:50）

### 10.1 改了什么

| 文件 | 改动 |
|---|---|
| `source/api_cc/include/commonPT.h` | 新增 `require_defined_outputs(map, where)`：列出所有未定义的槽位并指出真因在 stderr 的 `[E...]` 行；`flatten_canonical_atom_virial` 先在 undefined 时调用它，不再对 None 调 `reshape`。 |
| `source/api_cc/src/DeepPotPTExpt.cc` | 四处 `extract_outputs` 之后立即调用 `require_defined_outputs`（主 `compute`、`compute` 的第二条分支、电荷响应两条）。 |
| `source/op/pt/long_range/latent_charge_sum.cu` | **ik × 分布式不再直接中止**：`SOG_MIDTOWN_SLAB` / `SOG_MIDTOWN_DIST` 与 `force_mode != 0` 同时出现时打一次 `TORCH_WARN` 并**回落到 replicated**（力定律相同，只是每 rank 持全网格）；`SOG_SLAB_STRICT=1` 恢复硬报错。 |
| `source/op/pt/long_range/midtown_sum.cu` | replicated 网格在建表**之前**按 `n³×20 B` 与 `cudaMemGetInfo` 比对，不够就直接报可读错误——不再先花一分钟造几十 GB 的 host 绿函数再吃一个裸 `cudaMalloc` 失败。 |
| `sog_work/bench/bench_250k_lrpt.py` | stderr 保留 `[E` / `ERROR` / `out of memory` 行，不再只截尾部 300 字符。 |
| `sog_work/mdsuite/build_lrpt212_full.sh` | 修 `strings … \| grep -q` 在 `set -o pipefail` 下因 SIGPIPE 假报 schema 失败（改 `grep -c`）。 |

**为什么是回落而不是给 slab 补 ik 通路**：slab 的三次 `i·k` 逆变换意味着再走三遍 z 逆变换 +
两次 all-to-all，而 §7 已经说明分布式网格在整步里只省 8.7–11.9% 显存。补这条通路的代价与收益
不成比例；回落保住了"数值不变、不中止 MD"，代价只是显存。

### 10.2 验证

| 场景 | 修复前 | 修复后 |
|---|---|---|
| Python 探针 `SOG_MIDTOWN_SLAB=1 probe lr_mid4ik` | `OK nan`（rc=2） | `[W…] SOG_MIDTOWN_SLAB is set but force_mode != 0 (ik) …` + `energy = -8658.217698960`（与 replicated 的 `-8658.217696546` 同值） |
| LAMMPS `pair deepmd lr_mid4ik.pt2` + `SOG_MIDTOWN_SLAB=1` | `ERROR … Expected a proper Tensor … 'self'` | 正常跑完，`pe = -29946.558` |
| 同上，**不带** env | `pe = -29946.558` | 同值（逐位） |
| 同上，`SOG_SLAB_STRICT=1` | 旧的一行 | `ERROR … returned undefined output(s) 'atom_energy', 'atom_virial', 'energy', 'force', 'mask', 'virial'. A custom operator inside the graph threw and the AOTI proxy executor caught it …` |
| `lr_mid4` @ 526 848 原子 | 1 分钟后 `out of memory` → 误导信息 | **10 秒内**：`midtown: the replicated mesh needs 26.52 GiB … for a 1125x1125x1125 grid, but only 18.14 GiB is free. Use a coarser mesh …` |

回归：`test_grid_rule.py` + `test_long_range.py` 82 通过（lr212），
`test_mesh_backend.py` 20 通过（lr212）。

> **环境提醒**：仓库里的 `deepmd/lib/libdeepmd_op_pt.so` 现在对应 **lr212 / torch 2.12** 这条链
> （`build-lmp-mpi-212` 产物）。`lr` 环境（torch 2.11）要跑算子层脚本须换回
> `build-pt-mpi-gcc` 的产物。

---

## 9. 最小复现

### 9.1 看到 ik × slab 的真消息（10 秒）

```bash
cd /data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
/data/home/jiangzhen/.conda/envs/lr212/bin/python \
  docs/probe_mesh_operator_error.py lr_mid4ik            # energy finite, rc=0
SOG_MIDTOWN_SLAB=1 /data/home/jiangzhen/.conda/envs/lr212/bin/python \
  docs/probe_mesh_operator_error.py lr_mid4ik            # 打印真错误, rc=2
```

第二条的输出：

```
artifact : .../lr_mid4ik.pt2
env      : SOG_MIDTOWN_SLAB=1
[E1110 ... shim_common.cpp:1343] Exception in aoti_torch: latent_charge_sum:
    force_mode != 0 (ik) is implemented on the replicated midtown solve only;
    unset SOG_MIDTOWN_SLAB
Exception raised from latent_charge_sum at .../latent_charge_sum.cu:279
energy   : nan  -> NaN  (the operator threw and the shim swallowed it)
force max: nan
```

脚本用退出码表达结论：`0` 正常、`2` 拿到 NaN（算子异常被吞）、`1` 直接抛异常。

### 9.2 看到 mid4 大盒子的真错误（保留 stderr）

```bash
cd /data/home/public/jiangzhen/dp/sog_work/bench/matrix_slab_vs_rep
# 用小脚本调 bench_250k_lrpt，但 **stderr 落盘**，然后
grep -n "\[E" stderr.mid4_rep.526848.np1.txt
```

本次已经跑过并落盘（GPU2 空卡、np=1、4 步），两行即真因：

```
sog_gpu CUDA .../midtown/sog_gpu.cu:2696 out of memory
[E1110 19:34:23] Exception in aoti_torch: midtown: Form-A window symbol failed rc=-1
```

同一次运行里 LAMMPS 的 `log.mid4_rep.526848.np1` 只有那句
`Expected a proper Tensor`——对照两份文件就能看出 §3 的掩盖关系。

### 9.3 算子层 6 模式等价性（约 2 分钟）

```bash
bash docs/check_all_spline_modes_slab.sh     # 6 组：rep(np1) vs slab(np2)
```

三个脚本都在本目录：
`check_all_spline_modes_slab.sh`（驱动）→
`check_spline_parity_across_ranks.py`（每 rank 存 `par_<tag>_{e,f,i,v}<rank>.npy`）→
`compare_spline_parity.py`（按全局原子序重排后比较 E/F/virial）。
输出目录由 `OUT` 环境变量控制（默认 `/tmp/lr_md`）。

---

## 11. 设备链路复测：`pair_style deepmd/kk`（2026-11-10 20:2x）

**为什么重测**：§6.2 / §7 的数字（1.2 s/step @ 526 848）是用 **主机版** `pair_style deepmd`
量的，而本仓库文档的主线部署命令是 `pair_style deepmd/kk`（`PAIR_DEEPMD_KK_SOG.md` §4.1）。
本树以前没有 kk 插件产物，所以这两条链从没在这里比过。

### 11.1 先把本树自己的 kk 插件编出来

`source/lmp/plugin/CMakeLists.txt` 本来就有 `DEEPMD_LAMMPS_KOKKOS` 开关，只是没人编过：

```bash
module load gcc/13
export PATH=/usr/local/cuda-12.4/bin:$PATH NVCC_WRAPPER_DEFAULT_COMPILER=g++
R=/data/home/public/jiangzhen/dp/deepmd-kit-LR-lr-pt
OMPI=/data/app/openmpi-4.1.4/install/openmpi/4.1.4
cmake -S source/lmp/plugin -B /tmp/kk_lrpt_build \
  -DLAMMPS_SOURCE_ROOT=/data/home/public/jiangzhen/lammps/lammps_stable \
  -DCMAKE_CXX_COMPILER=/tmp/nvcc_wrapper_shim \
  -DDEEPMD_LAMMPS_KOKKOS=ON \
  -DKokkos_DIR=/data/home/public/jiangzhen/lammps/lammps_stable/build_kokkos/lib/kokkos \
  -DLIB_DEEPMD_CC=/data/home/public/jiangzhen/dp/lr212_install_lrpt/deepmd/lib/libdeepmd_cc.so \
  -DCMAKE_CXX_FLAGS="-fopenmp -pthread -I$R/source/api_cc/include" \
  -DCMAKE_MODULE_LINKER_FLAGS="-L$OMPI/lib -lmpi"
```

四个坑，每个都是**静默**的：

1. **CUDA 版本**：`nvcc` 11.8 不认 `-std=c++20`（LAMMPS 本身是用 CUDA 12.4 编的）。
   默认 PATH 上的 `nvcc` 是 11.8 → 必须 `PATH=/usr/local/cuda-12.4/bin:$PATH`。
2. **`-std=c++2a`**：CMake 对 NVIDIA 编译器发的是 `c++2a`，`nvcc_wrapper` 把它改成 `c++20`
   又被 nvcc 拒。用一个 shim 把两个别名改写成 `-std=c++20`。
3. **OpenMP**：Kokkos 的 OpenMP backend 要求编译器带 `-fopenmp`，否则
   `Kokkos_OpenMP_Instance.hpp:10` 直接 `#error`。
4. **MPI 符号被 C++ mangled**（最隐蔽）：CMake 把 `$(CXX_INCLUDES)` 放在 `$(CXX_FLAGS)`
   之前，`lammps_interface` 的 `src/STUBS/mpi.h` 先被命中，于是 MPI 声明没有 `extern "C"`，
   插件里留下 `_Z9MPI_BcastPviiii` 这样的符号，LAMMPS 加载时报
   `undefined symbol: _Z9MPI_BcastPviiii`、`Loaded 0 plugins`，然后只说
   `Unrecognized pair style 'deepmd/kk'`。**把 `-I$OMPI/include` 插到 argv 最前面**（shim 里做）。
   验收：`nm -D --undefined-only libdeepmd_lmp.so | grep _Z9MPI_` 为空。

产物：`/tmp/kk_lrpt_build/libdeepmd_lmp.so`（`LAMMPS_PLUGIN_PATH=/tmp/kk_lrpt_plugin`，
`dpplugin.so` 指向它）。输入用 `atom_style charge/kk` + `fix nve/kk` + `run_style verlet/kk`，
启动 `lmp -k on g 1`。

§11.2/§11.3 的表格由本目录的 `bench_kk_vs_host.py` 产出：

```bash
OUT=$PWD/matrix_kk python docs/bench_kk_vs_host.py 526848
```

它按 host/kk × rep/slab × np=1/2 跑同一批冻结产物，写 `kk_vs_host.csv`，
每行带 `pe`、峰显存和 `fallback_warn` 标记。

### 11.2 数值：与主机链逐位一致

| 场景 | `pair deepmd`（主机） | `pair deepmd/kk`（device） |
|---|---|---|
| 1 536 原子、60 步（跨两次 neighbor 重建） | `pe = -239576.69` | `pe = -239576.69` |
| 526 848 原子、6 步 `lr_bs4` | `pe = -82174044`，`T = 309.93727` | 同值，`T` 同值 |
| 526 848 原子、6 步 `lr_mid4ik` | （被杀，未测完） | `pe = -82174044` |

60 步那一行是关键：它跨过 `neigh_modify every 20` 的多次重建，末态能量仍逐位相同，
说明 kk 路径在邻居表更新时重建图、没有复用陈旧拓扑。

### 11.3 速度（526 848 原子，6 步，GPU 2/3 空闲）

| 模型 | pair | mesh | np | 总时长 (s) | ms/step | 峰值显存 (MiB) |
|---|---|---|---|---|---|---|
| `lr_bs4` | `deepmd` | rep | 1 | 13.593 | **2265** | 13 768 |
| `lr_bs4` | `deepmd/kk` | rep | 1 | 0.5976 | **99.6** | 11 980 |
| `lr_mid4ik` | `deepmd/kk` | rep | 1 | 0.4714 | **78.6** | 9 808 |
| `lr_bs4` | `deepmd/kk` | rep | 2 | 0.4668 | 77.8 | 10 768 |
| `lr_bs4` | `deepmd/kk` | slab | 2 | 0.9442 | **157.4** | 9 738 |
| `lr_mid4ik` | `deepmd/kk` | rep | 2 | 0.3361 | 56.0 | 6 804 |
| `lr_mid4ik` | `deepmd/kk` | slab | 2 | 0.3339 | 55.7 | 6 804 |

- **同一算例 kk 比主机快 23×**（13.593 → 0.5976 s），差异全在 `Pair` 段（99% 时间都在那里）：
  主机版每步把 526 k 个原子的坐标/力在 host↔device 间搬一遍并在 CPU 上重建边图，
  kk 版把坐标、电荷、力都留在 Kokkos 数组里。
- 1 536 原子 60 步：`0.484 s` → `0.0292 s`（**16.6×**），方向一致。
- **分布式在这条链上是 2× 的负担**：`slab np=2` 157.4 ms/step vs `rep np=2` 77.8 ms/step，
  换来 10 768 → 9 738 MiB（−9.6%）显存。主机链里那块 ~80 ms/step 的 z 转置 + all-to-all
  原本被 2.2 s 的 pair 淹没，kk 把 pair 压到 78 ms 之后就暴露出来了。
- `lr_mid4ik` + slab 走 §10 的回落：时长与显存和 `rep` **逐位相同**（333.9 vs 336.1 ms 是同一
  路径的重复测量），警告打在 stderr（`[W…] SOG_MIDTOWN_SLAB is set but force_mode != 0 (ik)`），
  不进 LAMMPS log——所以 `warn` 列看起来是 False 不代表没回落。

### 11.4 对 §7 结论的影响

结论不变、且更强：**论文的多 GPU 数据用 replicated**。
原因从"ik 只有 replicated 实现 + 收益小"变成"分布式在设备链上直接慢一倍"。
同时，论文 §6.5 的绝对时间目前是用主机版 `pair deepmd` 量的，
若投稿口径应按文档主线（`deepmd/kk`）重测，量级会小一个数量级（526 848 原子 99.6 ms/step）。

---

## 附：为什么这个误报值钱

它让"设计上的拒绝"、"显存不够"、"参数非法"三种完全不同的事故产生同一条消息，
而且只在 LAMMPS 侧；Python 侧连崩都不崩（给 nan）。
修 §3.6 那两处之后，同样的排查在第一条 stderr 行就能结束。
