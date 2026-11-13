# lr-pt 内部文档索引

> 这个分支（`docs-notes`）只承载设计笔记、基准台账和复现脚本；代码树在
> `main`，其顶层目录与上游 DeePMD-kit 一致（只有 `doc/`，没有 `docs/`）。
> 文中的 `/data/home/...` 是本机部署路径，供复现时对照，不是仓库的默认值。

## 总览与 PR

| 文档 | 内容 |
|---|---|
| [PR_SUBMISSION.md](PR_SUBMISSION.md) | PR 的拆分（A/B/C/D）、编译与测试命令、可直接粘贴的英文正文 |
| [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md) | 与上游同步的合并冲突地图（8 个文件、16 个 hunk）与建议解法 |
| [LONG_RANGE_PORT.md](LONG_RANGE_PORT.md) | 上游 `long_range` → pt backend 的接口映射与有意偏差 |
| [BUILD_AND_TEST.md](BUILD_AND_TEST.md) | 两棵树、两条链的全套编译、安装布局与测试流程 |

## 网格与误差

| 文档 | 内容 |
|---|---|
| [GRID_FROM_ACCURACY.md](GRID_FROM_ACCURACY.md) | 目标精度 → 网格的工作流（`\|E_k\|` 归一、能量相对误差） |
| [IK_MODE_ACCURACY.md](IK_MODE_ACCURACY.md) | CubeS₂ `force_mode="ik"` 的精度表与网格规则 |
| [AD_MODE_ACCURACY.md](AD_MODE_ACCURACY.md) | AD（权重微分）模式的精度表与网格规则 |
| [MESH_ERROR_MODEL_AND_FORM_A.md](MESH_ERROR_MODEL_AND_FORM_A.md) | 网格误差模型、Form A 精确影响函数与误差律推导 |
| [AD_EVERYWHERE_PLAN.md](AD_EVERYWHERE_PLAN.md) | 两条链统一到 AD 力的计划与结论 |
| [FORCE_DIFFERENTIATION_PLAN.md](FORCE_DIFFERENTIATION_PLAN.md) | 力的精度诊断与改造计划 |
| [MIDTOWN_AD_DESIGN.md](MIDTOWN_AD_DESIGN.md) | midtown spline 的 AD 力实现重构设计 |

## 性能与基准

| 文档 | 内容 |
|---|---|
| [MIDTOWN_VS_BSPLINE_BENCH.md](MIDTOWN_VS_BSPLINE_BENCH.md) | CubeS₂ vs B-spline 的等精度 Fourier 与整步速度 |
| [SPLINE_MATRIX_BENCH.md](SPLINE_MATRIX_BENCH.md) | 极端大体系下复制网格 vs 分布式网格的样条矩阵 |
| [MIDTOWN_PERF_PLAN.md](MIDTOWN_PERF_PLAN.md) | midtown 性能优化计划（含 2026-11-13 更正） |
| [MIDTOWN_PERF_REDESIGN.md](MIDTOWN_PERF_REDESIGN.md) | 让 CubeS₂ / QuadS 兑现“网格更少”优势的结构优化 |
| [MIDTOWN_ALL_CONFIGS.md](MIDTOWN_ALL_CONFIGS.md) | 4/6 阶 × AD/ik 的全配置复核（98k 与 331k） |
| [SLAB_DEPLOYMENT.md](SLAB_DEPLOYMENT.md) | z-slab 在 ≈50 万原子单卡极限体系上的部署结论 |

## 冻结与部署

| 文档 | 内容 |
|---|---|
| [FREEZE_SOP.md](FREEZE_SOP.md) | 生产冻结 SOP：`--compress` 的 compact canonical 下界 |
| [SEZM_CANONICAL_STATUS.md](SEZM_CANONICAL_STATUS.md) | SeZM canonical 部署的状态与多 rank 结论 |
| [SEZM_AOTI_MISCOMPILE.md](SEZM_AOTI_MISCOMPILE.md) | SeZM `.pt2` 的 AOTInductor 降级不保真问题 |
| [SLAB_IK_FAILURE_DIAGNOSIS.md](SLAB_IK_FAILURE_DIAGNOSIS.md) | slab × ik 崩溃的诊断、探针与修复 |

## LAMMPS

| 文档 | 内容 |
|---|---|
| [PAIR_DEEPMD_KK_SOG.md](PAIR_DEEPMD_KK_SOG.md) | `pair_style deepmd/kk` 如何驱动 SOG 长程模型（关键字、环境、失败模式） |

## 复现与脚本

- [MESH_ENGINE_TEST_REPRO.md](MESH_ENGINE_TEST_REPRO.md)：网格引擎的编译、冻结工件
  清单、四层测试与基准再生。
- `repro/`：`env.sh` 以及等价性对比、性能、显存账本、CubeS₂ 项验证脚本。
- `measure_*.py` / `check_*.py` / `bench_*.py` / `probe_*.py`：精度与性能测量、
  冻结一致性检查、网格算子误差探针。
- `bspline_ad_reference.cpp`：B-spline AD 的 C++ 参考实现。

## 相关仓库

- 代码（本 fork）：<https://github.com/Johnsonjz/deepmd-kit-lr-pt>
- 上游：<https://github.com/deepmodeling/deepmd-kit>
- 论文：<https://github.com/Johnsonjz/sog-lib-paper>
