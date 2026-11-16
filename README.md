# MDeCUR-Surv：病理 WSI＋RNA 生存预测

**新增候选方法 MorphID-Surv（2026-11-16）**：用户已选定冻结无标签形态表示与非线性核岭预测残差。独立实现为 `cancer_morphid_model.py`，探针工具为 `morphid_probe.py`。它不改变原Dense默认入口，尚无患者结果；定义、API和固定反例见 [MORPHID_SURVIVAL_PROTOCOL_20261116.md](MORPHID_SURVIVAL_PROTOCOL_20261116.md)。当前模型定义文件共8个：原框架3个依赖模块＋MorphID1个＋外部适配器4个。

当前主方法是完整 dense-token 框架，默认保留原 11 项辅助目标和 4 项 dense 辅助目标。
主训练入口是 [train_pathway_parity.py](train_pathway_parity.py)，单折比较入口是 [comparison_fold0.py](comparison_fold0.py)。

## 现在保留的模型文件

原Dense与外部对照保留 **7 个模型定义文件**：原框架3个依赖模块＋4个对照适配器；本次新增MorphID一个主模型文件，共8个。
原框架对应1个方法，MorphID为新增的1个冻结机制候选；另有4个外部对照。底层模块不应当作三个独立方法重复计数。

| 文件 | 作用 |
|---|---|
| [cancer_morphid_model.py](cancer_morphid_model.py) | 新增 MorphIDSurvival：冻结形态、交叉拟合预测残差与独立生存头 |
| [cancer_dense_model.py](cancer_dense_model.py) | 当前完整主模型 DensePathwaySurvival，包含 token 预测分支和 dense 目标 |
| [cancer_pathway.py](cancer_pathway.py) | 主模型使用的通路编码、局部匹配和融合基础层 |
| [cancer_model.py](cancer_model.py) | 上层实际复用的病理 MIL、RNA 编码和生存头基础层 |
| [comparison_camr.py](comparison_camr.py) | CAMR 双模态适配 |
| [comparison_pibd.py](comparison_pibd.py) | PIBD 双模态适配 |
| [comparison_deref.py](comparison_deref.py) | DeReF 双模态适配 |
| [comparison_dimaf.py](comparison_dimaf.py) | DIMAF 双模态适配 |

[decur_losses.py](decur_losses.py) 只存共享辅助损失，不定义独立生存模型。
RNA-only、WSI-only、早期 masked-mean 原型及其训练流水线已按用户要求删除。
当前主模型保留的编码器组件、辅助损失、checkpoint 冻结输入诊断属于完整框架，不是另立的单模态基线。

## 下载的官方对照源码

以本仓库目录为起点，源码统一在父目录：
- `../papers/comparators/CAMR/`
- `../papers/comparators/PIBD/`
- `../papers/comparators/DeReF/`
- `../papers/comparators/DIMAF/`

路径与固定文件哈希见 [comparison_sources.json](comparison_sources.json)。
四种方法的原仓库保持原路径，适配器仍使用这些上游源码。上游 checkout 附带的数据类文件不等于本工程的训练输入；本次清理没有加载其内容，也没有将整个上游目录加入本仓库发布。
MurreNet/MoHD 没有可验证的本地实现，不列为已下载模型。

## 当前可用入口

- `train_pathway_parity.py`：完整主模型默认入口。
- `comparison_fold0.py`：full dense、同架构 aux-off、CAMR 的共同单折计划；aux-off 是同一架构消融。
- `comparison_5fold.py`：四个多模态对照适配器的共享驱动；可按指定 fold 单独计划/执行。RNA-only、WSI-only 已移除。
- `uni_views/`：冻结 UNI 特征提取和缓存接口。
- `diagnose_parity_*.py`：已有检查点的输入控制和交互诊断；不训练独立的 RNA/WSI 基线。

旧的 `model.py` 共享损失已迁入 `decur_losses.py`；旧 `pipeline.py`、`synthetic_study.py` 和原型专用测试已删除，仍在使用的损失数学回归保留在 `test_decur_losses.py`。

## 运行与证据边界

单折路径与命令见 [PATIENTWISE_V3_HANDOFF_20261116.md](PATIENTWISE_V3_HANDOFF_20261116.md)。
四种对照的接口说明见 [COMPARATOR_5FOLD_PROTOCOL_20261116.md](COMPARATOR_5FOLD_PROTOCOL_20261116.md)。

清理改变了源码清单和模块路径，因此需要重新生成 plan；旧 plan 的源码哈希守卫会拒绝直接复用。
离散目标仍为 `patientwise_clamped_discrete_v1`，模型计算和 15 项默认辅助系数未调整。
旧广播漏洞训练出的权重不能靠重评分修复；验证选轮分数不属于独立测试。
token 交互不等于严格 PID synergy，bulk RNA 与 patch 的软匹配不是实测空间表达对应。

本次只是本地文件整理与接口检查，没有启动患者训练、服务器任务或五折运行。
更完整的目录索引、删除项目和恢复备份位置见父目录 [WORKSPACE_LAYOUT.md](../WORKSPACE_LAYOUT.md)。
