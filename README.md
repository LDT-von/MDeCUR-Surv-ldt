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
| [comparison_morphid.py](comparison_morphid.py) | MorphID 的单折比较适配器（薄封装，非独立模型定义） |
| [comparison_camr_official.py](comparison_camr_official.py) | CAMR 上游生存契约臂：有界标量 Cox 头＋连续偏似然（薄封装，非独立模型定义） |
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
- `comparison_fold0.py`：full dense、同架构 aux-off、CAMR、**CAMR 官方生存契约**、**MorphID（含无交互消融）** 的共同单折计划；aux-off 是同一架构消融，MorphID 两臂是同一模型的交互项消融。
- `comparison_5fold.py`：四个多模态对照适配器的共享驱动；可按指定 fold 单独计划/执行。RNA-only、WSI-only 已移除。
- `comparison_merge.py`：把**项目臂**（`comparison_fold0.py`）与**上游对照臂**（`comparison_5fold.py`，同一 fold 0）合成一张排名表。它不信任任何 runner 报出的指标，而是从两边保存的生存矩阵**用同一套代码重算**所有指标，并核对两份 plan 描述的是同一批患者。`camr_2m` 在两个 runner 各跑一次，作为**桥接臂**量化跨 runner 偏移。
- `scripts/fetch_upstream_comparators.sh`：在服务器上按固定 commit 拉取 `papers/comparators/` 下的四个上游仓库并校验逐文件 SHA256（这些第三方树**不在本仓库内**）。
- `uni_views/`：冻结 UNI 特征提取和缓存接口。
- `diagnose_parity_*.py`：已有检查点的输入控制和交互诊断；不训练独立的 RNA/WSI 基线。

## MorphID 的单折接入（2026-11-16）

`comparison_morphid.py` 是 MorphID 接入锁定单折协议的薄适配器，**不改动**已有臂的逻辑、
协议常量与指标口径。它存在的原因是 MorphID 属于冻结式设计：形态基底与患者轴交叉拟合探针
在**训练折上一次性拟合**（不接受任何生存标签），只有离散生存头参与梯度。

因此它与其它臂有两点必须如实记录的差异：

1. **每轮 patch 重采样对它无效**。其它臂每轮 `resample()` 会改变所见 patch，而 MorphID 的头
   训练读取的是**缓存的患者级 OOF 特征**，训练期间固定不变。这是设计使然，已写进报告元数据
   （`per_epoch_patch_resampling_affects_this_arm: false`）。
2. **形态基底不是完全嵌套的交叉验证**。原型在训练折拟合一次；探针在训练折内部再做患者级
   交叉拟合。验证/测试患者从不进入原型、标准化、探针折或头。

其它臂用逐 epoch 前向反传，MorphID 用缓存特征前向反传，但**批次划分、epoch 顺序、
合并尾批、optimizer 步数与抽样哈希完全一致**，summary 的调度与步数守卫可以直接横向比较。
簇数 K=4、latent_dim=8、探针折数=4 是**事先固定的常量**，不用验证集标签选择；
低 R² **不作为**"独有"开关；CLR 组成自由度为 K−1。

合成数据契约测试见 `test_comparison_morphid.py`（8 项）。本机**没有**患者数据、
**没有** CUDA 运行、**没有**患者分数；这些只证明接入正确，不证明患者效果。

旧的 `model.py` 共享损失已迁入 `decur_losses.py`；旧 `pipeline.py`、`synthetic_study.py` 和原型专用测试已删除，仍在使用的损失数学回归保留在 `test_decur_losses.py`。

## CAMR 的两条臂：协议适配 vs 上游生存契约（2026-11-16）

`comparison_fold0.py` 现在有两条 CAMR 臂，**除了生存头和生存损失之外完全相同**
（编码器、双线性跨模态门控、低秩融合、重建头、判别器、attention MIL 池化都一样）：

| 臂 | 生存头 | 训练损失 | 定位 |
|---|---|---|---|
| `camr_2m` | 4 箱离散 hazard logits | 逐患者离散 NLL `patientwise_clamped_discrete_v1` | **协议适配**：和其它臂同一似然、同一指标口径 |
| `camr_2m_official` | 上游 `Sequential(Linear(15,1), Sigmoid())`，缩放到 `[-4, 4]` | **连续 Cox 偏似然**，风险集按**时间**定义 | **上游生存契约**：还原 CAMR 官方的输出与损失形式 |

`camr_2m_official` 实现于 `comparison_camr_official.py`，做了三件必须记录的事：

1. **不复用上游的 `utils.R_set`**。上游 `CoxLoss2` 调用 `R_set`，它构造的是
   `torch.tril(torch.ones(n, n))`——按**行号**的下三角矩阵，不是按生存时间的风险集。
   本臂改用上游 `CoxLoss1` 的按时间风险集 `R[i,j] = t_j >= t_i`。这一条是正确性问题，
   不是风格选择：按行号的风险集会让风险集取决于 batch 内患者的任意排列。
   `test_comparison_camr_official.py` 用"打乱 batch 后损失不变"来锁定这一点。
2. **评估时把标量风险投影回协议分箱**。共享评估代码需要 `[患者, 箱]` 的生存矩阵，
   所以用**训练折**上的 Breslow 基线累积风险做比例风险投影：
   `S_x(t) = exp(-H0(t) * exp(risk_x))`。该投影对风险单调，所以排序指标不变，
   只是补上了标定后的概率。两处偏差写在元数据里：协议最后一箱在离散臂里是开区间，
   这里用**训练折最大观测时间**锚定；基线只用训练折结局。
3. **batch 风险集**。Cox 偏似然在每个 batch 内计算，所以风险集只含 batch 内患者。
   这是分批 Cox 的固有性质，已写进 `settings.camr_official_risk_set`。

**必须记住的可比性边界**：两条 CAMR 臂**不共享训练目标**。`summary` 会在
`arm_comparability_note` 和 `training_loss_versions` 里显式写出每条臂的训练似然。
`camr_2m_official` 的 NLL/Brier/IBS 来自 Cox 风险的 Breslow 投影，因此那些列衡量的是
"每个模型在共享指标下的标定"，**不是**各自训练目标的值。

仍然存在的、无法靠这条臂消除的偏差：BLCA 没有 CNA 输入（第三槽用常量占位）、
4096 patch 预算要用 attention MIL 汇总成上游的 80 维患者向量（上游不做这一步）、
优化器与调度用协议统一版本（上游是 Adam 4e-4 + 线性衰减）、
判别器在本工程有独立优化器（上游建了但从不 step）。因此
**任何 CAMR 臂的绝对分数都不能与 CAMR 论文的 0.650 直接对比**。

合成数据契约测试见 `test_comparison_camr_official.py`（15 项）。本机**没有**患者数据、
**没有** CUDA 运行、**没有**患者分数。

## 运行与证据边界

单折路径与命令见 [PATIENTWISE_V3_HANDOFF_20261116.md](PATIENTWISE_V3_HANDOFF_20261116.md)。
四种对照的接口说明见 [COMPARATOR_5FOLD_PROTOCOL_20261116.md](COMPARATOR_5FOLD_PROTOCOL_20261116.md)。

**RNA 矩阵的来源与已知缺陷（2026-11-17 只读审计）**：
`results/rna_matrix_provenance_audit_20261117.json` 与
[RNA_MATRIX_PROVENANCE_AUDIT_20261117.md](RNA_MATRIX_PROVENANCE_AUDIT_20261117.md)。
要点：BLCA 的 RNA-only C≈0.747 是**真实的预后信号**（去掉全部插补基因仍有 0.7456，
远高于标签置换空对照 0.5951），不是预处理残留、也不是结局衍生列；但上游 z-score
拟合在**整张矩阵**上（含验证折与 27 个无临床行样本），存在**输入侧转导泄漏**，
且 27.1% 的基因是常数插补的。后续应重做 RNA 预处理（训练折内 z-score + 仅观测基因）。
行/列语义与批次结构由 `scripts/audit_rna_matrix_semantics.py` 单独检查（两个脚本并存）。

清理改变了源码清单和模块路径，因此需要重新生成 plan；旧 plan 的源码哈希守卫会拒绝直接复用。
离散目标仍为 `patientwise_clamped_discrete_v1`，模型计算和 15 项默认辅助系数未调整。
旧广播漏洞训练出的权重不能靠重评分修复；验证选轮分数不属于独立测试。
token 交互不等于严格 PID synergy，bulk RNA 与 patch 的软匹配不是实测空间表达对应。

本次只是本地文件整理与接口检查，没有启动患者训练、服务器任务或五折运行。
更完整的目录索引、删除项目和恢复备份位置见父目录 [WORKSPACE_LAYOUT.md](../WORKSPACE_LAYOUT.md)。
