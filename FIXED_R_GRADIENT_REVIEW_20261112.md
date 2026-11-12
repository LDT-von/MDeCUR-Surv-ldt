# 固定R训练批次梯度审计：审查与更正（2026-11-12）

本记录依据用户2026-11-12转发的服务器回报（提交`b8ba2199da270f538ec179ec20c5d4b8868980ac`，审计输出SHA256 `186071bc9617f22f681e3f6294f7b1a1d6c9954f8d404b6ed99940934e575812`）与本仓库实现核对写成。本机未连接服务器，未读取服务器原始JSON、checkpoint或患者文件；下列服务器数字均为**转发证据**，不是本机独立复算结果。

审计协议本身见`FIXED_R_GRADIENT_AUDIT_20261112.md`，实现见`pathway_gradient_audit.py`、`diagnose_fixed_r_gradients.py`。本轮未修改训练实现、未重跑整套检查、未连接服务器、未展开五折、未读取outer_test。

## 1. 审计本身的可信度

可以采信，理由是守卫链完整且互相独立：

- 梯度重构最大绝对L2误差4.461e−08、相对6.911e−08，远低于实现容差`1e-6 + 1e-5·|∇total|`（`pathway_gradient_audit.py`）。
- 计划阶段钉住批次顺序、视图种子、输入指纹与源码哈希；审计阶段重新计算并要求完全相等。
- 模型状态哈希前后相同（`66cb07a5…a49c6`）、全部参数`.grad`仍为`None`、CPU/CUDA随机流未变、训练输入指纹未变。
- `training_performed=false`、`optimizer_steps=0`、`validation_loaded=false`、`test_loaded=false`。
- 11项辅助损失6/6记录全部`status=computed`，无跳过。

历史指纹边界按原样保留：旧checkpoint只有训练/验证**合并**内容哈希，本次**未**独立复算该历史合并哈希；本次在计划阶段**新建**训练集单独内容指纹并要求计划与审计严格一致（`training_input_sha256=ae86c280…b32098`）。不能表述为"已独立核对历史训练集单独哈希"。

## 2. 更正一：转发汇总中"辅助总和 vs 总下降方向"的标签是错的

转发第4节写道"辅助总和 vs 总下降方向：余弦0.9963~0.9998，全部aligned"。这与实现不符。

`pathway_gradient_audit.py`中两个字段分别是：

- `sum_weighted_auxiliary.survival_graph_space = relation(∇survival, Σw_i·∇aux_i)`：**辅助总和 vs 生存梯度**。
- `total_descent_direction = relation(∇survival, ∇total)`：**生存梯度 vs 总梯度**。

转发报出的"余弦0.9963~0.9998、ratio 1.0003~1.0044"符合后者：\(|\nabla total|/|\nabla L_{surv}|\) 必然≈1，因为辅助项只占总梯度约2.6%。若真是"辅助 vs 总下降方向"，该比值应≈0.026而非1.0003。

**因此"全部aligned"证明的是"加上辅助项后总梯度方向几乎不变"，不是"辅助项与下降方向一致"。该表述必须撤回。**

真正需要的量是 `sum_weighted_auxiliary.survival_graph_space.cosine` 与 `.direction`，转发汇总中**没有给出**。从单项余弦全部落在 ±0.0227 内判断，辅助总和很可能与生存梯度接近正交（对应 aggregate 余弦约 −0.026，即轻微冲突），但这是推断，必须以服务器JSON原值为准。

新增只读提取程序`extract_fixed_r_gradient_summary.py`，在不加载任何模型、manifest、患者文件的前提下读既有`fixed_r_gradients.json`，单独报告：

- `auxiliary_sum_versus_survival_gradient`：正确的辅助总和关系，含逐记录余弦、方向计数、ratio与逐模块绝对范数。
- `survival_gradient_versus_total_gradient`：修正标签后的下降方向关系。
- 逐项/逐模块的**绝对**范数（`survival_norm`、`weighted_auxiliary_norm_in_survival_space`）与相对ratio并列。
- 记录数与**独立批次数**（两条视图种子给出的梯度近乎逐位相同，不构成独立重复）。
- 梯度重构误差汇总。

## 3. 更正二：转发汇总第3点的怀疑方向需要反转

转发第3点问："pathway_tokenizer与matching_path/matching_rna在4项全局项上梯度为零——这4项的'匹配'部分可能实际未参与计算。"

逐项追连线后，这4项按设计就不经过条件匹配分支：

| 项 | 使用的分支 | 是否经过 `match_path`/`match_rna`/tokenizer |
|---|---|---|
| `global_path_within` | `base.path_projector(attention池化)` | 否 |
| `global_common` | 同上，global_p1 vs global_r1 | 否 |
| `global_unique` | 同上，独有维切片 | 否 |
| `pathway_path_within` | 独立锚点 `pathway_queries`（前向不读患者RNA） | 否 |

`cancer_pathway_stream.py:_feature_scores`中，全局attention分数是第1列，独立锚点是第2列，条件匹配是第3列；全局项只用前两列，锚点项只用第2列，`matched_*`与`pathway_common/unique/rna_within`才用第3列与tokenizer。

**结论：零梯度是隔离性验证通过，不是缺陷，也不是"匹配部分未参与计算"。** 真正驱动tokenizer与`match_*`的是`pathway_common`、`pathway_unique`、`pathway_rna_within`、`matched_common`、`matched_unique`五项，转发表中也确实是这五项。

## 4. 对转发三点审查的逐条结论

### 4.1 预测头与交互分支无辅助梯度：事实成立，但"不承接任何辅助信号"过强

11项辅助全部定义在**投影空间**（`base.path_projector/rna_projector/local_projector`、`model.local_path_projector/local_rna_projector`），而预测路径在`prediction_space='encoder'`下直接消费pooled表示、RNA编码与通路token。两条路径只在编码器、`base.attention`、`match_path/match_rna`、通路tokenizer这几个**共享参数**上交汇。`base.head`、`interaction_head/path/rna`在辅助反向图中不存在，因此`no_auxiliary_dependency_in_survival_space`必然66/66，属结构必然，不是bug。

需要修正的推论是：交互分支的**参数**未被辅助直接约束，不等于交互分支**不受辅助影响**。`interaction_path/interaction_rna`消费`matched_path`（经`match_path`与条件softmax汇聚得到）与`rna_tokens`（经tokenizer得到），而`matched_common/matched_unique`确实对`match_path`、`match_rna`、tokenizer产生了梯度。因此准确表述为：

> 辅助项不直接约束交互头的参数，但通过`match_path/match_rna`与通路tokenizer**间接塑造交互分支的输入**。

另需与上一轮query置换诊断分开：query置换不敏感来自条件匹配本身的有效信息量小；本审计说明辅助项不碰预测头参数。两者不是同一结论，不能互相解释。

### 4.2 patch_within 6/6全冲突：保留意见正确，本批不足以支持调参

转发的保留意见（负内积不等于AdamW实际更新方向）成立。补充三点：

1. AdamW是逐坐标归一化更新，冲突判断应在逐坐标层面做；模块级L2范数比既可能低估也可能高估更新级冲突。
2. `patch_encoder`上patch_within的ratio中位仅7.55e−04，即一阶冲突总量约为生存梯度的0.075%。23轮训练不足以造成可观损害。本批**不支持**"patch_within有害"。
3. 方向性冲突可预期：patch_within最大化A/B局部嵌入相关，生存头需要可分的汇聚表示，二者在共享低秩方向上天然对立。**冲突≠有害**。

`patient_attention`上`global_common`余弦−0.942（6/6）易被误读：该处生存梯度范数仅4.8e−04，比总体0.457~0.757小约三个数量级，属**分母放大**而非强冲突，正式报告中应与分母并列呈现。

若后续要选单因素候选，patch_within是唯一"6/6一致冲突"的项，但同时是相对量级最小项之一（ratio 7.55e−04）。现有`paired_pathway_validation.py`只支持`matched-off`、`uniform-common`、`interaction-off`三种比较，**没有aux-off或patch-within-off**，如需该消融须先扩展并新训两臂。

### 4.3 4项全局项零梯度：见第3节，方向反转。

## 5. 本批最有信息量的一个数

`matching_rna`上`matched_common`的`weighted_norm_ratio_to_survival`中位**0.795**（全场最高），余弦−0.157、4/6冲突。

含义是：决定RNA条件匹配结果的参数，其梯度约80%来自辅助项`matched_common`，仅约20%来自生存目标，且方向经常相反。这与已归档的`matched-off`消融（ΔNLL +0.0026958，95%区间[−0.002018,+0.008033]含0）并列看是自洽的：辅助项确实在大幅改动匹配参数，但最终验证指标几乎不动。两件事同时成立，说明被改动的参数方向对生存预测无实质贡献。

该结论必须补读绝对量才能定论：需要`records[*].audit.terms.matched_common.modules.matching_rna.survival_graph_space.{survival_norm, weighted_auxiliary_norm_in_survival_space}`。只有拿到绝对范数才能判断0.795是"两范数都小"还是"两范数都不小"。这是本轮新增只读提取程序要输出的内容之一。

## 6. 统计口径提醒

- 6条记录的有效样本量是**3批**，不是6。每批两组视图种子给出的梯度几乎逐位相同（转发已报repeat间差异<1e−04），属近似重复测量。报告应写"3批（每批2个近似重复视图）"，否则"6/6一致"会被读成6个独立证据。
- `norm_of_sum_over_sum_of_norms`仅0.438~0.687，说明11项之间存在明显相互抵消。因此"哪一项最重要"**不能**由单项范数排序直接读出，单项范数低不等于该项可删。
- 最佳检查点的局部梯度不能代表整条训练历史。

## 7. 本轮结论边界

本批证据支持的是结构性与几何性陈述：辅助项的作用位置、各模块连接状态、量级与方向、以及"辅助项不直接约束预测头参数但间接塑造其输入"。**不支持**任何"某辅助项有害/可删/应加大系数"的结论，也不支持患者层面的泛化或临床结论。未展开五折，未读取outer_test，未训练。

后续若需要机制候选，须先在同一新版本上完成同预算消融并用配对bootstrap给出区间；当前阶段的任何单因素改动都只是研究候选，不是已验证增量。
