# MorphID-Surv：冻结形态表示与非线性预测残差（2026-11-16）

> 当前版本：`morphid_frozen_conditional_residual_v1`。
> 用户已明确选择：**非线性核岭＋预测残差**，以及**先冻结无标签形态与探针表示验证机制**。
> 已有独立模型、探针、测试和固定合成审计；只有本地 CPU 生成数据证据。
> 未连接服务器、未读取患者数据、未执行患者训练/CUDA/五折。本批改动已随目录精简一并提交并推送到 origin/main。

## 1. 方法边界

本版本新增一个独立候选模型 `MorphIDSurvival`，保留原有 DensePathway 主方法和四种外部对照。
它不是将原来的11＋4项相关损失换名：原 Dense 的默认辅助系数不变；新模型当前阶段的形态与探针全冻结，只有生存头接收生存标签梯度。

“共同/独有”改为更有限、可验证的表述：
- `c = f_hat(RNA)`：在给定冻结表示与探针族下，RNA 可预测的病理特征。
- `u = p - f_hat(RNA)`：病理预测残差。
- 低 held-out R² 不证明独立或生物学独有；不显著的置换检验也不作“独有”开关。
- 不作 PID synergy、临床有效性或病理学家已验证形态解释的声明。

原始未修正方案已另存于工作区外的 `morphid_original_protocol_20261116.md`，不再作为算法定义。

## 2. 相对原方案必须修正的数学问题

1. 线性探针不能识别所有非线性依赖。h服从标准正态、p=h²−1时，相关为0、最佳线性R²也为0，但p完全由h决定。合成S1已改为检查非线性预测，而非要求线性探针完成不可能的识别。
2. 原投影公式维度不符；即使改成输出空间投影WW⁺，p=h+ε时也会把ε一起删除。本实现不用硬方向投影，直接减去预测值。
3. R²在未参与拟合的患者轴计算，负值保留，常量目标返回undefined/None。不能在一个患者内部把patch当独立探针样本。
4. 岭回归没有一般性的“n必须大于d×H”必要条件。记录患者数、设计秩、有效自由度和留出表现，不能用参数乘积伪造安全界限。
5. 如果以后原型/编码器接收生存梯度，必须称为任务监督塑造；当前冻结阶段不这样做。输入特征是否曾用生存标签预训练仍须核对来源；仅凭张量已冻结不能证明其历史无标签。

## 3. 冻结的形态表示

实现：`FrozenMorphologyBank`，位于 `cancer_morphid_model.py`。

输入是已冻结的patch特征。可接张量、(张量,有效mask)或reference-feature流式bag。
提供两种构建方式：
- 调用方提供固定中心及可选固定线性变换。
- `FrozenMorphologyBank.fit` 在无标签训练bag上做有限预算、按患者平衡采样的标准化、PCA和确定性k-means；不接受生存标签。

这是**特征空间原型后端**，不是HPL的Leiden/KNN实现，不声称复现论文HPC。
不直接沿用结肠癌研究的γ=1.5或47个簇；合成测试预先固定K=2。
本阶段不使用从旧生存模型学来的attention，采用无参数的软簇内均值：

q_jk = softmax_k(-||z_j-e_k||² / temperature)
p_ik = sum_j(q_ijk z_ij) / sum_j(q_ijk)
pi_ik = sum_j(q_ijk) / number_of_real_patches_i

- 距离前为负号，不能偏向最远原型。
- 先去除padding，再计算表示；NaN padding不会参与乘法或分母。
- 组成按真实patch软计数，不按attention权重。
- 低于预定mass cutoff的患者—簇表示不可估计，不以零目标拟合探针。
- 组成先做乘法零替换，再做CLR，保持非零比例和闭合性质；CLR自由度为K−1，不能把各系数解释为独立的绝对组织效应。
- 已有mask、chunk大小、patch顺序、整袋重复和全空bag回归。

## 4. 患者级交叉拟合的非线性探针

实现：`FrozenRBFProbe`、`CrossFittedRBFProbe`，位于 `morphid_probe.py`。

每簇以有效患者的RNA固定表示x预测冻结病理池化p。RBF核：

K_ab = exp(-||T(x_a)-T(x_b)||² / (2 bandwidth²))
A = (K + n alpha I)^(-1) (P - training_target_mean)
f_hat(x_new) = K_new A + training_target_mean

- 使用float64闭式Cholesky求解，没有独立探针优化器。
- 每个内折各自在其拟合患者上拟合标准化、可选PCA、核带宽及目标均值。
- 默认alpha=1e-3；未按验证患者或生存标签搜索超参数。
- 中位数带宽只由该探针拟合患者的距离得到。
- 患者ID须唯一、非空，无首尾空白；分组由ID和种子确定，行排列不改变归属。
- 生存头训练使用每位患者未参与拟合的OOF预测。
- 新患者推理使用固定折模型的平均预测。它与训练OOF单折估计存在有限样本分布差异，未宣称二者严格同分布。
- 某簇有效患者不足或缺少必要折覆盖时，整簇标记不可估计并记录；所有簇都不可估计则报错。
- 不把缺失簇的零填充值当真实重建目标；不使用R²硬筛“独有簇”。

### 冻结基底与嵌套边界

探针OOF是**给定冻结形态基底**的患者交叉拟合。
若形态PCA/原型在整个outer-train上拟合一次，不能把它说成整个形态流水线完全嵌套的交叉验证。
本地模型联测使用独立无标签形态参考池，探针/生存患者和fresh患者均未进入该参考池。
未来患者使用须明确记录形态拟合患者；验证/测试患者不得参与原型、标准化或探针拟合。

## 5. 生存预测路径与唯一可训练部分

路径为：

冻结patch特征 → 固定原型 → p与CLR组成
冻结RNA特征 → 交叉拟合核岭 → c
u = p - c
[RNA, c, CLR, 有效u] → 生存头 → 离散hazard logits → 生存曲线

保留c作为**RNA派生的非线性特征块**：统计冗余不代表有限容量RNA头一定能重建这部分预后信号。例如p=h²，单纯删除p后只留线性h头会丢失表达能力。
c不接受生存梯度，也不是额外的RNA生存基线。

生存头有加性RNA、c、组成和残差块，以及显式的残差×RNA乘积项。
- 乘积路径使用按训练尺度缩放、但不减残差均值的u，确保原始u为零或簇缺失时乘积为零。
- 缺失簇在标准化后再次掩码，防止中心化制造虚假的非零残差。
- 相乘不等于已识别纯交互/PID；u本身是以RNA条件化的估计量。
- 训练使用共享 `patientwise_clamped_discrete_v1`，censorship=1表示删失。
- 训练优化器只接收head参数；PCA、原型、预处理、探针为冻结buffers。
- 每次head训练前后核对冻结状态指纹。
- 不用标签选R²阈值或原型；本地训练不做验证峰值选轮。
- head标签顺序必须与OOF缓存患者ID顺序一致。

## 6. 可运行API

~~~python
from cancer_morphid_model import FrozenMorphologyBank, MorphIDSurvival

# 仅无标签训练/参考特征；不能传入验证/测试患者。
bank = FrozenMorphologyBank.fit(
    morphology_reference_bags, ids=morphology_reference_ids,
    clusters=4, latent_dim=8,
)
model = MorphIDSurvival(bank, rna_dim=train_rna.shape[1], bins=4)
model.fit_frozen(train_bags, train_rna, ids=train_ids, folds=4)

# label为训练事件拟合分箱得到的标签，censorship=1-event。
receipt = model.fit_survival_head(
    train_bin_labels, train_censorship, ids=train_ids,
    epochs=30, lr=5e-4, batch_size=20,
)
prediction = model(new_bags, new_rna, ids=new_patient_ids)
checkpoint = model.checkpoint()
reloaded = MorphIDSurvival.from_checkpoint(checkpoint)
~~~

推理ID若与探针/形态拟合ID重叠则拒绝；训练诊断须用 `training_oof_logits()`。
checkpoint保存冻结bank、探针折模型、预处理、head和版本，不携带OOF训练特征缓存；部署权重不能直接恢复head训练。
冻结拟合若在探针阶段失败，拒绝在同一个对象上重试，以免混入部分旧簇状态。

本轮没有患者CLI、数据清单生成器或服务器任务；该API先服务本地机制验证。

## 7. 已执行证据

### 固定探针反例：四项全部通过

回执：`results/morphid_probe_mechanism_final_20261116.json`。

| 场景 | 固定生成数据结果 | 支持的结论 |
|---|---|---|
| S1 非线性均值关系 | 线性fresh R² -0.000141；核岭OOF 0.995708、fresh 0.996392 | 该探针能处理这个指定非线性反例 |
| S2 p=h+ε，重叠方向 | 残差与真实ε相关0.982876；新残差方差0.510608，硬投影为0 | 不再因整个方向投影而删除ε |
| S3 高维独立噪声 | in-sample约1；OOF -0.681208、fresh -0.289925 | 训练拟合优度不当作留出可预测性 |
| S4 条件方差依赖 | 均值预测fresh R²约0；残差²与RNA²相关0.428507 | 均值残差仍可能依赖RNA，不能判独立 |

S4训练为80个反向配对块/160行构造数据，不作训练行iid推断；fresh为4096个独立生成样本。
所有gate在运行前固定，未为通过而调参；源码修正后的复跑与初跑指标相同。

### 核心联测

新增核心9项、探针10项，与已有逐患者NLL6项合并运行，**25 passed**。
覆盖：手算求解、OOF自身目标隔离、内折预处理隔离、负/未定义R²、mask/padding、低mass簇、RNA派生非线性风险表达、head-only梯度、冻结指纹、失败重用拒绝、ID隔离、精确checkpoint预测重放。

完整验证回执：`results/morphid_frozen_validation_20261116.json`，附同名log/XML；25项用时4.40秒、无失败或跳过。回执冻结本轮源码SHA-256，并确认原有7个模型定义文件与清理前备份逐字节一致。

这些是算法/工程的生成数据证据，不证明患者收益、真实形态簇语义或方法新颖性。

## 8. 后续顺序

1. 固定当前冻结机制接口与反例，保留原Dense作对照。
2. 真正患者运行前，单独核实冻结特征来源、形态参考拟合ID、RNA表示和队列划分。
3. 获得具体执行授权后才接入BLCA fold0；不自动启动五折、outer_test或服务器训练。
4. 可微原型联合训练属于另一个明确变体，不能与当前无标签冻结分解混称。
5. 无本队列对应tile图像与映射前，不声称病理学家可辨认；NMI/ARI也不能替代人工形态评估。
