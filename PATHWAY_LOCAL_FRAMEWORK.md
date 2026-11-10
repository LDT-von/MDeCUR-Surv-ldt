# 病理局部—RNA通路共同/独有生存框架（2026-11-10）

这是对现有`CancerDeCUR`生存模型的可选扩展，包含可执行训练、验证、检查点和重载。它使用真实通路基因集合，将病理patch和RNA通路构成软匹配，同时在局部和患者级施加DeCUR式相关约束。它借鉴DenseTwins的全局/局部目标及共同/独有思路，但不是原论文的数值或源码复现；病理patch与bulk RNA没有逐区域配对真值，匹配结果是训练所得的关联。现有普通融合入口保持独立，原实验不受影响。

本地完成CPU合成检查后才可称功能已接通；尚未得到5090真实患者训练结果。单次验证结果可检查训练是否工作，不能单独证明模块贡献或临床泛化。用户明确希望先跑完整模型，本文给出先小范围工程检查、再自动运行一次fold0完整训练的方案。

## 1. 输入及两种视图来源

统一使用已有患者manifest、RNA `.npy`、基因顺序文件和一份固定GMT：

- `--view-source feature_noise`：使用现有 `wsi_pt` 1024维单视图。预测使用原始特征；辅助视图在共享可训练patch编码器**之后**生成：`hA=h+sigma*RMS(h)*epsilonA`、`hB=h+sigma*RMS(h)*epsilonB`，RMS不参与梯度，A/B独立，保留每个patch和行顺序。`--feature-noise .05`是相对每个patch编码表示RMS的标准差。根据seed、患者、切片名、chunk行区间、epoch和batch生成可复现噪声；前向和checkpoint反向重算使用同一值。更改chunk大小会改变噪声抽样，需记录。**这是特征级扰动，不是WSI像素D4增强，不证明旋转/镜像不变性。**
- `--view-source image_d4`：加载严格验证的 `wsi_h5` reference、A、B，A/B由同一patch像素D4增强并经冻结UNI提取。当前5090转发报告没有此缓存；在其可用之前不能把feature_noise结果称为真实D4训练。

两种来源的RNA视图均为训练归一化后的RNA加独立高斯噪声，低方差屏蔽坐标保持0。验证/推理使用未扰动病理reference和RNA。RNA噪声是特征扰动假设，并未证明生物学合理性。

程序附带冻结的[Reactome人类顶层通路](resources/reactome_human_roots.gmt)：从官方GMT及层级关系按稳定ID提取29个顶层集合、与患者标签无关，源URL/校验值/选择规则见[资源记录](resources/reactome_human_roots.json)，Reactome数据为CC0。匹配严格按`genes.txt`的原有基因符号及顺序，不自动猜别名。一个基因可以属于多个通路；报告记录匹配数、覆盖率及被跳过通路。默认某通路至少有5个精确匹配基因；还要求每个保留通路至少有1个训练集RNA mask未屏蔽基因。输入文件实际基因顺序的正确性仍需由数据生成方核验，哈希不能自动证明。

## 2. 网络及对应关系

以默认`hidden=16, common=8, local_dim=8, local_common=4`为例：

1. 病理patch特征经共享`Linear+LayerNorm+GELU`得到16维，再用全局attention MIL得到患者病理16维；RNA原向量经原浅层编码器得到患者RNA 16维。二者是原生存预测主路径。
2. 每个RNA通路使用其显式基因子集，经该通路专属线性层/LayerNorm/GELU形成16维RNA通路token；通路可以共享基因但不把其当作新增患者。
3. **RNA条件软匹配**：对每个患者、每个通路，将RNA token映射为query，将病理patch表示映射为key；按归一化相似度及temperature对该患者的全部patch做softmax，得到该RNA通路条件下的病理区域向量。值向量使用投影前病理编码表示。A/B视图各自计算对应的软匹配。
4. **病理独立区域锚点**：另为每个通路保留一个跨患者共享的可学习病理query。它当前向计算不读取该患者RNA，对patch汇聚成通路索引的病理区域，再与同编号RNA通路token施加局部约束。该query仍通过跨模态训练被优化，不能据此宣称已证明统计独立。
5. 生存头使用完整患者病理/全局RNA表示，并加入经`GELU(Wp*v_match) ⊙ GELU(Wr*r_pathway)`逐通路交互再平均形成的小容量交互项：`log_rates=base_head([hP,hR])+interaction_strength*interaction_head(mean_k(interaction_k))`。默认strength=0.2。对应关系是预测的一部分，输出的相关权重并不是空间RNA真值或因果证据。
6. 病理区域和RNA通路各有局部线性辅助投影8维，前4维为共同、后4维为独有。局部匹配A/B两视图均对条件匹配区域与同通路RNA约束；独立锚点使用A视图对齐，A/B用于各自模态内一致性。

注意：患者RNA参与条件匹配，可能通过选择过程自我强化相关性。独立病理锚点和未来的患者配对打乱/匹配机制消融用于检查这一风险，不能消除统计偏差。Attention高权重不能直接解释为该patch具有相应RNA的实测表达。

## 3. 全部训练目标

`L_total=L_survival + 各系数×下列11个辅助目标`。所有相关矩阵按特征维计算，目标1表示对角1/非对角0，目标0表示全矩阵趋零；非对角惩罚系数0.005。路径级相关先在**患者维B**计算，再对通路取均值，不将B×K或patch数视为独立患者。保留原连续时间右删失NLL和训练集RNA预处理。

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--global-path-within` | 0.0005 | 患者级病理A/B模态内一致性/去冗余 |
| `--global-rna-within` | 0.0005 | 患者级RNA A/B模态内一致性/去冗余 |
| `--global-common` | 0.001 | 患者级病理/RNA共同8维相关趋单位阵 |
| `--global-unique` | 0.001 | 患者级独有8维跨模态相关趋零 |
| `--patch-within` | 0.001 | 同一病理patch A/B局部一致性，患者等权 |
| `--pathway-path-within` | 0.0005 | 病理独立区域A/B投影的一致性/去冗余 |
| `--pathway-rna-within` | 0.0005 | RNA通路A/B投影的一致性/去冗余 |
| `--pathway-common` | 0.001 | 病理独立区域与RNA同通路的共同4维对齐 |
| `--pathway-unique` | 0.001 | 病理独立区域与RNA同通路的独有4维去相关 |
| `--matched-common` | 0.0005 | RNA条件匹配区域与RNA token的共同4维对齐；A/B平均 |
| `--matched-unique` | 0.0005 | RNA条件匹配区域与RNA token的独有4维去相关；A/B平均 |

默认`--common-mode uniform`采用等权患者级共同相关；方差低时记录维数且保留带epsilon的目标。下方首轮完整模型命令**显式选择`stability`**，使用两视图的停止梯度稳定权重；`shuffled`用于将来检验权重对应关系。稳定模式遇退化批次会跳过共同项并记录，不能将稳定度视作临床可靠度。对完全塌缩的特征，相关损失有限并不保证有恢复梯度。

默认11个辅助项均开启；各项可通过同名CLI参数设为0。默认固定这些系数只用于首轮研究运行，不代表最佳设置。完整框架对照应保持同一预测架构/输入/患者batch/预算，只关闭待检验损失，再在未用于选参的外测检验。条件匹配损失尤其需要患者配对打乱和独立锚点消融。既有普通融合采用batch=2、较小预测头；本模型默认batch=20且有通路/交互模块，直接比较旧基线只能描述观测，不能将差异归因于某个新损失。

全局模态内16维单位相关目标需要至少17个患者才可能满秩；CLI按已启用目标计算必要下界。默认`--patient-batch-size 20`，最后不足17人的尾批并入前批，不丢患者；这只是秩的必要条件，不代表相关统计稳定。局部patch统计每名患者总权重1/B，多patch不是多患者。patch按chunk读入/反向重算、softmax通过logsumexp精确合并；不构建全队列padding或完整N×K注意力表，进程RSS仍受mmap页和小型计算图影响，真实5090峰值/吞吐需实测。

## 4. 5090现有数据：先检查再跑一次完整模型

目标机前述报告：`/data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv`、`genes.txt`、304个RNA向量可用，fold0为244 train/60 val；原病理特征位于uni目录但真实编码器来源未证。外测未导出。下列是新的输出目录示例，不覆盖原9次基线。

先由服务器助手从原CSV按原顺序取前20个train和前4个val，使用CSV标准库写到新目录（保留多切片JSON字段，不采样patch），运行一轮以检查基因映射、GPU kernel、所有11项有限损失/梯度及实际资源。训练20人满足默认最小17人的秩下界。该小数据工程检查的模型和验证指标不作研究结论。

```bash
env LD_LIBRARY_PATH=/home/ubuntu/.conda/envs/trisurv/lib python -u train_pathway.py \
  --manifest /data1/decur_runs/BLCA_PATHWAY_LOCAL_INPUTS/smoke_train_val.csv \
  --genes /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/genes.txt \
  --out /data1/decur_runs/BLCA_PATHWAY_LOCAL_SMOKE_SEED17_NEW \
  --view-source feature_noise --feature-noise 0.05 --rna-noise 0.05 \
  --common-mode stability --strength 2 --min-weight 0.1 --interaction-strength 0.2 \
  --hidden 16 --common 8 --local-dim 8 --local-common 4 \
  --patient-batch-size 20 --patch-chunk-size 2048 --device cuda:0 \
  --epochs 1 --patience 1 --seed 17 --starts 0 --horizon 12 --time-unit months \
  --endpoint 'Dataset-provided OS in months; time origin unverified'
```

预检确认所有启用项被计算且有限、`common_skipped=false`、交互/局部投影/两个编码器梯度非零、通路与基因覆盖可解释、无非有限梯度或OOM后，直接使用原244/60清单在**新目录**运行完整模型，沿用上述参数，修改：

```text
--manifest /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv
--out /data1/decur_runs/BLCA_PATHWAY_LOCAL_FULL_FOLD0_SEED17_NEW
--epochs 80 --patience 15
```

使用已验证的trisurv cu128解释器和命令级`LD_LIBRARY_PATH`。终端前台运行、保留完整日志；每epoch打印NLL、总损失、batch人数、耗时、global_common跳过数。稳定模式如发生跳过须报告次数及原因，不能把对应batch声称为完整共同约束；预检若跳过则先诊断，不能静默改为等权。默认验证NLL选轮，**不传`--evaluate-test`就不加载test特征/RNA或评估test**。预检失败应保留日志并定位，不静默关损失、丢patch、换队列或用伪图像视图。

`checkpoint.pt`包含模型结构/全部通路索引和名称、模型参数、RNA顺序/哈希、训练mask、尺度、系数、视图来源与输入指纹；`report.json`保存各batch各项原始损失、共同项跳过次数、局部方差诊断、模块梯度范数、匹配权重平均熵/最大权重和逐epoch NLL。`val_predictions.csv`包含患者级生存概率、交互logit及匹配分布摘要。高注意力或高相关不等于真实patch RNA表达。

重载和独立预测：

```python
import torch
from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import predict_pathway_model
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model = PathwayLocalSurvival(**checkpoint['model_kwargs'])
model.load_state_dict(checkpoint['model_state'], strict=True)
fitted = dict(model=model, starts=checkpoint['starts'], stream_config=checkpoint['stream_config'],
              **{key: checkpoint[key] for key in ('rna_mean','rna_scale','rna_mask','rna_preprocessing')})
# 使用checkpoint记录的基因顺序与gene_sha256构造无标签cohort后：
predictions = predict_pathway_model(fitted, cohort_without_labels)
```

如未来已有真实三视图，准备严格cached_uni清单，替换`--view-source image_d4`，训练器将读取reference/A/B而不会用feature_noise伪造D4。特征级和图像级增强应作为不同实验明确报告。

## 5. 实现证据与科学边界

本地合成检查包括：独立NumPy通路相关答案、已知patch/通路匹配、患者打乱对共同损失的影响、病理独立锚点对同患者RNA改变的前向不变性、匹配/局部/交互参数非零梯度、固定视图全量/分块损失及全部参数梯度一致、特征噪声checkpoint重算一致、源文件变化守卫、仅验证/显式测试的隔离、两种视图CLI和检查点重载。一次完整118项CPU检查通过；其后新增资源规模、reference隔离与稳定性共同项（含退化跳过）三项检查另行通过，合计覆盖121个不同检查；并复跑加强后的完整目标梯度检查通过。记录见 [验证回执](results/pathway_framework_validation_20261110.json)。这些检查均未使用真实患者或CUDA。

常规生存预测、通路token和patch交互已有[SurvPath](https://arxiv.org/html/2304.06819v2)，病理/组学局部对应已有[MOTCAT](https://arxiv.org/html/2306.08330v2)，密集共同/独有目标参照[DenseTwins](https://arxiv.org/html/2611.04710v1)。这里是这些思想与具体患者级路径/损失的候选组合；尚未完成穷尽查新，不能宣称首创或有效。
