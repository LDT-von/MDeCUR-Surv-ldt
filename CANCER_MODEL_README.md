# 癌症病理—RNA 生存预测模型：研究实现

**本实现可训练、可验证、可输出患者生存曲线，但只是研究模型。稳定性加权机制尚未证明优于普通 DeCUR 或随机权重；不能称为已经验证的新模型，也不能用于临床。** 所有已运行实验均为模拟数据，无真实癌症患者训练或推理。

## 模型和数据

每个患者的输入为 WSI patch 特征矩阵 `[N,D]` 和 RNA 表达特征向量 `[G]`。多个切片若属于同一患者，必须根据预先规定的规则先整理为一个患者的 patch 集合，数据来源和张数需保留。`cancer_model.py` 的 masked attention 多实例汇聚得到病理表示，RNA 编码器生成分子表示。默认 `--prediction-space projected` 保留历史的投影后拼接预测；新增 `--prediction-space encoder` 使用投影前的 `hP,hR` 拼接预测，患者级共同／独有投影仅用于训练辅助目标。两种模式都输出多个分段常数事件发生率。注意力分数是模型权重，不是已经验证的病理解释。

2026-11-09 已实现 DenseTwins 启发的病理局部目标、预测／辅助路径分离和两个模态内系数拆分。论文清单、准确公式、A—E 对照、开启命令与验证边界见 [局部／全局实现说明](LOCAL_GLOBAL_IMPLEMENTATION.md)。局部项默认关闭，历史基线仍可复现。

模型使用连续时间分段指数似然，允许在区间内部精确右删失：第 j 区间发生率 `h_j = softplus(z_j)+epsilon`，患者损失为 `sum_j(h_j * exposure_j) - event * log(h_event_bin)`。`exposure_j` 为实际观察到的该区间时长，末区间无限延伸；事件恰落区间边界时归入从该点开始的区间。输出 `S(t)=exp(-sum_j h_j exposure_j(t))`。时间起点与单位必须一致，固定区间边界只能在训练集上预先设定或在训练内部选择，不能利用测试集选择。假设普通右删失；左截断、复发、竞争风险、多状态、时变变量尚不支持。若选疾病特异终点，其他死因不能未经论证就按普通独立删失处理。

癌症生存任务应预先登记 OS/DSS、起点、最后随访定义、事件来源及样本获取时间。样本在生存计时开始之后采集时需专门处理入组时点和生存者偏倚，当前模型没有自动处理。跨院、跨癌种及跨平台不应视为已经验证。

## DeCUR 和候选约束

病理视图现在有两种明确来源：`--path-views cached_uni` 读取同一 patch 在图像上做完整视野旋转/镜像后由冻结 UNI v1 提取的两份缓存特征，保留全部 patch，强制 `patch_keep=1`；`legacy_patch_dropout` 保留原先的随机 patch 子采样，只用于历史合成对照。RNA 仍使用原有的特征噪声，可通过 `--rna-noise` 明确设置。图像增强尚未证明保留预后信息或改善患者预测。提取、空间成本、边缘处理及新 H5 患者清单见 [UNI 双视图工具](uni_views/README.md)。模态内约束与独特部分的跨模态零相关约束由 DeCUR 思路改写；共同部分支持四种模式：

| `--mode` | 共同约束 |
| --- | --- |
| `off` | 关闭共同项；模态内及独特项仍由各自系数控制 |
| `uniform` | 所有患者等权的共同相关项，作为直接 DeCUR 式对照 |
| `stability` | 用两个模态的视图余弦相似度生成停止梯度的患者权重 |
| `shuffled` | 相同权重随机打乱患者对应关系，作为权重是否真正有效的对照 |

设置 `--mode off --lambda-within 0 --lambda-unique 0` 才是无辅助项的普通融合。其余系数和超参数需在训练/验证流程中事先固定。若共同项在某个训练步骤的加权方差退化或有效样本量过低，程序跳过该项并在训练日志中计数；应检查跳过数、权重均值/标准差及有效样本量。权重只是对人为视图扰动的稳定度，可能受编码器尺度、偏置、采样与肿瘤异质性影响，不能表示病理质量、生物一致性或临床可靠性。

新增 `--lambda-path-within` 与 `--lambda-rna-within` 可独立覆盖模态内系数；各项未覆盖时使用 `lambda-within/2`。`--lambda-local` 开启汇聚之前的同行病理视图相关约束，要求 `cached_uni`、完整相同 mask 和至少两位患者，加权均值／方差／相关矩阵使用 `q_ij=1/(B*N_i)`。若已经设置了这些覆盖值，普通融合还需显式把两项覆盖值和 `--lambda-local` 置0。局部项只处理病理内部对应，尚未建立病理—bulk RNA的局部观测对应。

## 输入清单与运行接口

RNA归一化新增 `--rna-scale-policy mask --rna-min-std 1e-6`，只根据训练集屏蔽低方差坐标，保留输入维度及原文件，推理复用检查点mask；默认clamp兼容历史行为。CLI新增显式 `--patience`（默认15），验证NLL选轮。真实单视图运行报告中的常量基因放大问题、预检与完整fold0命令见 [RNA预处理](RNA_PREPROCESSING.md)。

2026-11-10补充：现有单视图PT可用 `--training-backend stream --path-views single_view`，清单特征列为 `wsi_pt`，保留全部patch及有序多切片；此模式仅做关闭全部辅助项的普通融合。stream新增 `--validation-only`，可只提供train/val，不加载test特征/RNA或解析test标签，输出validation预测和明确的测试未评估标记。格式、可执行命令、目标机接线和85项检查见 [单视图训练说明](SINGLE_VIEW_TRAINING.md)。以下默认padded/三段清单接口继续保留。

模型不下载数据，也不会自动发现本机或服务器上的患者数据。准备一个 CSV，每行一个唯一患者，字段：

```text
patient_id,split,wsi_npy,rna_npy,time,event
case_001,train,features/case_001_wsi.npy,features/case_001_rna.npy,2.4,1
case_002,val,features/case_002_wsi.npy,features/case_002_rna.npy,3.1,0
case_003,test,features/case_003_wsi.npy,features/case_003_rna.npy,1.8,1
```

示例仅展示格式，实际每个划分需要足够患者和事件；这三个名称不对应真实数据。`event=1` 表示事件，`0` 表示右删失。`.npy` 必须分别为数值型二维 `[patch数,D]` 与一维 `[G]`，所有患者维度和 RNA 基因顺序必须一致；路径相对于清单目录。只接受有限数值，缺失值需要另行设计并在训练折内拟合的预处理。脚本检查 ID、维度、掩码、标签以及跨划分重复，但无法判断不同 ID 是否实际上属于同一人，也无法自动验证采样日期、表达标准化、切片来源或 gene 列顺序。

以下是接口格式，**不是已经执行的患者实验**；起点、单位、边界、预测时间点和各项参数须根据预注册方案确认：

```powershell
python train_cancer.py --manifest PATH_TO_MANIFEST.csv --out NEW_OUTPUT_DIR --starts 0 1 3 --horizon 3 --time-unit years --endpoint 'OS from explicitly defined baseline' --mode stability
```

训练时组学均值、标准差只来自训练患者；以验证集 NLL 选择检查点，再评估测试集。输出目录必须不存在，包含 `checkpoint.pt`、测试患者的 `test_predictions.csv` 与 `report.json`；文件中含可识别的患者 ID，需按数据许可和用户的数据管理要求保存。报告记录分区患者/事件数、训练轨迹、测试 NLL、预测时间点 C-index、训练集估计删失分布后的 IPCW Brier、输入张量与清单哈希。C-index 用该时间点预测风险排序，仅是一个风险分数的 Harrell C；IPCW Brier 的有效性依赖删失模型等假设，若训练集对应删失生存率过低则返回空值。还应检查时间点风险集人数、校准、置信区间和外部数据，不能只看这两个指标。输入张量哈希有助于复现，但不能证明患者来源和终点定义正确。

默认训练器仍是全批次padding；新增 `--training-backend stream --path-views cached_uni` 使用lazy H5、患者小批量和checkpointed patch分块，完整汇聚后计算批内相关统计。固定同一批患者时已验证分块的损失／梯度一致性；患者minibatch会改变相关估计覆盖的人群。使用方法、CPU合成内存基准和限制见 [stream训练说明](STREAM_TRAINING.md)。仍无基因ID对齐、自动交叉验证、多中心外部验证或真实GPU吞吐证据。对生成特征的模型版本、区域取样和RNA预处理须独立留档。不要用测试结果反复选配置。

## 已知答案合成研究

`cancer_synthetic.py` 不读取真实数据，模拟共同预后信号、互补模态信号、部分病理视图噪声、以及不稳定视图同时承载预后信号四种场景，比较病理单模态、RNA 单模态、普通融合、DeCUR、随机权重和稳定性权重，按模拟患者独立划分训练/验证/测试并记录组内 NLL。

四轮详细结果和可重复命令见 `results/CANCER_SYNTHETIC_RESULTS.md`。其中在最后固定配置的 6 个新种子中，稳定性权重与等权 DeCUR 的平均 NLL 在四个场景的差异均不超过约 0.0005；在不稳定且含预后信号的模拟场景仅 1/6 次优于等权对照。更强的共同约束在另一轮模拟互补信号中损害了表现。**不能用这些结果宣称新增机制有效、发现临床规律、达到 SOTA 或保证发表。**

代码检查：`python -B -m unittest -v test_cancer_model test_cancer_data test_cancer_metrics test_prototype test_pipeline test_uni_views test_cancer_local`。2026-11-09 本轮完整51项通过，见 [验证回执](results/local_global_validation_20261109.json)。端到端测试使用临时目录中的假患者特征、伪WSI与伪编码器；新增项检查加权矩、有限差分梯度、预测／辅助路径和新配置检查点重载，没有加载真实UNI或训练真实患者。

后续stream改造的最新完整检查包含以上模块及 `test_cancer_stream test_cancer_stream_data test_cancer_stream_cli`，一次70项全部通过，见 [最新回执](results/stream_validation_20261109.json)。
