# 病理局部约束与患者级 DeCUR：2026-11-09 实现

本轮已实现三个可独立控制的改动：同一 patch 的病理局部相关约束、预测表示与辅助投影分离、病理／RNA 模态内损失系数拆分。实现位于 `cancer_model.py`、`cancer_auxiliary.py`、`cancer_train.py`、`train_cancer.py`。验证回执见 [local_global_validation_20261109.json](results/local_global_validation_20261109.json)。当前证据来自合成输入、数值／梯度检查和缓存接口；没有患者训练结果。

## 需要读的论文及其作用

| 论文 | 在本项目中要回答的问题 | 本轮使用范围 |
| --- | --- | --- |
| [DeCUR: Decoupling Common and Unique Representations for Multimodal Self-supervised Learning](https://arxiv.org/abs/2309.05300) | 模态内去冗余、跨模态共同／独有约束如何组合？ | 已有患者级辅助目标的来源；原始参考源码保留在父目录 |
| [DenseTwins: Dense Modality Decoupling for Multimodal Self-Supervision](https://arxiv.org/html/2611.04710v1) | 全局目标如何扩展到局部表示？投影头为何与下游表示分开？ | 本轮核对方法与公式；任务适配依据论文及附录伪代码 |
| [PIBD: Prototypical Information Bottlenecking and Disentangling for Multimodal Cancer Survival Prediction](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9ead108421b202494d01b5060d12aa34-Abstract-Conference.html) | 癌症生存中共同／独有分解和模态内冗余已有何种处理？ | 近邻与查新对照；本轮没有复现其模型或核验其效果 |
| [SurvPath: Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction](https://arxiv.org/abs/2304.06819) | bulk RNA 如何表示为通路，以及如何建模与病理 patch 的交互？ | 近邻对照；本轮核对原论文摘要，没有移植其编码器 |
| [MOTCAT: Multimodal Optimal Transport-based Co-Attention Transformer with Global Structure Consistency for Survival Prediction](https://arxiv.org/abs/2306.08330) | 病理—组学匹配与全局结构一致性已有何种做法？ | 近邻对照；本轮核对原论文摘要，没有实现 OT 匹配 |
| [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) | 为什么让双视图相关矩阵的对角接近1、非对角接近0？ | 理解本轮相关目标的基础，补充在前两篇之后阅读 |

三篇任务近邻 PIBD、SurvPath、MOTCAT 仍需结合全文和官方实现做进一步方法比较；列入必读清单不表示已经完成全文审查。上述参考不能证明本项目创新性或患者收益。旧 README 中的 MoHD、DIMAF、TTA 保留为扩展查新线索，本轮未重新审查。

## 预测与训练路径

`--prediction-space encoder` 开启分离路径。预测使用同次 UNI 提取的未增强 reference：

```text
reference patch [N,D] → 共享轻量 patch_encoder → attention MIL → hP ┐
                                                                  ├→ concat → 线性生存头
RNA → rna_encoder → hR                                             ┘
```

训练另有辅助路径，梯度没有 detach：

```text
patch A/B → 同一个 patch_encoder → 局部投影器 → 同行局部相关目标
                              └→ attention MIL → 病理辅助投影 [cP,uP]
RNA A/B → 同一个 rna_encoder → RNA辅助投影 [cR,uR]
                   患者级模态内／共同／独有相关目标
```

局部投影器是无偏置线性层，施加于汇聚之前的每个 patch 表示，输出维度由 `--local-dim` 控制。共同／独有切分只定义在患者级辅助投影中；预测使用完整 `hP,hR`。encoder 模式的 `forward` 不调用三个辅助投影器，验证／测试也不需要 A/B 缓存。辅助投影参数仍保存在完整训练检查点中。

默认 `--prediction-space projected --lambda-local 0` 保留历史预测路径与检查点参数布局。局部分支在已有模块之后创建，并保留创建前的随机数状态，使开启局部分支不改变共享参数初值及后续增强随机数序列。

同坐标 A/B 的身份由已有 H5 的 `coords`、`row_index`、校验值和患者／切片来源合同保障；训练器直接接收张量时，调用方仍须保障逐行配对。局部项强制 `cached_uni` 与 `patch_keep=1`，使用完整相同 mask。不会把随机删 patch 的历史视图用于局部配对。

## 局部统计与总损失

对于 B 位患者，第 i 位有 N_i 个有效 patch，权重为：

```text
q_ij = 1 / (B * N_i)
muA  = sum_ij q_ij * zA_ij
varA = sum_ij q_ij * (zA_ij - muA)^2
zA_normalized = (zA - muA) / sqrt(varA + eps)
C = sum_ij q_ij * zA_normalized_ij * zB_normalized_ij^T
L_local = sum_k (C_kk - 1)^2 + 0.005 * sum_{k != l} C_kl^2
```

B 视图使用同样的加权均值与方差；`eps=1e-4`。每位患者的总权重都是 `1/B`。padding 在统计前排除，局部项同时要求至少两位非空患者。重复某位患者所有 patch 不改变该患者的统计权重；大量 patch 仍不等于大量独立患者。

这是采用加权中心化统计的病理局部适配。DenseTwins 式(1)使用未中心化的归一化内积，且其局部匹配使用特征近邻；本实现采用受控同坐标身份，并沿用本仓库的中心化相关定义。两者的数值和对应规则均有区别，不能据此声称复现了原论文全部目标。这里没有 patch—RNA 的局部匹配，也没有 patch 邻接／空间图。

总目标已经接入训练器：

```text
L_total = L_survival
        + lambda_local * L_local
        + lambda_path_within * L_path_within
        + lambda_rna_within * L_rna_within
        + lambda_common * L_common
        + lambda_unique * L_unique
```

`--lambda-path-within`、`--lambda-rna-within` 是各自损失的实际乘数。未给定某个覆盖值时，该项使用 `--lambda-within / 2`，保留历史平均项的系数含义；显式的0会关闭对应项。新增局部系数默认0。`--mode off` 只关闭共同项。

训练日志新增 `path_within_loss`、`rna_within_loss`、`unique_loss`、`local_loss`、`local_diagnostics` 与 `total_loss`；未启用的项记为 null。局部诊断记录患者数、对应 patch 数、最小加权方差、低方差维数和相关对角均值。常量表示的损失有限且会被诊断标出，精确塌缩仍可能具有零梯度，不能承诺这个目标必然防塌缩。

报告和检查点包含 `effective_loss_weights` 与 `model_kwargs`，分别记录实际系数和重建模型所需参数。新检查点按下面方式加载：

```python
checkpoint = torch.load(path, map_location='cpu', weights_only=True)
model = CancerDeCUR(**checkpoint['model_kwargs'])
model.load_state_dict(checkpoint['model_state'], strict=True)
```

## 对照配置与接口

同一轮 A—E 全部使用 `--prediction-space encoder --path-views cached_uni`，保持相同预测头、RNA编码器、患者划分和初始化种子。另行比较 projected／encoder 才能分析投影分离的影响。

| 臂 | mode | path within | RNA within | unique | local | 归因 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | off | 0 | 0 | 0 | 0 | 普通融合 |
| B | uniform | 固定值 | 固定值 | 固定值 | 0 | 患者级 DeCUR |
| C | uniform | 同 B | 同 B | 同 B | 固定值 | 局部项增量 |
| D | stability | 同 B | 同 B | 同 B | 同 C | 稳定性加权增量 |
| E | shuffled | 同 B | 同 B | 同 B | 同 C | 患者权重对应对照 |

这是可执行接口示例，数值用于说明参数，不是已选定的患者超参数，也未执行该患者命令：

```powershell
python train_cancer.py --manifest PATH_TO_PATIENTS.csv --out NEW_OUTPUT_DIR --starts 0 1 3 --horizon 3 --time-unit years --endpoint 'OS from defined baseline' --path-views cached_uni --prediction-space encoder --mode uniform --lambda-local 0.001 --local-dim 16 --lambda-path-within 0.0005 --lambda-rna-within 0.0005 --lambda-common 0.001 --lambda-unique 0.001
```

A 臂需显式将局部、两个 within 和 unique 系数置0，并选择 `--mode off`。RNA仍用现有高斯特征噪声；`--rna-noise 0` 可单独检验。共同稳定性仍是旧余弦定义，尚未修复其对偏置的敏感性。

## 已完成的验证及剩余边界

本轮修改前重跑35项既有检查通过；新增16项针对性检查通过，随后一次完整运行51项全部通过。完整命令：

```powershell
python -B -m unittest -v test_cancer_model test_cancer_data test_cancer_metrics test_prototype test_pipeline test_uni_views test_cancer_local
```

新检查覆盖独立 NumPy 加权矩答案、有限差分梯度、重复单患者 patch 的权重不变性、配对打乱、padding NaN、平移不变性、局部／全局梯度传回编码器、推理跳过辅助投影、独立系数开关、四种共同约束模式、验证选轮、合成 H5 缓存 CLI 和检查点重载。原始默认路径和既有 UNI 工具回归也通过。

这些检查只证明目标、梯度和接口按定义工作。它们没有检验真实增强是否保留预后信息、局部目标是否改善患者预测、辅助投影是否在实际训练中吸收约束、或该组合是否有独立创新。UNI双视图仍需在有原始WSI的目标机核验。后续已实现可选lazy H5／患者小批量／checkpointed patch分块，最新一次70项通过，详见 [stream训练说明](STREAM_TRAINING.md)；默认padded保留，真实队列和CUDA仍未验证。旧UNI ZIP、首次工作区快照及本节51项回执保留历史内容。
