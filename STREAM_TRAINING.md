# 按患者与patch分块训练：2026-11-09

用户确认真实WSI、坐标和UNI权重仍在另一台电脑，本轮只完成本地改造与合成检查。新增 `--training-backend stream` 已接入现有癌症CLI；默认 `padded` 保留旧接口。

## 已解决的工程问题

- `cancer_stream_data.py` 只保留患者ID、RNA、标签和已验证的H5描述符，不把所有患者的reference/A/B读入内存，不创建全队列padding张量。支持同患者有序多切片。
- 缓存特征、坐标、row_index和D4记录都分块校验。第一次仍需完整扫描文件核验校验和，这会产生I/O成本。
- `cancer_stream.py` 在一批患者内逐个处理bag，每次最多读取指定数量patch。使用checkpoint重算，反向传播重新读取相同H5区间，减少特征和激活驻留。
- 每个患者完整汇聚后才计算患者级模态内、共同和独有相关项。局部项先汇总完整统计量，保留 `q_ij=1/(B*N_i)`，不会把各chunk的相关损失直接平均。
- 启用相关项时患者batch至少2人。末批若只有1人，并入前批；记录真实批量，不丢患者，因此实际批量可比指定值多1人。
- RNA统计只拟合训练集；验证NLL选轮；验证／测试推理仅读reference，预测不返回全部patch注意力向量。
- H5核验后，读取前后检查文件身份、大小和时间戳；forward后若修改文件，checkpoint反向重读会失败。该检查用于防止运行中误改，不是针对任意蓄意篡改的安全保证。

## 数学边界

**固定同一批患者时，patch分块应与完整bag计算在数值容差内一致。患者小批量会改变每次相关估计覆盖的人群。**不能把后者称为原全队列相关目标的完全等价实现。初轮对照应保持患者批量、顺序种子、更新预算和各项超参数相同；批量大小可设置为整个训练集人数，以保留整队列相关估计。

每个chunk的注意力统计为logsumexp分数 `l` 与softmax加权均值 `m`。两个chunk合并：

```text
l_total = logaddexp(l_left, l_right)
m_total = exp(l_left-l_total)*m_left + exp(l_right-l_total)*m_right
```

局部A/B先计算每chunk的均值、方差和交叉协方差，再用可微Welford合并为每患者统计。若合并右侧的patch比例为t：

```text
deltaA = meanA_right - meanA_left
meanA = meanA_left + t*deltaA
varA = (1-t)*varA_left + t*varA_right + t*(1-t)*deltaA^2
covAB = (1-t)*covAB_left + t*covAB_right + t*(1-t)*deltaA*deltaB^T
```

所有患者随后等权合并，包含患者内与患者间变异，再计算局部相关损失。中心矩使用float64累积，避免直接 `E[z²]-E[z]²` 消减；预测／编码器默认仍为float32。RNA噪声在checkpoint外一次确定，H5区间和视图名固定绑定到重算函数。PyTorch非reentrant checkpoint保留随机数状态；当前模型只有确定性的线性层、LayerNorm、GELU、Tanh，如未来加入有状态层，应重新验证重算行为。

内存仍包含RNA、患者表示，以及每chunk的小型汇聚／中心矩状态和autograd图，随chunk数量增长。实现降低了原始特征与激活驻留，没有承诺严格常数内存。患者batch很小时，全局相关估计仍可能噪声较大或退化；局部patch数不能补足独立患者数，需要检查批量、ESS、跳过次数和方差。

## 接口示例

患者CSV继续使用已验证的完整UNI v1 H5，每张切片必须含reference＋A/B，沿用既有split；不能使用pilot文件或混用旧下载特征。以下是待目标机接线后的格式示例，本轮未执行患者命令：

```powershell
python train_cancer.py --manifest PATH_TO_PATIENTS.csv --out NEW_OUTPUT_DIR --starts 0 1 3 --horizon 3 --time-unit years --endpoint 'OS from defined baseline' --path-views cached_uni --training-backend stream --patient-batch-size 16 --patch-chunk-size 2048 --device cpu --prediction-space encoder --hidden 16 --common 8 --mode uniform --lambda-local 0.001 --lambda-path-within 0.0005 --lambda-rna-within 0.0005 --lambda-common 0.001 --lambda-unique 0.001
```

16与2048是接口默认值，不是患者队列调参结论。`--device`只接受显式选择，不自动使用GPU；本轮只验证CPU，CUDA执行、显存、真实H5吞吐尚未核实。

报告记录 `stream_config`、每步患者数与各损失、实际有效系数、缓存来源和 `cohort_input_sha256`。stream使用 `verified-cache-stream-v1`：按有序患者ID、RNA、标签和已验证的特征校验值形成指纹；不构造padding张量，所以 `cohort_tensor_sha256` 为null。指纹不依赖缓存绝对路径或mtime，仍不能证明临床身份／终点来源正确。

检查点按以下方式重载，再使用 `predict_stream`：

```python
from cancer_model import CancerDeCUR
from cancer_stream import predict_stream
checkpoint = torch.load(path, map_location='cpu', weights_only=True)
model = CancerDeCUR(**checkpoint['model_kwargs'])
model.load_state_dict(checkpoint['model_state'], strict=True)
fitted = dict(model=model, rna_mean=checkpoint['rna_mean'],
              rna_scale=checkpoint['rna_scale'], starts=checkpoint['starts'],
              stream_config=checkpoint['stream_config'])
prediction = predict_stream(fitted, cohort_without_labels)
```

## 本轮证据

一次完整运行 **70项检查全部通过**，包括此前51项和本轮19项（数学／梯度8项、lazy数据9项、CLI与H5重算守卫2项）。完整命令与源码校验见 [验证回执](results/stream_validation_20261109.json)。

同一患者集合、固定A/B与RNA视图，检查覆盖chunk=1、非整除、chunk超过bag大小、checkpoint开关、reference预测、辅助投影、局部目标、联合损失、全部参数梯度以及AdamW一步参数更新的一致性；另检查极端分数、大常量偏移、尾批合并、无标签预测和检查点重载。

另以4名合成患者、每人8192个1024维patch、三视图、chunk=512，在两个新CPU进程中测量一次合成特征生成＋模型前向／反向：

| 方式 | 进程峰值工作集 | 特征生成＋前向／反向 | 单次最多读取行数 |
| --- | ---: | ---: | ---: |
| 完整padding | 1262.84 MiB | 0.935秒 | 8192 |
| stream＋重算 | 465.41 MiB | 2.190秒 | 512 |

本例峰值工作集降低约63.1%，耗时约为2.34倍。工作集包括Python、PyTorch和分配器；输入是即时生成的合成特征，未测量真实H5磁盘或UNI吞吐，也不代表真实队列可达到相同比例。两次loss差约2.1e-8，梯度L2范数差约9.2e-7；更完整的参数梯度一致性由双精度检查覆盖。[原始基准记录](results/stream_memory_benchmark_20261109.json)

复现基准需要提供新的输出路径，不会覆盖旧记录：

```powershell
python -B benchmark_cancer_stream.py --out results/NEW_STREAM_BENCHMARK.json
```

## 仍待目标机完成

真实WSI解码／冻结UNI小规模试提取、患者与RNA／切片对应核验、缺失切片影响、实际H5读取速度、CPU/GPU内存和训练吞吐均未完成。稳定性定义、预测器容量与RNA通路建模没有在本轮改动；A/B/C等患者对照尚未运行。本轮是工程与数值证据，没有患者预测收益或临床效果证据。现有冻结ZIP和首次快照仍保留历史内容，不能代表本轮代码。
