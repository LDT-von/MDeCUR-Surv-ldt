# 训练集低方差RNA屏蔽与fold0完整基线（2026-11-10）

## 已收到的目标机运行记录

用户转发报告：单视图CUDA流程已在5090执行，fold0训练244患者/291切片/4,763,402 patch/114事件，验证60患者/69切片/1,232,771 patch/26事件。两轮验证NLL为2.3547、2.2319，验证C-index=0.5640（非测试），未加载外测输入。报告墙钟169.76秒、CPU峰值RSS约27,121 MB（含mmap页）、GPU峰值allocated约33.8 MB。环境沿用trisurv的torch 2.10+cu128，命令级 `LD_LIBRARY_PATH=/home/ubuntu/.conda/envs/trisurv/lib` 解决了继承的libstdc++冲突；未改依赖。以上来自用户报告，本地未直接核验服务器文件/日志。

同一报告发现8个训练内常量基因。旧的std下限1e-6使OR10X1的验证值产生约2.174e8的标准化值。有限损失/梯度不等于合理数值尺度；旧两轮用于工程核验，不能据此评价本方法效果。

## 新的可选规则

`--rna-scale-policy clamp` 默认保留旧公式。下一轮显式选 `--rna-scale-policy mask --rna-min-std 1e-6`：

1. 只从训练患者计算每个基因的均值、总体标准差与极差；不接收验证/测试数据或标签来拟合规则。
2. 训练标准差小于阈值或训练内极差为0的基因，在全部分区及训练辅助噪声视图中固定为0。保留原始17025维和基因顺序，原文件不改写。
3. 保留基因使用训练均值/标准差；不裁剪验证值，不用验证/测试值重新拟合。被屏蔽坐标的保存scale置1，推理仍须使用保存的mask。
4. 保存 `rna_mask`、`rna_train_std`、`rna_mean`、`rna_scale`、`rna_preprocessing`。报告记录policy、阈值、原维数、保留/屏蔽数量及索引、训练和验证变换后的最大绝对值。若全部基因被屏蔽或活跃坐标变换后非有限，明确报错。

屏蔽规则只处理训练中无法估计尺度的坐标。RNA上游生成、基因筛选与标准化参照来源仍未知；这个修正不证明上游没有泄漏，也不建立生物学或临床有效性。各折应独立拟合自己的mask，所有比较臂在同一折共享该规则。1e-6是本轮预先固定的阈值，不按验证收益调它。

## 检查点重载

```python
import torch
from cancer_model import CancerDeCUR
from cancer_stream import predict_stream
saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model = CancerDeCUR(**saved['model_kwargs'])
model.load_state_dict(saved['model_state'], strict=True)
fitted = dict(model=model, starts=saved['starts'], stream_config=saved['stream_config'],
              **{key: saved[key] for key in ('rna_mean', 'rna_scale', 'rna_mask', 'rna_preprocessing')})
prediction = predict_stream(fitted, unlabelled_cohort)
```

不要在验证/测试时重新选基因。旧检查点没有mask时兼容旧clamp行为；声明mask策略但缺少mask字段会报错。手工重载新检查点必须携带上述RNA字段。

## 5090接下来可以执行的范围

沿用已经成功的304患者输入清单和cu128解释器。先只读取train/val的RNA向量，调用 `cancer_rna.fit_rna_scaler(train_rna, 'mask', 1e-6)` 和 `transform_rna`；按现有 `genes.txt` 将屏蔽索引映射回名称，保存诊断到新的输出位置。核实OR10X1属于屏蔽项且变换后为0，并报告新的最大绝对值。预期8项来自旧报告，实际名单由当前代码计算，若不一致应解释，不强行覆盖。仍有异常的活跃坐标时先排查，不自动裁剪。

预检通过后，从相同seed重新训练一条完整fold0单视图普通融合基线：最多80轮、验证NLL无改善15轮早停；仅使用244训练/60验证，全patch，外测输入仍不加载。原两轮输出保留，使用全新的目录。保持单区间指数生存模型 `starts=0` 与12个月预测点，作为这条预先固定的基础配置；它不是最终临床模型或经调参选出的最佳区间。

在已成功的trisurv环境内、仓库根目录运行；输出目录必须不存在：

```bash
env LD_LIBRARY_PATH=/home/ubuntu/.conda/envs/trisurv/lib python train_cancer.py \
  --manifest /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv \
  --out /data1/decur_runs/BLCA_SINGLE_VIEW_MASK_FOLD0_SEED17_NEW \
  --training-backend stream --path-views single_view --validation-only \
  --prediction-space encoder --hidden 16 --common 8 \
  --mode off --lambda-within 0 --lambda-path-within 0 --lambda-rna-within 0 \
  --lambda-common 0 --lambda-unique 0 --lambda-local 0 --rna-noise 0 --patch-keep 1 \
  --rna-scale-policy mask --rna-min-std 1e-6 \
  --patient-batch-size 2 --patch-chunk-size 2048 --device cuda:0 \
  --epochs 80 --patience 15 --seed 17 --starts 0 --horizon 12 --time-unit months \
  --endpoint 'Dataset-provided OS in months; time origin unverified'
```

新运行使用训练集拟合的新预处理，不能续接旧clamp模型。结束后报告实际轮数、best epoch、训练/验证NLL轨迹、mask名单及诊断、资源/耗时和输出路径，确认 `test_evaluated=false`。验证指标用于开发选择，不作为独立测试结论。当前完整DeCUR/病理局部目标仍需要合规的reference+A/B，单视图入口不启用这些目标。

## 本地验证

新增8项合成检查：常量/近常量已知答案、旧clamp与旧检查点兼容、活跃坐标不被裁剪、非法/缺失mask拒绝、辅助噪声遵守mask、padded/stream推理一致使用训练mask、改变被屏蔽验证值不改变训练轨迹/选中参数、CLI保存/重载验证预测。最新完整93项检查全部通过（75.880秒），见 [本轮验证回执](results/rna_mask_validation_20261110.json)。本地检查未读取真实患者或运行CUDA。
