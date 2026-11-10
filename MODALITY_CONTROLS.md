# 同队列病理/RNA单模态对照（2026-11-10）

## 已有基线及下一步问题

用户转发5090运行完成：39b0563、fold0、seed17、RNA训练集mask（屏蔽8个坐标），244训练/60验证，最多80轮、patience15，实际19轮，验证NLL最优第4轮。该检查点验证NLL=2.186943、C-index=0.6307、IPCW Brier@12月=0.1620；外测未加载。CPU峰值RSS约27GB（含mmap页），GPUallocated峰值33.8MB，墙钟479.21秒。/data1剩8.3TB、该轮产物不到10MB，磁盘不是本次训练阻塞。以上来自用户报告，本地没有访问目标机日志。

训练NLL总体下降、验证NLL在第4轮后未改善，与过拟合或优化波动一致，早停按协议选择了第4轮。训练日志是更新过程中的batch平均NLL，验证值在轮末计算，二者不是同一参数快照上的精确泛化差距。当前一条融合结果还不能回答病理是否在RNA之外增加预测信息。

下一步只补相同fold0/seed17的两条普通单模态对照，复用已有融合结果。以验证NLL选中各自检查点，再在该检查点报告C-index与Brier，不能另外挑最高C-index轮次。它们是开发期对照，不能据此给出独立测试或生物学协同结论。

## 显式入口与行为

新增 `--modality both|pathology|rna`，默认both保留现有预测路径和初始化顺序。模型构造参数、报告和stream配置记录modality，检查点重载必须使用保存的model_kwargs；旧检查点没有该字段时默认both。

- both：执行病理MIL和RNA编码后拼接预测，已有DeCUR/局部辅助接口保持。
- pathology：跳过RNA编码器及RNA投影，RNA预测表示严格为0；病理编码与MIL正常学习。
- rna：跳过病理编码、attention与病理投影；stream不枚举或读取PT patch块，病理预测表示严格为0。

模型保留相同模块初始化顺序与生存头布局；未使用分支没有梯度，生存头对应的输入半区为0。单模态不是将原特征替换成伪数据，也不把无效分支的投影偏置当新特征。RNA-only的padded attention返回全0表示不适用，不能解释成空间热图。

这次是**相同配对患者队列的输入可用性对照**。加载器仍要求同一患者清单的完整病理文件与RNA，校验来源、文件哈希、RNA顺序/维度和划分；RNA-only启动阶段仍会读取PT文件计算SHA256，但训练/推理不使用patch块。Pathology-only也保留相同的RNA预处理记录，模型不使用其数值。这里没有新增可接收任意缺失模态患者的临床部署接口。

单模态只支持生存目标：`--mode off`、所有有效辅助系数0、`--rna-noise 0`。错误地请求共同/独有/双视图辅助项会报错。当前对照均使用single_view、prediction-space=encoder；保持16维编码器、RNA mask、患者batch和优化设置一致。源特征编码器、RNA上游生成和OS起点的已知来源限制仍保留。

## 5090执行范围

1. 拉取当前提交，阅读本文件与 `RNA_PREPROCESSING.md`。保留旧输出与未跟踪结果；复用 `/data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv`，不导出outer_test输入。
2. 使用此前成功的trisurv cu128解释器及命令级LD_LIBRARY_PATH。只新增RNA-only、pathology-only两次seed17训练，各使用新目录；已有both结果不重跑。
3. 两条均最多80轮、验证NLL无改善15轮早停，starts=0、horizon=12个月，保留原全部patch、隐藏维数、batch=2、chunk=2048、学习率/weight decay及mask阈值。RNA-only仍执行相同患者batch和epoch预算，不能按更快的墙钟额外训练更多步。
4. 结束后核对三臂 `cohort_input_sha256` 相同、RNA mask与基因顺序相同、val患者ID顺序一致、time/event相同，再生成对照表与配对预测比较。原both检查点model_kwargs无modality时按默认both解释即可。
5. 输出各自实际轮数、best epoch、该检查点的验证NLL/C-index/Brier、耗时/资源和文件位置；确认test_evaluated=false。异常时保存日志，不能静默改变患者、patch或预处理。

RNA-only命令示例（输出目录必须不存在；pathology-only改modality和新输出目录，其余相同）：

```bash
env LD_LIBRARY_PATH=/home/ubuntu/.conda/envs/trisurv/lib python train_cancer.py \
  --manifest /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv \
  --out /data1/decur_runs/BLCA_SINGLE_VIEW_MASK_FOLD0_SEED17_RNA_ONLY_NEW \
  --training-backend stream --path-views single_view --validation-only --modality rna \
  --prediction-space encoder --hidden 16 --common 8 \
  --mode off --lambda-within 0 --lambda-path-within 0 --lambda-rna-within 0 \
  --lambda-common 0 --lambda-unique 0 --lambda-local 0 --rna-noise 0 --patch-keep 1 \
  --rna-scale-policy mask --rna-min-std 1e-6 \
  --patient-batch-size 2 --patch-chunk-size 2048 --device cuda:0 \
  --epochs 80 --patience 15 --seed 17 --starts 0 --horizon 12 --time-unit months \
  --endpoint 'Dataset-provided OS in months; time origin unverified'
```

固定三条结果后再决定是否增加种子或推进新机制，不能从这一个60人验证集的最佳数值直接声称泛化优势。当前starts=[0]是单区间指数模型，各患者危险率随时间恒定；12月Brier不代表其他时点校准良好。未来DeCUR/局部项仍要求合规三视图，单模态开关没有补上这些输入。

## 本地检查

新增9项CPU合成检查覆盖默认both公式/参数/RNG与旧检查点兼容、未选分支不执行/无梯度、改变未使用输入的预测不变性、RNA-only不枚举patch块、padded/stream预测/NLL/参数梯度一致性、配置拒绝、训练元数据与CLI重载、三臂输入指纹一致。最新完整102项检查通过（84.614秒），回执为 [modality_validation_20261110.json](results/modality_validation_20261110.json)。新单模态入口尚无目标机训练结果。
