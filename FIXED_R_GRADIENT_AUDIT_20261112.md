# 固定R训练批次逐项梯度审计：服务器交接

仅诊断既有新R检查点，求梯度但不更新参数、不创建优化器、不选新轮次、不预测验证集、不展开五折。入口`diagnose_fixed_r_gradients.py`，不能运行`train_pathway.py`代替。

## 固定对象与批次协议

- 新R目录：`/data1/decur_runs/NEW_VERSION_R_FOLD0_SEED17_20261111/`。
- checkpoint SHA256：`6c21fcf518347c817b626a7af514692c25f1d9c02ae4c2db5edb53f1f4bde84b`。据上一批用户转发，与此前归档provenance一致；服务器仍须现场核对。
- manifest：`/data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv`，字节哈希必须匹配checkpoint/report。
- 严格恢复checkpoint内完整模型配置、11项实际系数、RNA scaler/mask、噪声强度、视图类型、batch和patch chunk。完整R必须为正交互强度、全部11项有效辅助系数为正。
- 训练患者原始顺序上，CPU `torch.randperm`固定seed=20261112；按生产`patient_batches`规则分批（包括末尾过小批次合并），选前3批。选择不读取事件/时间来分层，不根据梯度重选。
- 每批2组视图，repeat=11/22；`view_seed=20261112+repeat*100003+batch_index`，`rna_noise_seed=20261112+repeat*100019+batch_index`。批次索引从0开始。
- 对真实R的244训练、batch20，诊断3个不重叠的20人批次，共60名训练患者、6条审计记录。这60人不是原60名验证患者；两次噪声复用患者，不增加独立样本数。
- 全部patch参与，原chunk2048，不采样patch，不改变batch，不做OOM自动降批次。资源不足则保留失败日志并报告，不静默换协议。

## 两步执行与守卫

第一步生成计划：只用CPU加载/核对训练数据和检查点，记录批次ID/顺序/种子、输入哈希、源码哈希、版本、配置；不做模型前向或反向。第二步按计划哈希执行，重新核对计划所有内容；不能自行改JSON、种子、系数后继续。

manifest会检查所有行的ID和split，未选split的特征/RNA路径不打开，time/event不解析。`train_only=True`只加载训练集；验证和外测文件即使不可用也不需要访问。保存的report含历史验证汇总，但本程序不加载验证患者输入或重新计算验证预测。

**历史指纹边界必须如实报告**：旧checkpoint仅有训练/验证合并内容指纹，没有训练集单独指纹。不读取验证输入就不能独立复算该历史合并哈希。本入口核对原manifest字节哈希、checkpoint/report配置、训练人数/事件数/patch数，以及训练RNA scaler重算一致性；另在计划阶段建立训练内容指纹，审计阶段必须匹配。不能把它表述成已独立核对历史训练集单独哈希。计划与本次审计之间，输入被严格锁定。

原RNA scaler仅重算核对，不替换保存值。训练文件内容指纹覆盖患者顺序、RNA/标签张量、病理文件内容校验；RNA原文件额外校验哈希，病理每次分块读取及结束核对文件身份/大小/时间戳。checkpoint/report/manifest在结束时再次检查哈希。

模型以train模式计算原训练目标，开启autograd是为了求导；当前模型使用LayerNorm，发现BatchNorm/dropout则拒绝，避免未审查的状态或随机模式。每条记录后检查state字节哈希不变、所有参数`.grad`仍为空，全部结束后检查CPU/所用CUDA随机流不变。使用`autograd.grad`，无optimizer/backward累加/step。损失分项来自同一次生产前向，分块checkpoint反向重算重放相同视图。

## 服务器步骤

先只读检查本项目进程、git状态、实际GPU占用，保护既有4个未跟踪合成JSON及全部实验目录。若当前有本项目进程在运行，不中断它、不直接换其源码，保留现场并报告。正常空闲且无跟踪修改时，`git pull --ff-only origin main`，核对Windows最终消息给出的完整提交SHA；禁止reset/clean/强制覆盖。

现有trisurv解释器中运行：

```bash
python -m unittest test_fixed_r_gradients test_pathway_gradient_audit
python -m unittest discover
python -m unittest scripts.test_make_development_folds
```

前14项针对性检查包含在完整套件内，不重复相加。本地完整170项（discover 168＋独立scripts 2）CPU检查通过，0失败/错误/跳过，约165秒，回执`results/fixed_r_gradient_release_validation_20261112.json`。目标机记录自己的实际数量和失败，不把Windows结果当作CUDA验证。

建立全新目录`/data1/decur_runs/FIXED_R_TRAIN_GRADIENTS_20261112_<短提交SHA>/`，下列`RUN`指该目录。保存命令、日志、checkpoint事前哈希和计划哈希；不上传患者ID/诊断结果/权重到GitHub。

```bash
python -u diagnose_fixed_r_gradients.py \
  --manifest /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv \
  --run-dir /data1/decur_runs/NEW_VERSION_R_FOLD0_SEED17_20261111 \
  --checkpoint-sha256 6c21fcf518347c817b626a7af514692c25f1d9c02ae4c2db5edb53f1f4bde84b \
  --plan-out "${RUN}/gradient_plan.json"

sha256sum "${RUN}/gradient_plan.json"
```

将计划哈希记录为`PLAN_SHA`，核对计划确为3×20训练患者、两组视图、原11权重、原chunk、无val/test加载。计划成功后无需再次询问用户即可继续已授权诊断。

若现场确认物理GPU1空闲，使用以下命令；若GPU0空闲则相应选择可见设备，但逻辑设备仍`cuda:0`。两卡繁忙可明确使用CPU，不干扰其他任务。

```bash
CUDA_VISIBLE_DEVICES=1 python -u diagnose_fixed_r_gradients.py \
  --manifest /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv \
  --run-dir /data1/decur_runs/NEW_VERSION_R_FOLD0_SEED17_20261111 \
  --checkpoint-sha256 6c21fcf518347c817b626a7af514692c25f1d9c02ae4c2db5edb53f1f4bde84b \
  --plan "${RUN}/gradient_plan.json" \
  --plan-sha256 "${PLAN_SHA}" \
  --device cuda:0 \
  --out "${RUN}/fixed_r_gradients.json"
```

CPU执行时去掉`CUDA_VISIBLE_DEVICES`并用`--device cpu`。程序内部已有最小CUDA上下文初始化；沿用现有解释器和已验证的命令级库路径，不安装/替换环境。如需额外测RSS wrapper，只包装本诊断脚本并归档源码，禁止套用固定执行训练入口的旧wrapper。错误时保存计划/日志，不放宽检查或更换种子重试；确认纯环境问题后的同计划重试需另存日志并如实报告。

## 输出与解释

JSON逐批次、逐视图、逐辅助项输出与生存NLL的梯度余弦/内积、原始与实际带权梯度范数、相对生存梯度比例；全生存反向图空间与各模块分别统计。专属辅助投影的图外梯度单列；未连接、连接但零梯度、跳过、关闭、可定义方向分别记录，不把未定义余弦写成0。

另有带权辅助梯度之和、总梯度方向、逐项梯度重构总梯度的误差。汇总仅计数与均值/中位数/极值，不把6条记录当6个独立患者实验，不据此生成泛化显著性结论。

优先查看patch encoder、RNA encoder、pathway tokenizer、matching_path、matching_rna及两个预测头。小的生存梯度分母会放大比例，必须同时报告绝对范数。负内积表示当前参数/训练批次上的普通SGD一阶冲突，不等于AdamW实际更新，也不证明正则项损害验证效果。最佳检查点不能代表历史各轮梯度；不能据单批次自动删除loss、修改系数或开始训练。

回报：完整提交SHA、解释器/torch/设备、计划与checkpoint哈希、训练输入指纹及上述历史边界、3批患者数/事件数/patch数（聊天不必贴患者ID）、6记录完成状态、11项总体与重点模块方向/量级表、共同项跳过原因、总辅助梯度、最大重构误差、模型前后哈希、`.grad`/RNG/输入不变守卫、耗时/峰值显存及输出SHA。真实患者结果回到本窗口后再决定有限单因素候选；不五折，不读取outer_test。
