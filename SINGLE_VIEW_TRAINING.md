# 现成单视图特征的全 patch 流式基线（2026-11-10）

新增 `--training-backend stream --path-views single_view`，直接安全内存映射现有 `.pt` 浮点张量，不要求原始WSI、坐标属性或UNI权重。每位患者的所有明确列出的切片按清单顺序汇聚，保留全部patch。它是现成特征＋RNA的普通融合生存基线；没有构造A/B视图，没有实施局部目标或新增跨模态交互。

## 输入与来源

患者CSV必需字段为：

```text
patient_id,split,wsi_pt,rna_npy,time,event
FAKE_CASE_001,train,features/FAKE_SLIDE_001.pt,rna/FAKE_CASE_001.npy,0,0
FAKE_CASE_002,train,features/FAKE_SLIDE_002.pt,rna/FAKE_CASE_002.npy,18,1
FAKE_CASE_003,val,features/FAKE_SLIDE_003.pt,rna/FAKE_CASE_003.npy,24,0
```

这只是合成格式示例。每个ID唯一；`wsi_pt`可以是单一路径，也可以是CSV中正确转义的JSON路径列表，例如字段内容为 `["features/slideA.pt","features/slideB.pt"]`，由CSV writer处理外层引号。路径相对于患者清单目录。多切片需要在同一患者行列全，不能把切片当独立患者。患者归属由清单及外部映射证据提供，加载器不从文件名推断。

每个PT必须是单个非空实浮点 `[N,D]` 张量，支持 `torch.load(..., map_location='cpu', weights_only=True, mmap=True)`。不支持mmap的旧序列化文件会明确报错，不退回一次性加载；应另行显式转换到新文件并保留来源。每块复制到CPU float32后检查有限性，再送入模型。没有A/B数据集；请求增强视图会失败。

`cancer_single_view.py:PTPatientBag`记录源文件SHA256、形状、dtype、切片顺序，核对重复路径/物理文件/切片名以及运行中的文件变动。首次加载会完整读取源文件计算哈希，产生I/O成本；后续特征由内存映射按需分页。映射的页面可能留在操作系统/进程工作集中，不能把chunk大小等同于严格的RAM上限。模型激活与原始特征块避免了全队列padding；小型autograd状态仍随chunk数增长。长生命周期调用者可用 `bag.close()` 释放映射。

来源固定标记为 `external_single_view_pt`、`encoder_status=unverified`，不会从目录名或1024维推断UNI v1。哈希证明输入文件身份，不能证明患者、预处理或编码器来源。RNA须为同一基因顺序的一维有限数值 `.npy`；当前入口检查维数，不自动验证基因ID。准备清单时另存基因顺序及其校验值、患者映射和输入表校验值。

## 生存目标与测试隔离

single_view强制 `patch_keep=1`，`--mode off`，全部有效辅助损失系数为0，`--rna-noise 0`。注意mode off本身只关闭共同项，所以还要显式关闭within与unique，已有独立within覆盖值也必须为0。配置不兼容会在读取患者文件之前报错。

新增stream专用 `--validation-only`：只要求train/val两段，训练集RNA均值/标准差用于两段；验证NLL选轮，输出 `validation_predictions.csv`、`checkpoint.pt` 与 `report.json`。报告写 `evaluation_split=val`、`test_evaluated=false`，不产生test指标或test预测文件。如果传入含test行的清单，仍检查其ID/split以发现重复ID，但不打开该行特征/RNA文件、不解析其time/event。推荐流程核验直接使用仅含train/val的清单。输入清单自身仍会被读取并计算哈希。

未加 `--validation-only` 时仍要求显式train/val/test，选择验证最优检查点后计算test指标，保留原接口。它不是开发调试默认模式。数据指纹方案 `single-view-pt-stream-v1`包含实际加载分区的有序患者、源文件内容记录、RNA和标签；不会构造整队列padding张量。

普通右删失 `time=0,event=0` 是有效输入，似然贡献为0，不需要因为它更改分箱或自动删人。时间起点未核实应明确记录；不要将数据集提供的OS月数写成已核实的诊断/手术起点。

## 给5090的接线与两轮流程核验

以下数据/环境信息来自用户转发的5090只读报告，本地没有独立访问或训练：BLCA候选380患者/451切片，fold0 inner_train=244人/291切片、inner_val=60人/69切片，outer_test=76人/91切片；单文件PT mmap已在目标机报告成功，trisurv/dct的cu128构建报告基础CUDA kernel可用。base的旧cu124构建报告不兼容5090，使用已有兼容环境，不自动安装依赖。

1. 保留既有划分与 `results/repeat_cancer_v*.json`。在原数据目录之外创建新的输入/输出目录，例如 `/data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS`。从既有fold0映射inner_train→train、inner_val→val；先不导出outer_test输入。
2. 按临床 `wsi` 列表及已核对的完整切片名，生成每患者有序PT路径列表。RNA保留现有17025基因顺序，逐患者另存向量，记录原始CSV、基因顺序和划分文件哈希。标签用 `clinical/all/blca.csv` 的有效OS月数、`event=1-censorship_os`；不能混用另一套近似缩放的时间。
3. 输入准备时报告训练集原RNA方差过小的基因数，以及按训练均值/标准差转换后train/val的最大绝对值。保留当前标准差下限 `1e-6`，不自行删OR10X1、不裁剪、不增加特征筛选。RNA生成流程/参照样本与OS时间起点仍记为未知。统计分布本身不能确定其生成机制。
4. 可先从原inner_train/inner_val各选2位患者建立独立的工程检查清单，仍保留每位患者全部patch，运行1轮；通过后用原244/60患者清单运行2轮。两者都是流程核验，不改变正式划分，不使用outer_test，也不据此宣称患者效果或确定最终超参数。报告实际CUDA峰值显存、CPU峰值RSS、耗时、患者/patch数量和有限梯度/损失情况。

从仓库根目录，在已核实的cu128环境中执行。下面的输入/输出路径必须换成实际新建路径；`starts=0`（单区间指数模型）与12个月horizon只用于这次工程核验，不是已选定的最终生存方案：

```bash
python train_cancer.py \
  --manifest /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv \
  --out /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_SMOKE \
  --training-backend stream --path-views single_view --validation-only \
  --prediction-space encoder --hidden 16 --common 8 \
  --mode off --lambda-within 0 --lambda-path-within 0 --lambda-rna-within 0 \
  --lambda-common 0 --lambda-unique 0 --lambda-local 0 --rna-noise 0 --patch-keep 1 \
  --patient-batch-size 2 --patch-chunk-size 2048 --device cuda:0 \
  --epochs 2 --seed 17 --starts 0 --horizon 12 --time-unit months \
  --endpoint 'Dataset-provided OS in months; time origin unverified'
```

若CPU/GPU内存、非有限值或设备kernel报错，保留日志和输出现场，先定位问题；不要自动改患者集合、丢patch或读outer_test。正式多折/多种子比较需固定输入、RNA处理、模型容量、患者batch及更新预算，再设定生存区间与评估方案。

## 本地证据

完整85项合成/回归检查通过（既有70＋新增15），见 [验证回执](results/single_view_validation_20261110.json)。新增检查覆盖安全mmap和无eager回退、完整多切片顺序、分块独立副本/变更守卫、无伪A/B、单视图配置、time=0删失、完整bag预测/NLL/全部参数梯度/AdamW一步更新一致性、train-only RNA归一化、validation-only隔离、CLI检查点重载和旧test模式保留。它们是CPU合成证据，没有真实患者、UNI提取或CUDA训练证据。
