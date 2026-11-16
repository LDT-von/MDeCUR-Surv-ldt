# DCT 对齐协议（DCT Parity Protocol）

> 当前修复版的入口接线、验证回执和仅 BLCA fold0 的 V3 运行命令见 [PATIENTWISE_V3_HANDOFF_20261116.md](PATIENTWISE_V3_HANDOFF_20261116.md)。旧协议命令不作为本次重跑指令。

> **2026-11-16：历史离散 NLL 存在跨患者广播漏洞，以下旧数字保留为历史记录。**
> 旧实现将形状为 `[B]` 的删失权重与 `[B, 1]` 的逐患者损失相乘，得到 `[B, B]`，
> 导致权重跨患者混合。**本仓库所有使用该旧离散 NLL、且包含混合事件/删失患者批次的历史训练均受影响**；
> 涉及的验证 NLL 也不能按正确的逐患者 NLL 解释。保存预测计算出的 C 仍是相应预测的排序结果，
> 但不能把这些模型称作“在正确 NLL 协议下训练的结果”，不能据其确定辅助目标或 batch 的净作用。
> 修正评分时间或重算旧预测不能修复已受影响的训练权重。修复后的训练必须使用新协议标识和新目录，
> 不覆盖旧检查点、预测、日志或下表数字；是否启动由用户决定，本轮不自动展开训练或五折。
> 该说明针对已定位的本仓库实现，不据此推断其他仓库或不同损失实现也有同一漏洞。

> **2026-11-14 默认主方法更新**：当前 `train_pathway_parity.py` 默认完整dense模型及全部辅助。下面记录的是原始参考/历史aux-off实验；历史架构复现还须显式 `--legacy-architecture`，不把aux-off作为主方法。新的一折同协议比较、评估时间对齐审计及服务器命令见 `FULL_DENSE_TOKEN_PROTOCOL_20261114.md`。

本文件说明本仓库的 `train_pathway_parity.py` 采用的**参考协议**：切分、生存终点、
时间离散化、损失、patch 预算、RNA 预处理、计分口径与模型选择规则。
目标是对齐本仓库与参考实现 `DCT-Reg` 的**数据处理与计分口径**。
下文记录了历史对齐设置，不代表每项语义已经正确；NLL 广播漏洞和旧时间映射需要单独审计，
不能仅凭部分指标逐位相同宣称整个训练/评估协议等价。

> 范围声明：本协议对齐的是**协议**（切分/终点/分箱/损失/patch 预算/计分/选轮），
> **不是**对齐模型结构。本仓库的模型是通路局部生存模型，与参考实现的
> 模型/transport 机制不同。因此 parity 数字用于回答"在完全相同的协议下，
> 我们的模型表现如何"，**不能**声称复现了参考实现的模型结果。

---

## 1. 切分（split）

| 项目 | 取值 |
|---|---|
| 来源 | `/data1/dataset_csv/splits/5fold/blca/fold_0.csv` |
| 列 | `train` / `val`（legacy validation，非独立 outer test） |
| train 患者 | 303 |
| val 患者 | 77 |
| train 事件 | 101 |
| val 事件 | 27 |

临床表 `blca.csv` 缺失标签行按参考实现的做法 `dropna()` 丢弃
（BLCA 中为 1 例，编号见 `metadata.json` 的 `patients_dropped_for_missing_labels`）。
同一患者多次记录取首行去重。

**重要**：`val` 是参考实现用于选轮的 legacy validation 集，
**不是**独立测试集。在 val 上选出的最优 C-index 是乐观有偏的，
不能当作独立测试估计。协议的所有产物都显式声明这一点。

## 2. 生存终点（endpoint）

- **DSS**（disease-specific survival）：`survival_months_dss`
- 删失指示：`censorship_dss`，其中 `1 = censored`，`0 = event`
- 因此本仓库统一使用 `event = 1 - censorship_dss`

## 3. 时间离散化与损失

- 在**训练集**的未删失 DSS 时间上用 `pd.qcut(q=4, labels=False, duplicates='drop')`
  拟合 4 个分箱边界，外侧边界替换为 `-inf` / `+inf`
- 标签离散化：时间落在哪个箱 → 该箱为 1（one-hot）
- 损失：**逐患者离散 NLL**（negative log-likelihood of the discrete survival），
  采用本仓库 `patientwise_clamped_discrete_v1`；事件权重为 1，删失权重为 `1-alpha`。
  原参考训练器是否存在同类广播问题尚未核实，不能由旧指标一致推断训练目标等价。

分箱边界在 BLCA fold0 上的取值（双精度）：

```
[-inf, 8.087582602, 15.12312194, 21.40250518, inf]
```

## 4. 精度约定（关键）

参考实现**用双精度（float64）的临床值**拟合分箱边界、删失分布和指标时间网格，
而**模型输入是 float32 张量**。本仓库照此实现：

- `manifest.csv` —— 模型输入用 float32 精度的时间（`%.6f`）
- `survival_reference.csv` —— 分箱/删失/指标网格用的**双精度**原始临床时间与事件

二者分离保存。若把 float32 时间用于分箱，边界会有约 `1e-8` 的偏差。

## 5. patch 预算

| 阶段 | 采样方式 |
|---|---|
| 训练 | 从该患者**所有 slide 拼接后**的行中**无放回随机**抽取 4096 行，每个 epoch 重抽 |
| 评估 | **等间隔**确定性抽取 4096 行 |
| 患者不足 4096 行 | 末尾**显式补零行**到 4096 |

行数不足的患者（BLCA fold0：train 24511 行、val 10579 行被补齐）
与参考实现一致。

## 6. RNA 预处理

- `--rna-scale-policy none`：原始基因表达**直接输入**，不做标准化
  （与参考实现的 `Pathways` 格式一致）
- 基因集合来自参考实现使用的 `combine_signatures.csv` → GMT，
  得到 **329 条通路 / 9532 个匹配基因位点**
  （2 条空通路被跳过，与参考实现一致）

## 7. 计分口径

风险定义为**离散生存概率在 4 个箱上的负和**：

```
risk = -sum_j survival_j
```

评估指标全部用 `sksurv`：

| 指标 | 实现 |
|---|---|
| Harrell C-index | `concordance_index_censored` |
| IPCW C-index | `concordance_index_ipcw`（删失分布仅用**训练行**拟合） |
| Brier | `brier_score`（逐时间点） |
| IBS | `integrated_brier_score` |
| IAUC | `cumulative_dynamic_auc` |

**历史指标时间网格**按参考实现的 `_select_valid_metric_time_grid` 规则确定，
BLCA fold0 存档网格为 `[8.087582602, 15.12312194]`（2 个时间点）。
该旧实现存在生存列与对应时间边界错位的问题；下文旧 Brier/IBS/IAUC 仅保留作 legacy 复算记录，
不能作为时间语义已修正的预测误差或动态区分度证据。独立的 `corrected/time_semantic_audit`
只在有支持的有限训练边界上评估对应生存列，不把开放最后一箱配到验证集最大随访时间。
Harrell C/IPCW C 使用 `-sum(S)`，不经过这套时间网格；时间映射缺陷不能解释两种 C 的逆序。

### 交叉验证

`test_cancer_parity_reference.py` 直接加载参考实现的
`_select_valid_metric_time_grid` 与 `_calculate_metrics`，
逐位比较本仓库的输出。在 fold0 上：

```
C diff    0.0
IPCW diff 0.0
IBS diff  0.0
IAUC diff 0.0
```

这些检查证明与参考计算一致，不排除双方共享的约定问题，也没有验证旧 NLL 的逐患者广播语义。

## 8. 模型选择

- 指标：**验证集 C-index 峰值**（`best_cindex`）
- 平局时取**最早的 epoch**
- 不触碰 outer test

## 9. 运行

```bash
# 1) 构建清单（清单 + 双精度参考 + 每患者 RNA npy）
python scripts/build_parity_manifest.py \
  --clinical /data1/dataset_csv/clinical/all/blca.csv \
  --split /data1/dataset_csv/splits/5fold/blca/fold_0.csv \
  --rna /data1/TCGA_Gene_Data/BLCA/rna_inter.csv \
  --feature-dir /data/CPathPatchFeature/blca/uni/pt_files \
  --out /data1/decur_runs/PARITY_DCT_FOLD0 \
  --endpoint dss --split-columns train val

# 2) 构建 GMT
python scripts/combine_signatures_to_gmt.py \
  --signature-csv /data1/dataset_csv/signatures/combine_signatures.csv \
  --genes /data1/decur_runs/PARITY_DCT_FOLD0/genes.txt \
  --out-gmt /data1/decur_runs/PARITY_DCT_FOLD0/combine_signatures.gmt \
  --min-genes 1

# 3) 训练
python train_pathway_parity.py \
  --manifest /data1/decur_runs/PARITY_DCT_FOLD0/manifest.csv \
  --survival-reference /data1/decur_runs/PARITY_DCT_FOLD0/survival_reference.csv \
  --genes /data1/decur_runs/PARITY_DCT_FOLD0/genes.txt \
  --pathways /data1/decur_runs/PARITY_DCT_FOLD0/combine_signatures.gmt \
  --out /data1/decur_runs/PARITY_DCT_FOLD0/run_smoke \
  --num-patches 4096 --n-bins 4 --alpha-surv 0.5 --epochs 2 \
  --patient-batch-size 8 --patch-chunk-size 4096 --legacy-architecture --disable-auxiliary \
  --rna-scale-policy none \
  --endpoint DSS --time-unit months --split-name "5fold/blca/fold_0 (legacy_val)" \
  --device cuda
```

### 环境

参考实现与本仓库的 parity 脚本依赖 `sksurv` 与支持 sm_120 的 PyTorch，
本机使用 `/home/ubuntu/.conda/envs/trisurv/bin/python`
（torch 2.10.0+cu128，sksurv 0.25.0）。
系统默认 `python3` 的 PyTorch 不支持 RTX 5090（sm_120），在 GPU 上会报
`no kernel image is available for execution on the device`。

## 10. 性能

| 项目 | 数值 |
|---|---|
| GPU 前向（单患者 4096 patch） | 0.18 s |
| GPU 前向+反向（单患者） | 0.20 s |
| epoch（303 训练患者） | 约 61 s（首个 epoch 因特征文件 sha256 校验约 137 s） |

`PTPatientBag` 构造时会对每个 slide 文件做 sha256 校验（29 GB / 457 个文件），
这是既有的完整性审计契约，不因性能而省略。

## 11. fold0 结果（2026-11-13）

30 轮完整训练，`--num-patches 4096 --alpha-surv 0.5 --rna-scale-policy none`，
lr 5e-4 / weight_decay 5e-4 / seed 3；历史记录意图对齐参考实现的
`dct_v313_blca_uni_p4096_5fold_aligned`。这些配置字段相同不能替代完整运行核对，
也不能抵消本仓库旧 NLL 的实现缺陷。

产物：`/data1/decur_runs/PARITY_DCT_FOLD0/run_fold0_30ep/`

### 与参考实现的对照（同一切分、同一 UNI v1、同一 p4096、同一 30 轮）

| 指标 | 本仓库（fold0 峰值 ep20） | 参考实现（fold0 峰值 ep7） |
|---|---|---|
| Harrell C-index | **0.7361** | 0.7266 |
| IPCW C-index | 0.5158 | 0.5235 |
| IBS | **0.1239** | 0.2086 |
| IAUC | **0.8313** | 0.7937 |
| 最终轮 C-index | 0.7076 | 0.6371 |

C-index 的历史点估计高 0.0095。IBS 低 0.0847、IAUC 高 0.0376 是旧时间映射下的数值差，
暂不用于推断校准或动态预测改善。NLL 广播漏洞还使本仓库这次训练不能作为正确似然协议的对照。
本仓库验证 C 从峰值 0.7361 到末轮 0.7076，参考运行从 0.7266 到末轮 0.6371；
两个时点不足以证明收敛稳定性或泛化稳定性更好，末轮也不是独立重复实验。

**这些数字都不能作为独立测试结果**：val 是 legacy validation 集，
峰值是按 val C-index 选出来的，乐观有偏。参考实现的 5 折
outer test 数字（`results/dct_main_table_cindex.json`，BLCA mean 0.7208）
是另一套评估协议，不能与本表直接比较。

### 参考实现五折（legacy_val，各折 val 峰值）

| fold | C | IPCW | IBS | IAUC |
|---|---|---|---|---|
| 0 | 0.7266 | 0.5235 | 0.2086 | 0.7937 |
| 1 | 0.7259 | 0.6631 | 0.1540 | 0.7018 |
| 2 | 0.6573 | 0.6649 | 0.7891 | 0.6852 |
| 3 | 0.6623 | 0.6342 | 0.2300 | 0.3518 |
| 4 | 0.7304 | 0.6864 | 0.2369 | 0.5545 |

参考运行的折间范围为 0.6573–0.7304，本仓库此历史批次只跑了 fold0。
**单折点估计不能说明整体领先**。先处理现有协议缺陷和运行来源核对，本阶段不自动补跑其余折。

### 切分目录的注意事项

参考实现有**两套**切分目录：

| 目录 | 指向 | fold0 构成 |
|---|---|---|
| `5fold` | `dataset_splits_summary/uni` | train 304 / val 77 |
| `5fold_uni2h` | `dataset_splits_summary/uni2h` | train 304 / val 76 |

两者的 train/val 成员**完全不同**（互为交换）。
本协议用的是 `5fold`（UNI v1），与参考实现的
`dct_v313_blca_uni_p4096_5fold_aligned` 一致；
而参考实现的 `e046b_legacy_val_full_replay` 用的是 `5fold_uni2h`。
对照时必须核对 `experiment_settings.txt` 里的 `split_dir`，
否则数字不可比。

## 12. 未做的事

- 未跑五折全量，只跑通 fold0 的端到端链路
- 未使用 UNI2-h（按约定先用 UNI v1 `.pt`）
- 未触碰 outer test
- 本节描述历史批次，不代表当前代码从未修改；本轮 NLL 修复需要另立训练批次

## 13. fold0 对照组（2026-11-13）

在同一协议（同一清单、同一分箱、同一 patch 预算、同一 seed 3、同一按 val
C-index 峰值选轮）下运行的一组对照，用来回答"0.7361 这个分数是谁贡献的"。
全部产物在 `/data1/decur_runs/PARITY_DCT_FOLD0/`。

### 13.1 结果

| 对照 | 设置 | 存档均值 C | 所报告验证 C | 说明 |
|---|---|---|---|---|
| 历史预测架构，辅助关闭 | `--disable-auxiliary`，batch 8 | 0.6929 | **0.7361** | 按验证 C 选轮的历史参考运行 |
| 推理时病理置零 | 同一 checkpoint，推理期病理置零 | — | **0.7686** | 冻结干预后 C 增加 0.0325，不证明病理无用 |
| 推理时 RNA 置零 | 同一 checkpoint，推理期 RNA 置零 | — | 0.5531 | 冻结干预后 C 减少 0.1830，不是解释贡献比例 |
| `rna_only` | 训练+推理都去掉病理 | 0.6938 | 0.7639 | 历史重训练峰值比 0.7361 高 0.0278 |
| `null` | 训练标签置乱 | 0.5648 | 0.6363 | 单次置乱与验证选轮，不能证明有/无泄漏 |
| `path_only` | 训练+推理都去掉 RNA | 0.5585 | 0.6006 | 历史重训练峰值；未量化不确定性 |
| raw RNA 岭回归 | 17025 维原始表达直接岭回归 | — | **0.7472** | 调参、删失处理与输入协议仍需核对 |
| 协同交互关掉 | 同一 checkpoint，`interaction_strength=0` | — | 0.7314 | 冻结干预差 −0.0047，不是重训练增量 |

“存档均值 C”需要补齐其汇总定义；若指各 epoch 的均值，不能把 epoch 当独立重复样本。
除 §11 的参考运行外，本节尚缺各对照的完整目录与 checkpoint SHA；在补齐前不把推理干预称作重新选轮峰值。

### 13.2 结论（重要）

1. 这些探索结果提示当前模型对 RNA 输入的依赖值得检查。冻结输入置零、重训练单模态模型和岭回归
   回答不同问题，不能合并为“分数完全来自 RNA”或“病理是噪声”的结论；旧 NLL 漏洞进一步限制训练归因。
2. 冻结交互关闭后的 C 差为 −0.0047，尚未报告相应的配对不确定性或训练重复结果。
   单次 null 的峰值 0.6363 不是该差值的标准误，不能据此判定差异可忽略、显著或不显著。
   匹配熵 0.633 和最大权重 0.046 需要明确归一化基数、聚合方式及来源；这些摘要不足以断言未选择关键区域。
3. 当前是 77 人 / 27 事件的验证集，且用于选轮。尚未量化比较的不确定性，不能称 0.01 的差异没有意义，
   也不能预先承诺某个 OOF 标准误。单次标签置乱不能证明不存在标签、预处理或身份相关泄漏。
4. 当前记录不足以判定 BLCA 不适合评价病理增量，也不足以据此指定其他队列更合适。
   先修复并验证现有协议、补齐运行来源；本轮不自动启动其他队列或五折。

### 13.3 交互分支的量级说明

| 量 | 数值 | 含义 |
|---|---|---|
| 交互与基础项的幅度比 | 1.948 | 历史记录；范数和“比值均值/均值之比”定义待核对 |
| 缩放后的相对比值 `0.2 × 1.948` | 0.390 | 无量纲比值，不是交互 logit 的绝对量或解释贡献率 |
| base logits 绝对均值 | 2.86 | — |

历史文本另记录 raw 交互 5.34 与 base 2.86。若前者确为同一运行、同一范数的均值，
缩放后的均值应为 `0.2 × 5.34 = 1.068`；`5.34 / 2.86 ≈ 1.8671`，
不等于 1.948，可能混用了“比值均值”和“均值之比”。在补齐诊断 JSON 路径、checkpoint SHA、
范数及聚合定义前，不将这组数值用于该 parity run 的分支归因，也不据 raw 幅度宣称交互主导。

## 14. 辅助损失对照协议（2026-11-13）

前面的旧模型训练对照记录为 `--disable-auxiliary`，即 **11 项辅助损失权重全为 0、
`common_mode=off`**；它们不能代表旧完整辅助目标方法。本节保留辅助开关的历史比较，
但这些旧训练同样受离散 NLL 广播漏洞影响，不能作为修复后方法有效性的证据。

### 14.1 关键约束：批次大小是混淆变量

`minimum_batch_size` 给出秩下界：开启 `global_path_within` /`global_rna_within`
时要求 `patient_batch_size >= hidden + 1 = 17`。而参考运行用的是 batch 8。
因此**不能**拿"辅助开启"的结果直接和"batch 8 的参考运行"比。

对照设计（两个 arm 完全对称）：

| arm | 辅助损失 | batch | 其余一切 |
|---|---|---|---|
| `aux_on` | 本仓库方法的默认权重，11 项全开 | 17 | 相同 |
| `aux_off` | 全部权重 0，`common_mode=off` | 17 | 相同 |

以上是对称设计的目标，不等于历史运行已证明满足。303 名训练患者在 batch17 下余 14 人，
若两臂按不同最小批次下界合并尾批，实际患者分组和更新次数可能不同。
必须核对运行时源码、实际 batch 列表、初始化和采样，才能判断是否只改变辅助目标。

**修复后新批次入口（本轮不执行）**：脚本现已显式使用历史架构、共同的
`--patient-batch-size 17 --batch-tail-minimum 17` 和 `PYTHONHASHSEED=0`。
它们写入新独立目录，不能用于宣称重现下方旧 NLL 的历史分数。

```bash
bash scripts/run_aux_comparison.sh
```

可通过 `PARITY_INPUTS_ROOT` 指向明确的 UNI v1 输入目录（默认历史 `PARITY_DCT_FOLD0`），
通过 `PARITY_BATCH_OUT` 指定**尚不存在**的新批次目录；也可覆盖 `PARITY_PROJECT_ROOT`、
`PARITY_PYTHON`、`PARITY_RUN_ROOT`、`PARITY_DEVICE_0/1`。脚本拒绝已有批次、run 目录、日志和命令文件，
不删除旧产物，并逐 PID 检查运行状态。`run_parity_controls.sh` 使用同样保护；单次 null 分数不必恰等于随机水平。

### 14.2 诊断记录

trainer 现在按 epoch 汇总辅助项与通路诊断，写入 `report.json` 的
`history[*].auxiliary_terms` 与 `history[*].pathway_diagnostics`，
`controls.auxiliary_weight_overrides` 记录本次覆盖的权重。

### 14.3 冒烟测试已观察到的现象（2 轮，batch 17）

| 项 | 目标对角 | 实测对角 | 解读 |
|---|---|---|---|
| `pathway_path_within` | 1 | 0.983 | 所测对应坐标的视图线性相关较高 |
| `pathway_rna_within` | 1 | 0.992 | 所测对应坐标的视图线性相关较高 |
| `pathway_common` | 1 | **0.002** | 平均共同对角接近 0；不能代替完整矩阵检查 |
| `pathway_unique` | 0 | -0.003 | 平均独有对角接近 0；不证明正交或独立 |

这些是旧训练早期的相关统计，不足以判定共同/独有信息语义是否学到。
匹配归一化熵记录为从 0.633 到 0.91，但比较来源、对应 epoch 与汇总定义尚需核对，不能直接归因为辅助项。

### 14.4 历史对照结果（30 轮，保留原始数字）

| arm | 辅助项 | batch | 均值 C | 峰值 C | 最佳轮 | 匹配熵 |
|---|---|---|---|---|---|---|
| `aux_off` | 全 0 | 17 | 0.6637 | **0.7219** | 17 | 0.692 |
| `aux_on` | 11 项全开 | 17 | 0.6605 | 0.7052 | 17 | 0.775 |
| 参考运行 | 全 0 | 8 | 0.6929 | 0.7361 | 20 | 0.633 |

**开关对应关系：0.7219 是 aux-off/batch17，0.7052 是 aux-on/batch17。**
同一个 0.7219 同时用于辅助开关与 batch 对比，不是两条重复运行。按表中历史记录，
aux-on 相对 aux-off 的峰值低 0.0167，所报均值低 0.0032；未观察到此次辅助开启后的验证 C 提升。
这不能证明辅助目标有害、无效或等效，也不能用单次 null 的峰值宣称差异属于噪声或统计不显著。
“均值 C”需核实是否为 30 个 epoch 的平均值；这些 epoch 不是独立重复。

历史 aux-off/batch8 为 0.7361，新 aux-off/batch17 为 0.7219，点估计差为 −0.0142。
目前缺少新两臂的完整目录、实际命令和运行时源码，不能断言仅 batch 不同。
即使只改 batch 参数，固定 epoch 数也会改变 optimizer 更新次数等训练过程，
不能将其单独归因于批内统计；NLL 广播漏洞还使 batch 组成成为旧错误目标的一部分。
修复后的开关与 batch 比较必须另立新批次，不用这三个旧分数裁定正确 NLL 下的模块收益。

### 14.5 历史辅助目标诊断读数（不等于机制证据）

`aux_on` 最佳轮（第 17 轮）的通路诊断：

| 项 | 目标对角 | 实测对角 | 状态 |
|---|---|---|---|
| `pathway_path_within` | 1 | 0.9916 | 对应坐标视图相关较高，仍需非对角与秩检查 |
| `pathway_rna_within` | 1 | 0.9931 | 对应坐标视图相关较高，仍需非对角与秩检查 |
| `pathway_common` | 1 | **0.2235** | 平均正相关，距目标 1 尚远 |
| `pathway_unique` | 0 | 0.0016 | 平均线性相关较小，不能证明正交或独立 |
| `low_variance_coordinates` | — | 0.0 | 按给定阈值未标记低方差坐标，不能排除低秩 |

共同对角的历史记录从早期 0.002 到第 17 轮 0.2235。需先核对这两个统计是否来自可比运行，
再结合有效权重、跳过记录和逐项梯度判断辅助目标的作用；该趋势本身不证明共同语义或预测价值。

### 14.6 三个层次的总体判断

1. 所测 within 对角较高，但未同时报告完整相关矩阵与有效秩，不能证明整个解耦机制正常。
2. 共同/独有对角均值不能确立信息语义；均值可相互抵消，即便完整交叉相关为零也不证明非线性独立。
3. 历史 batch17 比较未出现辅助开启后的验证 C 提升，但 NLL 漏洞、运行对称性和不确定性仍需处理，
   不能据此裁定修复后机制的预测价值，也不能断言该队列分数由 RNA 决定。

### 14.7 归因前必须补齐的运行元数据

- 每条 run 的完整目录、运行时 commit/源码哈希、命令与 resolved config；历史架构必须明确标识。
- manifest/reference、genes/GMT 与特征清单哈希，患者/事件数、终点、分箱和指标版本。
- 实际有效辅助权重与跳过原因；初始化哈希、患者批次列表或摘要、尾批处理与每轮 optimizer 步数。
- patch/RNA 视图随机数设置及 `PYTHONHASHSEED`；同 seed 不等于实际输入与训练过程完全相同。
- 指标是选轮峰值、末轮、冻结干预还是重训练；诊断对应的 checkpoint SHA、epoch、范数和聚合定义。
- null 的置乱对象、种子及 time/event 配对规则；岭回归的输入、删失处理和调参范围。

以上信息缺失时保留“待核对”，不补写未验证的运行来源，不将旧分数包装为修复后结果。

## 15. 新增 CLI

`train_pathway_parity.py`：

- `--auxiliary-weight NAME=VALUE`（可重复）—— 覆盖单项辅助损失权重；
  与 `--disable-auxiliary` 互斥。
- `--common-mode {off,uniform,stability,shuffled}` —— 指定全局共同子空间的估计方式。
- `--batch-tail-minimum` —— 为对照预先固定共同的尾批下界；辅助比较两臂显式设为 17。

训练批次使用 `patient_batches`，尾批不足下界时合并而不是丢弃。
历史批次的实际分组仍需核对；新脚本的共同设置不能倒推旧运行已使用相同分组。

