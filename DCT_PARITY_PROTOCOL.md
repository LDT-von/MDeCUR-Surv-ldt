# DCT 对齐协议（DCT Parity Protocol）

本文件说明本仓库的 `train_pathway_parity.py` 采用的**参考协议**：切分、生存终点、
时间离散化、损失、patch 预算、RNA 预处理、计分口径与模型选择规则。
目标是让本仓库的**数据处理与计分口径**与参考实现 `DCT-Reg` 完全一致，
使得两边的数字可以放在同一张表里比较。

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
- 损失：**离散 NLL**（negative log-likelihood of the discrete survival），
  与参考实现的 `nll_loss` 一致；删失样本按删失位置累计权重

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

**指标时间网格**由参考实现的 `_select_valid_metric_time_grid` 同规则确定：
只有落在有效分箱内部的边界才作为评估时间点。
BLCA fold0 上的网格为 `[8.087582602, 15.12312194]`（2 个时间点）。

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
  --patient-batch-size 8 --patch-chunk-size 4096 --disable-auxiliary \
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
lr 5e-4 / weight_decay 5e-4 / seed 3，与参考实现的
`dct_v313_blca_uni_p4096_5fold_aligned` 设置逐项一致。

产物：`/data1/decur_runs/PARITY_DCT_FOLD0/run_fold0_30ep/`

### 与参考实现的对照（同一切分、同一 UNI v1、同一 p4096、同一 30 轮）

| 指标 | 本仓库（fold0 峰值 ep20） | 参考实现（fold0 峰值 ep7） |
|---|---|---|
| Harrell C-index | **0.7361** | 0.7266 |
| IPCW C-index | 0.5158 | 0.5235 |
| IBS | **0.1239** | 0.2086 |
| IAUC | **0.8313** | 0.7937 |
| 最终轮 C-index | 0.7076 | 0.6371 |

C-index 高 0.0095，IBS 低 0.0847，IAUC 高 0.0376。
末轮对比更有意义：本仓库末轮 0.7076 vs 参考实现末轮 0.6371，
说明**收敛稳定性明显更好**（参考实现从峰值 0.7266 掉到 0.6371，本仓库只从 0.7361 掉到 0.7076）。

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

fold 间波动很大（0.6573–0.7304），本仓库目前只跑了 fold0，
**单折领先不能说明整体领先**，需要补齐其余 4 折。

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
- 未修改训练实现与 loss 系数

## 13. fold0 对照组（2026-11-13）

在同一协议（同一清单、同一分箱、同一 patch 预算、同一 seed 3、同一按 val
C-index 峰值选轮）下运行的一组对照，用来回答"0.7361 这个分数是谁贡献的"。
全部产物在 `/data1/decur_runs/PARITY_DCT_FOLD0/`。

### 13.1 结果

| 对照 | 设置 | 均值 C | 峰值 C | 说明 |
|---|---|---|---|---|
| 全模型 | `--disable-auxiliary`，batch 8 | 0.6929 | **0.7361** | 参考运行 |
| 推理时病理置零 | 同一 checkpoint，推理期病理置零 | — | **0.7686** | 病理是噪声 |
| 推理时 RNA 置零 | 同一 checkpoint，推理期 RNA 置零 | — | 0.5531 | 分数完全依赖 RNA |
| `rna_only` | 训练+推理都去掉病理 | 0.6938 | 0.7639 | 无病理更差 |
| `null` | 训练标签置乱 | 0.5648 | 0.6363 | 无标签泄漏，但 null 也能刷到 0.64 |
| `path_only` | 训练+推理都去掉 RNA | 0.5585 | 0.6006 | 纯病理接近随机 |
| raw RNA 岭回归 | 17025 维原始表达直接岭回归 | — | **0.7472** | 无深度模型即接近全模型 |
| 协同交互关掉 | 同一 checkpoint，`interaction_strength=0` | — | 0.7314 | 协同分支贡献 ≈ 0.005 |

### 13.2 结论（重要）

1. **0.7361 的高分主要来自 TCGA-BLCA 的 RNA 预后信号**，不是病理分支，
   也不是通路解耦模块。raw RNA 岭回归已达 0.7472，去掉病理反而升到 0.7686。
2. **协同交互分支没有可测量的预测增量**：关掉后只掉 0.005，77 人验证集上
   远小于噪声（同一协议下 null 对照都能到 0.6363）。
   匹配锐度也支持这一读数：归一化匹配熵 0.633、最大匹配权重 0.046，
   即软匹配是弥散的，没有"挑出少数关键区域"的行为。
3. **统计功效不足**：val 只有 77 人 / 27 事件，且 checkpoint 按 val C-index
   峰值选轮。0.01 量级的差异在此协议下没有意义。
4. BLCA 队列 RNA 信号太强，**不适合**用来评估病理模块的增量贡献；
   评估解耦模块需要 5 折 OOF（SE≈0.011）和 RNA 预后更弱的队列（BRCA/GBM）。

### 13.3 交互分支的量级说明

| 量 | 数值 | 含义 |
|---|---|---|
| `|interaction| / |base|` | 1.948 | 交互项原始幅度约为 base logits 的 2 倍 |
| 实际贡献 `0.2 × |interaction|` | 0.390 | 乘以 `interaction_strength=0.2` 后的净贡献 |
| base logits 绝对均值 | 2.86 | — |

**可比量是 logit 跨度占比，不是 raw 交互幅值**。raw 交互 5.34 与 base 2.86
不在同一尺度上（一个未缩放、一个已缩放），不能直接相除后宣称"交互主导"。

## 14. 辅助损失对照协议（2026-11-13）

前面的所有对照都用 `--disable-auxiliary`，也就是 **11 项解耦辅助损失权重全为 0、
`common_mode=off`**。这意味着"解耦模块是否有效"这个问题**从未被检验过**。
本节记录补上这一对照的协议。

### 14.1 关键约束：批次大小是混淆变量

`minimum_batch_size` 给出秩下界：开启 `global_path_within` /`global_rna_within`
时要求 `patient_batch_size >= hidden + 1 = 17`。而参考运行用的是 batch 8。
因此**不能**拿"辅助开启"的结果直接和"batch 8 的参考运行"比。

对照设计（两个 arm 完全对称）：

| arm | 辅助损失 | batch | 其余一切 |
|---|---|---|---|
| `aux_on` | 本仓库方法的默认权重，11 项全开 | 17 | 相同 |
| `aux_off` | 全部权重 0，`common_mode=off` | 17 | 相同 |

批次大小、清单、分箱、patch 预算、lr/weight-decay/seed、选轮规则全部相同，
**唯一差别是辅助项是否贡献梯度**。

运行：

```bash
bash scripts/run_aux_comparison.sh
```

### 14.2 诊断记录

trainer 现在按 epoch 汇总辅助项与通路诊断，写入 `report.json` 的
`history[*].auxiliary_terms` 与 `history[*].pathway_diagnostics`，
`controls.auxiliary_weight_overrides` 记录本次覆盖的权重。

### 14.3 冒烟测试已观察到的现象（2 轮，batch 17）

| 项 | 目标对角 | 实测对角 | 解读 |
|---|---|---|---|
| `pathway_path_within` | 1 | 0.983 | 视图自洽达成 |
| `pathway_rna_within` | 1 | 0.992 | 视图自洽达成 |
| `pathway_common` | 1 | **0.002** | 共同子空间几乎不相关 |
| `pathway_unique` | 0 | -0.003 | 独有子空间已正交 |

即：**视图一致性成立，但"病理–RNA 共同信息"这一步没有学到**。
同时匹配归一化熵从 0.633 升到 0.91，辅助训练让软匹配更弥散。

### 14.4 对照结果（30 轮，两臂对称）

| arm | 辅助项 | batch | 均值 C | 峰值 C | 最佳轮 | 匹配熵 |
|---|---|---|---|---|---|---|
| `aux_off` | 全 0 | 17 | 0.6637 | **0.7219** | 17 | 0.692 |
| `aux_on` | 11 项全开 | 17 | 0.6605 | 0.7052 | 17 | 0.775 |
| 参考运行 | 全 0 | 8 | 0.6929 | 0.7361 | 20 | 0.633 |

**结论：打开 11 项解耦辅助损失没有提高分数，反而略降。**
峰值差 0.0167（0.7219 → 0.7052），均值差 0.0032；
在 77 人 / 27 事件的验证集上，这个差值处于噪声区（同一协议下 null 对照
都能到 0.6363），因此**不能声称辅助损失有害，只能声称它没有可测量的增益**。

同时暴露了一个此前被忽略的效应：**批次大小本身影响很大**。
`aux_off` 只把 batch 从 8 改成 17，峰值就从 0.7361 掉到 0.7219（-0.0142）。
也就是说参考运行的 0.7361 部分来自 batch 8 这个配置，而不是任何模块。

### 14.5 辅助项确实生效了（诊断读数）

`aux_on` 最佳轮（第 17 轮）的通路诊断：

| 项 | 目标对角 | 实测对角 | 状态 |
|---|---|---|---|
| `pathway_path_within` | 1 | 0.9916 | ✅ 视图自洽 |
| `pathway_rna_within` | 1 | 0.9931 | ✅ 视图自洽 |
| `pathway_common` | 1 | **0.2235** | ⚠️ 共同子空间仍近乎不相关 |
| `pathway_unique` | 0 | 0.0016 | ✅ 独有子空间正交 |
| `low_variance_coordinates` | — | 0.0 | ✅ 无退化坐标 |

对照冒烟测试的 0.002，共同对角从 0.002 升到 0.2235 —— 解耦目标**在把
共同子空间往正确方向推**，但 30 轮后仍未接近目标值 1.0。

### 14.6 三个层次的总体判断

1. **视图一致性（`within`）有效**：对角 0.99+，这部分目标达成了。
2. **跨模态共同/独有分解没达成**：共同对角 0.2235、目标 1.0；
   独有对角 0.0016 达标（但这可能只是"根本没学"而非"学会了正交"）。
3. **即使目标部分达成，也不带来预测增益**：开着辅助比关着低 0.0167。

结论链条是完整的：**解耦机制本身没坏（within 项证明它能学），
但它在当前 BLCA fold0 协议下不产生可测量的预测价值。**
这与 13.2 的结论一致 —— 这个队列的分数由 RNA 决定。

## 15. 新增 CLI

`train_pathway_parity.py`：

- `--auxiliary-weight NAME=VALUE`（可重复）—— 覆盖单项辅助损失权重；
  与 `--disable-auxiliary` 互斥。
- `--common-mode {off,uniform,stability,shuffled}` —— 指定全局共同子空间的估计方式。

训练批次改用 `patient_batches`，尾批次不足秩下界时**合并**而不是丢弃。

