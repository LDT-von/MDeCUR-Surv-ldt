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
