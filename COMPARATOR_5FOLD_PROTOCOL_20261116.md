# 五折解耦方法对照协议（2026-11-16）

> **2026-11-16 折数收敛为 1 折**：运行调度已削减为只跑 fold 0，协议标识改为
> `blca_1fold_dss_uni_v1_p4096_dnll_v1`，驱动版本改为
> `onefold_comparison_v1_single_fold`。代码细则、单折运行命令与"单折不能宣称
> 优劣"的限制声明见
> [FIVEFOLD_COMPARISON_PROTOCOL_20261116.md](FIVEFOLD_COMPARISON_PROTOCOL_20261116.md)
> 文首与 §4.3。

> 2026-11-16 文件精简：当前仅保留 PIBD/CAMR/DeReF/DIMAF 四个多模态对照，RNA-only 与 WSI-only 已删除。驱动版本为 `onefold_comparison_v1_single_fold`，须重新生成plan；下方历史单模态说明不再是运行入口。DCT仍只作独立标注的历史归档。当前模型目录以 [README.md](README.md) 为准。

本文件说明 **PIBD / CAMR / DeReF / DIMAF** 四个解耦方法在 **BLCA 生存预测**上
与 **DCT 参考、RNA-only 岭回归、WSI-only** 的五折对照：源码拉取、协议对齐、
模型适配、训练复现、分数比较的完整做法，以及**防泄漏边界**和**结果解读限制**。

> **本轮状态**：源码已拉取并固定 commit；协议层、四个适配器、两个基线、五折
> 运行器和汇总器已实现；22 项 CPU 合成/回归检查通过（含 8 项脚本静态安全检查）。
> **尚无真实患者训练结果、无 CUDA 运行证据**。真实五折必须在服务器上跑
> `scripts/run_5fold_comparison.sh`，本窗口不 SSH、不启动患者任务、不触碰正在运行的训练队列。
>
> **2026-11-16 修正**：补丁抽样种子不再依赖 Python 哈希随机化（改为 SHA-256 派生），
> 详见 `FIVEFOLD_COMPARISON_PROTOCOL_20261116.md` §2.3。

---

## 1. 源码拉取（只读、固定 commit）

四个官方仓库克隆到 `papers/comparators/`，记录 URL、commit 和**逐文件 SHA256**。
清单与校验逻辑在 `comparison_sources.py` / `comparison_sources.json`。

| 方法 | 仓库 | commit | 校验文件数 |
|---|---|---|---|
| PIBD | `https://github.com/zylbuaa/PIBD` | `bd5bd94e6f8d48e7679c6e68209a3a65c9e56a78` | 5 |
| CAMR | `https://github.com/wxq-ustc/CAMR` | `8fbae0a0bc4e9da31547b5eab5d773e6b2049e93` | 3 |
| DeReF | `https://github.com/ZJUMAI/DeReF` | `6bdbf81e76d1c7fa1d49b05f1a8fec2fbfc7b6d3` | 5 |
| DIMAF | `https://github.com/Trustworthy-AI-UU-NKI/DIMAF` | `3fe81236aafac88977b730745314b8c9906cb4d3` | 8 |

**第三方文件永不在磁盘上被修改。** 每次 `plan` 都会先跑 `verify_upstream()`；
任何文件被改动都会让计划重建失败，从而拒绝训练。

上游的两类硬编码限制用**运行时替换**解决，而不是改文件：

| 上游限制 | 处理方式 | 记录位置 |
|---|---|---|
| PIBD `PoE.gaussian_noise` / `PIB.gaussian_noise` 无条件 `.cuda()` | `device_agnostic()` 换成语义相同的显式设备采样器 | `comparison_sources.py` |
| PIBD `PIB.forward` 内 `.cuda()`（proxy 索引、attention mask、arange） | `_device_neutral_pib_forward()`，**只改三处设备调用**，公式/索引/阈值/返回值不变 | `comparison_pibd.py` `PIB_FORWARD_DIFF` |
| DeReF `Transformer_P/G.forward` class token `.cuda()` | `_device_neutral_transformer_p/_g()` | `comparison_deref.py` |
| DIMAF `DirNIWNet` 需要 `np.ndarray` 原型；PANTHER 输出宽度为 `d+1` | 传 numpy 数组；`histo_dim = 1024 + 1` | `comparison_dimaf.py` |

每个适配器的 `adaptation_metadata()` 都带 `runtime_diffs` 与 `adaptations` 字段，
写进报告，可审计。

**上游目录保持干净**：上游加载器在导入期间关闭 Python 字节码写入
（`sys.dont_write_bytecode`），因此 `papers/comparators/` 四个 checkout 不会因为
导入而产生 `__pycache__`。`test_comparison_5fold.py` 每次都检查这一点；
四个仓库的 `git status` 当前均为 clean。

---

## 2. 协议对齐（一个协议对象，四个多模态臂）

所有臂共用 `comparison_protocol.py` + `comparison_5fold.py`：

| 项目 | 取值 |
|---|---|
| 终点 / 单位 | DSS / months |
| 分箱 | 训练集未删失时间 `pd.qcut(q=4)`，外侧 `±inf`，4 箱 |
| 损失 | 逐患者 clamped 离散 NLL，`alpha_surv = 0.5`（`patientwise_clamped_discrete_v1`） |
| 风险 | `risk = -sum_j S_j` |
| patch 预算 | 4096 行；训练无放回随机重抽，评估等间隔确定性抽取，不足显式补零 |
| RNA 预处理 | `none`（原始表达直接输入），统计量仅用训练行 |
| 划分 | manifest 自带的 train/val，**outer test 永不打开** |
| 选轮 | 验证 Harrell C 峰值，同分取最早轮 |
| 指标 | `sksurv`：`concordance_index_censored` / `concordance_index_ipcw` / `brier_score` / `integrated_brier_score` / `cumulative_dynamic_auc`，删失分布只用训练行拟合 |
| 超参 | 30 轮、lr 5e-4、weight_decay 5e-4、seed 3、patient_batch 20、CosineAnnealingLR |

**四个多模态臂各自的"原始创新模块"确实被执行**（不是重写）：

| 臂 | 执行的上游模块 | 适配点 |
|---|---|---|
| `pibd_2m` | `PIB`、`PoE`、`MITransformerLayer`/`MIAttention`、`MIEstimator`/CLUB、`SNN_Block`、`NLLSurvLoss` | 训练循环换成共享离散 NLL；上游 proxy/KL/CLUB 保留为辅助项；每轮随机 WSI bag 取自固定预算行 |
| `camr_2m` | `networks.py`/`fusion.py`/`utils.py` 的共享-私有编码、跨模态门、低秩融合、重构、`DiffLoss` | 三模态降为病理/RNA（**不伪造 CNA**）；两路低秩融合；加 attention MIL 与 RNA 投影；Cox 头换离散 hazard logits；判别器用显式 D-only + encoder 混淆到 0.5 的交替目标 |
| `deref_2m` | `Transformer_P`/`Transformer_G`、`Private_encoder`、`Share_encoder`、`Gate`、`expert`、上游 Nyström attention、BilinearFusion | 6 个 omic 组来自上游 `csv/signatures.csv` 六个功能家族（精确基因名匹配）；上游随机特征重组 `seg_length` 保留；上游网络逐患者执行（其原生 batch=1） |
| `dimaf_2m` | `MultiSNN`/`SNN_Block`、`CrossAttentionLayer`、`DIMAF`、`PANTHERBase`/`DirNIWNet`/`mog_eval`、`DistanceCorrelationLoss`/`OrthogonalLoss`/`HSICLoss`、上游 `kmeans_clustering` | 原型在**当前折训练 patch** 上聚类（不用上游 pickle）；slide embedding 走协议 bag 而非上游 H5 loader；上游解耦损失保留为辅助项 |

**两个单模态基线**（`comparison_baselines.py`）：

- `rna_ridge`：**RNA-only** 岭回归。线性 hazard score `wᵀx + b` 按训练行拟合，
  `alpha` 用**训练行**在同一离散 NLL 下选择。**不复用**历史上那个拟合"观测随访箱
  one-hot"的岭回归（那不是删失感知的生存模型）。
- `wsi_only`：**WSI-only** attention-MIL + 同一离散 hazard 头。RNA 分支是**移除**
  而不是补零，避免常量输入泄漏。

**DCT 参考**：`reference/dct_5fold_archive.json` 归档参考实现自己的五折 legacy-val
数字（来自 `DCT_PARITY_PROTOCOL.md` §11）。它**只被读取，不被重跑**，但用**同一条
mean±std 规则**进入同一张表：

| 指标 | mean | std | 逐折 |
|---|---|---|---|
| C-index | 0.7005 | 0.0333 | 0.7266 / 0.7259 / 0.6573 / 0.6623 / 0.7304 |
| IPCW C | 0.6344 | 0.0579 | 0.5235 / 0.6631 / 0.6649 / 0.6342 / 0.6864 |
| IBS | 0.3237 | 0.2345 | 0.2086 / 0.1540 / 0.7891 / 0.2300 / 0.2369 |
| iAUC | 0.6174 | 0.1532 | 0.7937 / 0.7018 / 0.6852 / 0.3518 / 0.5545 |

---

## 3. 防泄漏边界

所有由数据估计的量，**只在当前折的训练行上拟合**：

| 量 | 拟合来源 | 代码位置 |
|---|---|---|
| 生存分箱边界 | 训练行未删失时间 | `comparison_protocol.load_fold` → `dct_bin_edges` |
| RNA 标准化统计 | 训练行 | `cancer_rna.fit_rna_scaler` |
| DIMAF 病理原型 | 当前折训练 patch 特征 | `comparison_dimaf.fit_prototypes` |
| RNA 岭回归 alpha | 训练行（同一离散 NLL） | `comparison_baselines.RNAOnlyRidge.fit` |
| DeReF 六个 omic 组 | 基因名与 RNA schema 匹配（**不看表达值、不看标签**） | `comparison_5fold.load_deref_groups` |
| 通路分组 | GMT 与 RNA schema 匹配（**不看表达值、不看标签**） | `cancer_pathway.read_pathway_mapping` |

**outer test 的拒绝发生在打开任何输入文件之前**：`comparison_protocol._reject_test_rows`
先读 manifest 文本，发现 `split` 不是 `train`/`val` 立刻抛错；`load_fold` 的
`include_test` 参数恒为 false。

**计划锁定**：`plan` 把 manifest、survival reference、genes、GMT、DeReF signatures、
上游文件哈希、本仓源码哈希、运行时版本全部写进 JSON。`run` 会**从输入重建计划**并
与存档逐字节比对，不一致就拒绝训练（`Plan no longer matches ...`）。

**校验记录**：每折报告里 `fold_metadata` 记录患者/事件数、分箱边界、
患者顺序哈希、标签哈希、输入文件 SHA256 和 `fit_boundary` 说明。

---

## 4. 训练复现与分数比较

```bash
# 单折计划
python comparison_5fold.py plan --fold 0 \
  --manifest .../manifest.csv --survival-reference .../survival_reference.csv \
  --genes .../genes.txt --gmt .../combine_signatures.gmt \
  --deref-signatures .../deref_signatures.csv --device cuda --out plan_f0.json

# 单臂单折训练
python comparison_5fold.py run --plan plan_f0.json --plan-sha256 <sha> \
  --model pibd_2m --device cuda --out run_f0_pibd

# 汇总（可选带上归档的 DCT 参考）
python comparison_5fold.py summary --runs run_f*/ --dct-reference reference/dct_5fold_archive.json \
  --out summary.json
```

服务器批量脚本：`scripts/run_5fold_comparison.sh`。它**新建**批次目录、
拒绝覆盖任何已有 plan/run/log/summary、逐条写命令文件、**不触碰正在运行的训练队列**、
不加载 outer test。

汇总输出（`summary.json`）结构：

```
rows[]: {model, metric: c_index|ibs|iauc, mean, std, per_fold[5], folds[5]}
models[]: pibd_2m, camr_2m, deref_2m, dimaf_2m, rna_ridge, wsi_only, dct_reference
```

汇总器**拒绝**不完整折、重复折、跨多个 plan、以及 smoke / test-evaluated 运行。

---

## 5. 结果解读限制（重要）

1. **验证选轮 = 乐观有偏。** `val` 是 legacy validation 集，每折报告的 `best_validation`
   是按验证 C 峰值选出来的轮次。DCT 参考归档数字同样如此，两者偏置方向一致，
   但**都不是独立测试估计**。参考实现自己的 outer-test 五折数字属于**另一套评估协议**，
   不能和本表直接比。
2. **五折均值不是独立测试估计。** 折之间共享同一批患者的划分来源。
3. **不做显著性检验。** 本汇总只给 mean ± std（总体标准差）。任何"解耦方法优于
   单模态基线"的判断需要配对的不确定性，**当前没有**。
4. **IBS/iAUC 处在历史时间网格上。** 该网格被审计为生存列与时间边界错位；
   它可以用于**同一口径下的横向比较**，**不能**当作校准误差或动态区分度的正确语义读数。
   Harrell C / IPCW C 不经过这套网格，其逆序不能由网格缺陷解释。
5. **适配版不是原论文复现。** 四个多模态臂的 `official_model_reproduction` 均为
   `false`；上游创新模块被执行，但训练循环、生存头和输入契约有明确适配。
   `CAMR` 是两模态而非三模态，`DeReF`/`DIMAF` 的输入契约与原仓库不同。
6. **共同/独有不等于预后有用/无用。** 解耦结构成立不等于预测提升；
   稳定度不等于临床可靠性。
7. **合成检查不等于患者效果。** 当前只有 22 项 CPU 合成/回归与静态检查通过，
   **没有任何真实患者结果**。

---

## 6. 本轮未做的事

- 未 SSH、未在服务器上跑真实五折、未启动任何患者训练任务。
- 未触碰 outer test。
- 未修改 `papers/comparators/` 下的任何第三方文件（每次 plan 都做 SHA256 校验）。
- 未改动正在运行的训练队列；新脚本写入全新批次目录。
- 未实现 MurreNet / MoHD：无可核验的作者实现，**不建占位模型**。
- 未把历史单折（fold0）数字当作五折结果。

---

## 7. 文件索引

| 文件 | 作用 |
|---|---|
| `comparison_sources.py` / `.json` | 上游仓库清单、SHA256 校验、运行时加载器、设备重定向 |
| `comparison_protocol.py` | 共享五折协议层：加载、分箱、RNA scaler、损失、指标、拟合边界 |
| `comparison_pibd.py` | PIBD 适配器（`PIB`/`PoE`/`PID`/CLUB） |
| `comparison_camr.py` | CAMR 适配器（共享-私有、低秩融合、对抗门、`DiffLoss`） |
| `comparison_deref.py` | DeReF 适配器（4 路解耦、MoE 门控、特征距离 MSE） |
| `comparison_dimaf.py` | DIMAF 适配器（PANTHER 原型 + 4 路交叉注意力 + 距离相关损失） |
| `comparison_baselines.py` | RNA-only 岭回归、WSI-only MIL、DCT 归档加载 |
| `comparison_5fold.py` | 五折 `plan` / `run` / `summary` 运行器 |
| `comparison_synthetic.py` | 合成队列生成（形状与真实一致，供 CPU 端到端测试） |
| `test_comparison_5fold.py` | 14 项合成/回归检查（含上游字节级完整性与"不留缓存"检查） |
| `test_comparison_5fold_shell.py` | 8 项脚本静态安全检查（不启动脚本） |
| `reference/dct_5fold_archive.json` | 归档的 DCT 五折参考数字（只读） |
| `scripts/run_5fold_comparison.sh` | 服务器五折批量脚本 |
