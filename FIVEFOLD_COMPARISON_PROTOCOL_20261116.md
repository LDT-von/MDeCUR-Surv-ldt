# 五折统一对比协议与运行手册（2026-11-16）

本文件记录 **PIBD / CAMR / DeReF / DIMAF** 四个有官方源码的方法，与 **RNA-only 岭回归**、
**WSI-only** 两个单模态基线，在**同一 BLCA DSS 生存预测协议**下的适配、训练复现与分数比较方案。

MurreNet 与 MoHD **没有已验证的官方源码**，因此不提供占位模型（见文末"不可用方法"）。

---

## 1. 当前状态总览

| 方法 | 官方源码 | 上游 commit | 适配器 | 上游创新模块是否执行 | CPU 合成回归 | 真实患者训练 |
|---|---|---|---|---|---|---|
| PIBD | 已公开 `zylbuaa/PIBD` | `bd5bd94` | `comparison_pibd.py:PIBD2M` | 是（PIB/PoE/PID/CLUB/SNN） | 通过 | 未做 |
| CAMR | 已公开 `wxq-ustc/CAMR` | `8fbae0a` | `comparison_camr.py:CAMR2M` | 部分（见 §3.1） | 通过 | 未做 |
| DeReF | 已公开 `ZJUMAI/DeReF` | `6bdbf81` | `comparison_deref.py:DeReF2M` | 是（Transformer_P/G、Gate、MoE） | 通过 | 未做 |
| DIMAF | 已公开 `Trustworthy-AI-UU-NKI/DIMAF` | `3fe8123` | `comparison_dimaf.py:DIMAF2M` | 是（PANTHER、MultiSNN、CrossAttention） | 通过 | 未做 |
| RNA-only Ridge | — | — | `comparison_baselines.py:RNAOnlyRidge` | 非上游方法 | 通过 | 未做 |
| WSI-only MIL | — | — | `comparison_baselines.py:WSIOnlyMIL` | 非上游方法 | 通过 | 未做 |
| DCT（参考臂） | — | — | `comparison_fold0.py`（已有） | 项目自有 | 通过 | 已有历史运行 |
| MurreNet | **未找到** | — | 无 | — | — | — |
| MoHD | **未找到** | — | 无 | — | — | — |

**已验证**：`python -m pytest test_comparison_5fold.py test_comparison_camr.py test_comparison_fold0.py test_cancer_parity.py test_cancer_parity_bag.py` → **61 passed**（纯 CPU 合成数据）。
全仓库套件 `python -m pytest -q` → **315 passed, 3 skipped**（哈希随机化开启状态下验证）。

**未验证**：本会话**没有任何真实患者 WSI+UNI 提取、没有 CUDA 运行、没有临床效果证据**。合成检查通过不等于患者效果。

---

## 2. 统一协议（DCT Parity Protocol）

所有臂共享以下常量，定义在 `comparison_protocol.py`：

| 项 | 值 |
|---|---|
| 终点 | DSS（疾病特异性生存） |
| 时间单位 | 月 |
| 分箱 | 4 个分位箱，边界由**当前折训练行未删失时间**拟合 |
| 似然 | 患者级离散 NLL，`alpha=0.5`，`discrete_nll` |
| 风险 | `risk = -sum(S)`，S 为离散生存矩阵 |
| 补丁预算 | 4096（含显式零填充） |
| 划分 | 5 折（fold 0–4），manifest 为权威，不被改写 |
| 评估 | `parity_metrics`（sksurv），训练集删失分布作为参照 |
| 选择规则 | 验证集 Harrell C 最大，并列时取最早 epoch |
| 外层测试 | **从不打开** |

### 2.1 防泄漏边界

所有从数据估计的量都**只在当前折的训练行上拟合**：

- 生存分箱边界 → 仅训练行未删失时间
- RNA 标准化统计量 → 仅训练行
- DIMAF 病理原型 → 仅训练折的 patch 特征（上游 `kmeans_clustering`）
- RNA 岭回归 alpha → 仅训练行，按同一离散 NLL 选择
- DeReF 六个功能家族 → 仅使用基因名与 RNA 顺序 schema（不含表达值、不含标签）

验证行只被评分。`comparison_protocol.load_fold` 在打开任何输入文件**之前**就拒绝含
`test` 行的 manifest（`_reject_test_rows`）。

### 2.2 计划锁定（plan）

`comparison_5fold.py plan` 冻结一折的全部输入、上游源码 SHA256、代码 SHA256、协议常量与
运行时版本，写成不可变 JSON。`run` 会**重新构建**该计划并逐字段比对，任何输入、代码、
协议或运行时变化都会让训练**拒绝启动**。这保证"分数差异只能来自模型本身"。

### 2.3 采样确定性（2026-11-16 修正）

补丁抽样种子过去由 `hash((seed, patient_id))` 导出。`PYTHONHASHSEED` **只在解释器启动前**
才生效，运行时设置环境变量不会关闭哈希随机化，因此该抽样**在不同进程间会变化**，
而旧守卫读的是运行时环境变量，给出的是**假保证**。这曾表现为测试套件的偶发失败。

修正：抽样种子改由 SHA-256 摘要导出（`cancer_parity._stable_seed`），**不再依赖 Python 哈希
状态**，跨进程、跨机器、跨 Python 版本稳定。计划中记录 `hash_seed_state`
（环境变量值 + `sys.flags.hash_randomization` 实际状态 + `sampler_depends_on_python_hash: false`），
报告不再声称它未真正拥有的确定性。

`PYTHONHASHSEED=0` 仍建议保留（shell 脚本已导出），但它现在只是额外保险，**不再是正确性的前提**。

---

## 3. 各方法的适配说明（重要：请勿当作原文复现）

四个适配器的标签都带 `2M-P4096-DNLL` 后缀，表示：**这是适配，不是原文复现**。

### 3.1 CAMR
上游 CAMR 是"病理 + 基因表达 + CNA → 三个 80 维患者向量 → 标量 Cox 输出"。
`CAMR2M` **直接调用上游未修改的类**：`networks.MisaLMFGatedRec`（共享编码器、三私有编码器、
双线性跨模态门、低秩融合、生存编码器、三个重建头）、`fusion.Discriminator`、
`utils.DiffLoss`、`utils.MSE`。

但**融合前的输入适配与生存头仍为本项目重实现**，因此是适配而非原文复现
（`official_model_reproduction: false`）：

1. 三模态降为两模态。BLCA 协议无 CNA 输入，第三个槽位填入**不含任何患者信息的常量**
   （上游低秩融合要乘三个因子，不能简单置零），**不伪造**表达、CNA 或生存值。
2. 病理经注意力 MIL 汇聚到 80 维、RNA 投影到 80 维，这两步在**上游模块之外**。
3. 上游有界 Cox 输出换成无界离散风险 logits，以统一离散 NLL 打分；
   上游编码器、门控、低秩融合与重建头不变。
4. 上游把判别器排除在 optimizer 之外（其对抗项实际从不更新任何权重）；
   本项目给判别器独立 optimizer 并显式更新。
5. 上游第三个"fake"判别项因常量槽位无患者信息而剔除。

**这是四个方法中上游执行程度相对最低的一个**（生存头与输入适配为重实现），
结论中必须单列说明。

### 3.2 PIBD
上游 `PIB.forward` 与高斯采样器硬编码 `.cuda()`。为保持 `papers/comparators/` 原样，
运行时安装**设备中立副本**（仅三处设备调用差异，见 `PIB_FORWARD_DIFF`），磁盘文件不变。
上游 proxy / KL / CLUB 目标作为**辅助项**保留，主损失换为共享离散 NLL。

### 3.3 DeReF
上游网络为 batch size 1 编写，基因组分支的 token 堆叠方式在适配器中按上游意图
（`[1, 6, D]`）喂入；两个 Transformer 的 class token `.cuda()` 改为设备中立。
上游随机特征重组（`seg_length` 选择）保留，**每个 forward 仍是随机的**。

### 3.4 DIMAF
上游原型文件被替换为**当前折训练行聚类**得到的原型（上游 `kmeans_clustering`），
上游 `PANTHERBase`/`DirNIWNet` 数学不变；slide embedding 步骤改为驱动协议 patch bag
而不是上游 H5 dataloader；上游解耦损失（distcor/HSIC/orthogonal）作为辅助项保留。

---

## 4. 服务器运行手册

### 4.1 前置条件

- 目标环境已有 `torch`、`numpy`、`sksurv`、`pandas`（`comparison_sources.py` 只做哈希校验，不安装任何东西）
- `papers/comparators/{PIBD,CAMR,DeReF,DIMAF}` 四个 checkout 存在且未被修改
- 输入：`manifest.csv`（含 `patient_id,split,wsi_pt,rna_npy,time,event`）、双精度 `survival_reference.csv`、
  有序基因名 `genes.txt`、GMT `signatures.gmt`、DeReF `csv/signatures.csv`

### 4.2 单折单臂

```bash
cd Conditional_DeCUR_Surv

# 1) 冻结计划
python comparison_5fold.py plan \
  --fold 0 \
  --manifest /path/manifest.csv \
  --survival-reference /path/survival_reference.csv \
  --genes /path/genes.txt \
  --gmt /path/combine_signatures.gmt \
  --deref-signatures ../papers/comparators/DeReF/csv/signatures.csv \
  --device cuda \
  --out runs/plan_fold0.json

# 2) 取计划哈希
PLAN_SHA=$(python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" runs/plan_fold0.json)

# 3) 训练一个臂
python comparison_5fold.py run \
  --plan runs/plan_fold0.json --plan-sha256 "$PLAN_SHA" \
  --model pibd_2m --device cuda \
  --out runs/fold0/pibd_2m
```

`--model` 取值：`pibd_2m`、`camr_2m`、`deref_2m`、`dimaf_2m`、`rna_ridge`、`wsi_only`。

冒烟检查用 `--smoke`（只跑 1 epoch）。**summary 明确拒绝 smoke 运行**，所以冒烟结果不会
污染最终表格。

### 4.3 五折全量

```bash
for F in 0 1 2 3 4; do
  python comparison_5fold.py plan --fold $F ... --out runs/plan_fold$F.json
  SHA=$(python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" runs/plan_fold$F.json)
  for M in pibd_2m camr_2m deref_2m dimaf_2m rna_ridge wsi_only; do
    python comparison_5fold.py run --plan runs/plan_fold$F.json --plan-sha256 "$SHA" \
      --model $M --device cuda --out runs/fold$F/$M
  done
done

python comparison_5fold.py summary --runs runs/fold*/* --out runs/summary.json
```

### 4.4 与 DCT 参考臂合并

DCT 臂的历史运行由 `comparison_fold0.py` 产生（协议 `blca_fold0_full_dense_patientwise_v3`，
与本协议同终点、同分箱数、同似然、同指标调用）。合并时须核对：分箱边界、样本数、
事件数、`loss_version`、`parity_metrics` 调用点一致；若不一致必须说明差异来源，
不能直接并列。

---

## 5. 结果输出与解读规则

`summary` 输出每个模型每个指标的：

- `mean` / `std`（总体标准差，5 折）
- `per_fold`：逐折 C-index
- `folds`：折号列表

指标方向：`c_index` 越高越好，`iauc` 越高越好，`ibs` 越低越好。

**解读约束（必须遵守）**：

1. 数字来自**验证集选出的 epoch**，不是独立外层测试估计；不能宣称临床性能。
2. **不做显著性检验**。均值±标准差不构成"优于"的证据。
3. "共同/独有结构" ≠ "预后有用/无用"；稳定度 ≠ 临床可靠性。
4. 四个适配器是**适配**，不是原文数字的复现；与论文表格直接对比无效。
5. `CAMR2M` 的上游执行程度最低，结论中必须单列说明。
6. MurreNet / MoHD 缺席，不得推断"这两个方法表现如何"。

---

## 6. 不可用方法（MurreNet、MoHD）

登记于 `comparison_sources.json` 的 `methods` 段，并在计划的 `unavailable_models` 中重复声明：

- **MurreNet**（MICCAI 2025）：会议页与 arXiv（2507.04891）均未提供官方仓库；作者主页
  `mingxinliu.top` 与作者 GitHub `lmxmercy` 下未找到可验证的官方实现。
- **MoHD**（Medical Image Analysis 2026-07，DOI `10.1016/j.media.2026.104098`）：未找到
  可验证的官方仓库；其所需的 patch/region 层次化特征契约在本协议下亦未确立。

**不提供占位实现、不做代理重实现、不在结果表中留空行暗示"已比较"。**

---

## 7. 相关文件

| 文件 | 作用 |
|---|---|
| `comparison_protocol.py` | 共享加载、分箱、损失、指标、折叠聚合 |
| `comparison_5fold.py` | 5 折 runner：`plan` / `run` / `summary` |
| `comparison_sources.py` / `.json` | 上游加载器、哈希登记、不可用方法台账 |
| `comparison_pibd.py` / `comparison_deref.py` / `comparison_dimaf.py` / `comparison_camr.py` | 各方法适配器 |
| `comparison_baselines.py` | RNA-only Ridge、WSI-only MIL |
| `test_comparison_5fold.py` | 合成 5 折回归测试（9 项） |
| `DCT_PARITY_PROTOCOL.md` | 协议常量与 DCT 参考臂 |
| `../papers/comparators/` | 四个上游 checkout（只读，不得修改） |
