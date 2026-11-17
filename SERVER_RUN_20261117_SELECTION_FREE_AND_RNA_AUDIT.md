# 服务器端执行说明：本批（选轮自由列 + RNA 语义审计）

**日期**：2026-11-17
**性质**：全部为**只读**工作。除第 2 步的合并表重算会**新写一个 JSON 文件**外，
不重训任何模型、不改模型/损失/采样/训练器、不碰 outer test、不动任何已有患者产物。

---

## 0. 为什么是这几件事

审计产物 `results/onefold_comparison_audit_20261117.json` 里**已经有**全部 10 臂的
轮统计量了（`epoch_statistics`），也已经有纯噪声零假设（`pure_noise_null`）。
问题只在于它们停留在只读审计脚本里，**没进合并排名表**。

所以第 1–2 步是**纯代码活 + 一次表重算**，零 GPU、零重训。
第 3 步是一份**只读审计**，回答"RNA 矩阵里到底有什么"。

---

## 1. 拉取代码

```bash
cd <代码目录>            # 必须是代码目录，不是工作区根目录
git pull --ff-only origin main
git log --oneline -1     # 确认 HEAD 到达目标提交
```

---

## 2. 重算合并排名表（新增选轮自由列）

新表**写到一个新路径**，不覆盖旧表。

```bash
cd <代码目录>
python comparison_merge.py \
  --fold0-plan  <原 fold0 plan.json> \
  --fivefold-plan <原 fivefold plan.json> \
  --fold0-runs   <原 6 个 project 臂目录> \
  --fivefold-runs <原 4 个 upstream 臂目录> \
  --survival-reference <原 survival_reference.csv> \
  --required-epochs 30 \
  --out <新路径>/merged_ranking_v2.json
```

新表比旧表多出：

| 字段 | 含义 |
|---|---|
| `rows[].epoch_mean_c_index` | 该臂 30 轮的**轮均值**（选轮无关） |
| `rows[].epoch_sd_c_index` | 轮间标准差 |
| `rows[].epoch_max_c_index` | 轮最大值（= 该臂 published `best`） |
| `rows[].selection_inflation` | 轮最大值 − 轮均值 = 选轮抬升 |
| `rows[].epoch_mean_above_null_p99` | 轮均值是否超过噪声零假设的 p99 |
| `rows[].epoch_max_above_null_p99` | 轮最大值是否超过噪声零假设的 p99 |
| `epoch_mean_rankings` | **选轮无关**的排名（这是推荐读法） |
| `pure_noise_null` | 纯噪声零假设（随机风险、同一 metric call、同一队列、同样 30 轮） |
| `null_reference` | 零假设的 `epoch_mean_p99` / `epoch_max_p99` |

命令行会直接打印 `rankings_epoch_mean` 和 `null_reference`，便于一眼确认。

**注意**：合并表会拒绝合并轮数不一致、两份 plan 患者/标签/分箱不一致、
或共享打分与 runner 记录不一致的臂。报错即表示**不要发布**，不要绕过。

---

## 3. RNA 矩阵语义审计（只读）

```bash
cd <代码目录>
python scripts/audit_rna_matrix_semantics.py \
  --rna /data1/TCGA_Gene_Data/BLCA/rna_inter.csv \
  --clinical /data1/dataset_csv/clinical/all/blca.csv \
  --splits-root /data1/dataset_splits_summary/uni/blca \
  --out results/rna_matrix_semantics_audit_20261117.json
```

### 它回答什么

1. **行语义**：每一行是不是合法基因 symbol。有没有 `sample_type` / `patient_id` /
   `batch` 这类**元数据行**被当成基因用了。
2. **列语义**：每一列是不是合法 TCGA barcode；列集合与各折 split 的对照
   （有没有列不属于任何折、有没有折里的患者没有对应列）。
3. **结局衍生行**：矩阵是"基因 × 患者"，所以结局数据如果混进来，是藏在
   **行**里（标签形如 `EVENT` / `survival_months_dss`，每患者一个值）。
   脚本沿行轴做形状检测并**报告行标签**。
4. **值分布**：分位数、均值/标准差、负值比例、整数比例、每患者均值/标准差范围、
   PC1 方差比例（用于观察是否存在批次/中心结构）。

### 它明确回答不了什么（脚本内 `limits` 已写明）

- **上游是否对全队列做了标准化** —— **从写下来的数值不可判定**：全队列 z-score
  和只用训练行 z-score 的矩阵形状完全一样。只能靠上游脚本或重建来定。脚本把它
  记为 **unknown**，不会猜。
- **是否用结局筛过基因** —— 数值里不留痕迹，同样记为 unknown。
- 病理特征的编码器来源 —— 本脚本不涉及。

---

## 4. 本批明确**没有**做的事

- 没有重训任何臂；第 2 步只重算已有预测的打分。
- 没有碰 outer test；`comparison_merge.py` 与审计脚本都不读它。
- 没有改任何模型/损失/采样/训练器代码。
- 没有覆盖或删除任何已有产物；新表写新路径。
- **没有**恢复五折、**没有**加打乱标签空对照臂、**没有**给 3.2 建 runner。
- 本批**不产生**任何临床效果结论。

---

## 5. 仍然待办（本批未做）

| 项 | 状态 | 备注 |
|---|---|---|
| `deref_2m` / `dimaf_2m` 修复后重跑 | **进行中** | 两个 RNG 缺陷已修（`2eef9e2`），重跑后这两个臂的轮均值才可用 |
| 打乱标签空对照臂 | 未做 | 需要 1 臂 × 30 轮 GPU；噪声零假设**已经**免成本给出 |
| UNI 特征来源回溯 | 未做 | 仓库内只能查"特征缓存脚本有无按患者分层/按生存筛选痕迹"；真实回溯本机不可判定 |
| 零重叠独立评估队列 | 未做 | 需外部队列清单 |
| 恢复五折 | 未做 | 见 `comparison_protocol.FOLDS`；是唯一真正增加统计功效的改动，但五折后 CI 仍宽于第一梯队差距 |
| 3.2 (CLARA) runner | 未做 | 3.2 目前只有合成 CPU 证据 |

---

## 6. 提交与拉取

- 提交：`Add selection-free ranking columns and a read-only RNA matrix audit`
- 拉取：`git pull --ff-only origin main`
