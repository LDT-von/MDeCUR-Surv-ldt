# 服务器端 AI 交接说明（2026-11-13）

本文件面向**运行在服务器上的 AI 助手**，说明本仓库当前状态、最新机制发现、正在进行的超参搜索协议，以及需要对方接手核对的事项。

---

## 0. 一句话结论

`PathwayLocalSurvival`（`cancer_pathway.py`）的**风险排序几乎完全由交互分支驱动**，基础头（`pooled_path + global_rna`）单独只有接近随机的判别力；因此当前超参搜索聚焦于**交互强度、匹配温度、隐藏容量与辅助权重**，而不是去调基础头的权重。

---

## 1. 仓库与协议基线

- 仓库：`/data1/MDeCUR-Surv-ldt`（origin: `https://github.com/LDT-von/MDeCUR-Surv-ldt.git`）
- 分支：`main`，最新提交见 `git log --oneline -3`
- 预注册协议：**`HYPERPARAM_SEARCH_PROTOCOL_20261112.md`**（本次新增并推送）
- 框架说明：见 `PATHWAY_LOCAL_FRAMEWORK.md`、`LOCAL_GLOBAL_IMPLEMENTATION.md`、`MODALITY_CONTROLS.md`
- 固定 R 诊断与梯度审计：见 `FIXED_R_DIAGNOSTIC_20261112.md`、`FIXED_R_GRADIENT_AUDIT_20261112.md`、`GRADIENT_AUDIT_20261112.md`
- 结果回执目录：`results/`（历史合成/审计回执）

**注意**：`results/` 中已跟踪的 JSON 均已核验无患者标识（`TCGA-*` 扫描为 0）。真实患者特征（`.pt`/`.npy`/`.h5`）、清单（`.csv`）、checkpoint 均被 `.gitignore` 排除。

---

## 2. 最新机制发现（关键）

工具：`diagnose_interaction_contribution.py`（本次新增并推送，参数化 CLI，可在任意冻结检查点上复跑）

对冻结的 new-R 检查点（fold 0，60 名验证患者，26 事件）：

| 项 | 值 |
|---|---|
| 完整模型 C-index | **0.645503** |
| 仅基础头 C-index（同一组冻结参数上去掉交互贡献） | **0.524868** |
| 交互分支 raw 绝对值均值 | 5.3406 |
| 乘 0.2 缩放后均值 | 1.0681 |
| 基础头绝对值均值 | 2.8583 |
| 缩放后交互/基础 比值 | 0.3737 |
| 交互贡献的 logit 跨度占比 | 1.3547（>1，即交互项在跨度上压过基础头） |
| 基础 vs 完整 的可比较对排序一致率 | 见 JSON `risk_ranking.ranking_agreement` |

**解读（注意分寸）**：
- 基础头单独 C=0.5249 ≈ 随机，说明它本身几乎不携带判别信息。
- 交互分支把 C 从随机水平拉到 0.6455，所以**排序能力来自交互分支**。
- 但 `scaled_abs_mean`（1.068）与 `base_abs_mean`（2.858）**不可直接相比**就宣称"交互主导"——两者不是同一尺度上的可加量。真正可比的是 logit 跨度占比（1.3547），它 >1 说明交互项在**动态范围**上超过基础头。
- 这仍是**单检查点、单队列、选轮验证**上的机制诊断，**不是**重训 interaction-off 臂的效果估计。

---

## 3. 正在进行的超参搜索（预注册协议）

协议文件：**`HYPERPARAM_SEARCH_PROTOCOL_20261112.md`**

### 3.1 设计要点

- **Stage 1 家族级筛选**：单折 `fold_0`（243 训练 / 61 验证），31 个配置（C00–C30），在 val NLL、IPCW Brier、C-index 三指标上做**支配关系过滤**，并施加**家族配额**（每个因素家族最多保留 1 个配置），容忍带 -0.03。
- **Stage 2 五折确认**：基线 + 各家族优胜者（共 ≤5 个配置）在全部 5 个内部开发折（304 名开发患者，合并 OOF C-index 的 SE ≈ 0.011）上运行。
- **Stage 3 方差分解**：优胜配置与基线在 seed 29 / 43 上跨 5 折，区分折间方差与种子间方差。
- 全程**不触碰 outer test**。

### 3.2 吞吐修订（§3.0）

`patch_chunk_size` 从 2048 改为 **8192**：稳态 epoch 时间从 ~168s 降到 ~58s，数值上 val NLL 差异 ≤ ~2e-5（仅因特征噪声的分块边界种子不同）。Stage 1–3 全部采用 8192。

### 3.3 当前进度（截至交接时刻）

- 队列已启动，5 个 worker 在 GPU0/GPU1 上并发取任务。
- 已完成配置数**很少**（重启后仅个位数），剩余约 28 个待跑。
- 调度器 `orchestrate.py` 会自动推进 Stage 1 → Stage 2 → Stage 3 → `build_final_report.py`。

### 3.4 运行目录

`/data1/decur_runs/PATHWAY_HYPERPARAM_SEARCH_20261112/`

| 文件 | 作用 |
|---|---|
| `stage1_screen.py` | 31 个配置的网格定义 + 单配置训练入口 |
| `stage1_queue.py` | 带 `fcntl.flock` 锁 + `os.replace` 原子替换的 JSON 队列 worker |
| `reconcile_queue.py` | 重启后对账（in-flight / 已完成 / 释放未产出） |
| `orchestrate.py` | 全流程监督者，自动推进各 Stage |
| `stage2_five_folds.py` | 多折运行器 |
| `summarize_stage1.py` / `summarize_stage2.py` / `summarize_stage3.py` | 聚合 |
| `build_final_report.py` | 生成最终报告 |
| `warm_daemon.py` | 预热 `.pt` 特征到 page cache，缓解 IO 争用 |

---

## 4. 需要服务器端 AI 接手做的事

1. **核对仓库状态**：确认 `main` 与 origin 同步，读取 `HYPERPARAM_SEARCH_PROTOCOL_20261112.md` 全文。
2. **核对队列健康度**：
   ```bash
   cd /data1/decur_runs/PATHWAY_HYPERPARAM_SEARCH_20261112
   python3 reconcile_queue.py
   python3 -c "import json,sys; sys.path.insert(0,'.'); from stage1_queue import read_queue; from collections import Counter; print(Counter(i.get('status') for i in read_queue()))"
   ```
3. **不要重复启动**已在跑的配置，不要清队列，不要 reset/clean 仓库。
4. **不要触碰 outer test**，也不要自动扩展到五折之外的额外折。
5. 若队列空了（worker 全退），可重新拉起 worker：
   ```bash
   cd /data1/decur_runs/PATHWAY_HYPERPARAM_SEARCH_20261112
   for i in 0 1 2 3 4; do
     WORKER_ID=v$i PHYSICAL_GPU=$((i % 2)) IDLE_LIMIT=2 \
       nohup python -u stage1_queue.py > logs/vq$i.log 2>&1 &
   done
   ```
6. **核对本机复现**：`diagnose_interaction_contribution.py` 的 CLI 版本已加入仓库并核验可复现原数字（原脚本 hardcode 路径，现改为 `--checkpoint/--manifest/--out` 参数）。

---

## 5. 已知风险与限制（请如实转述，不要夸大）

- 所有 C-index 均来自**选轮验证队列**（fold 0 的 60 名验证患者），**不是**独立测试集结果。
- 单折 C-index 的 SE ≈ 0.03，因此 fold 0 上 0.02 的差异**不构成证据**。Stage 2 的五折 OOF 是唯一能给出较可靠比较的层级。
- `patch_chunk_size` 8192 与 2048 之间存在 ~2e-5 的 val NLL 差异，不可宣称完全等价。
- 之前的固定 R 实验里，重训 I 臂 C=0.647619 与旧配对区间**包含 0**，**尚未证明机制增量**。
- 本机（本地开发机）**未运行过任何患者训练**；真实患者结果全部来自服务器转发回执。
- 队列 JSON 曾出现并发写损坏，已通过原子替换 + 严格锁修复；对方如需改队列代码请保留这两点。
- 本次未改 `cancer_pathway.py` / `cancer_pathway_stream.py` / `cancer_train.py` 的训练实现，**未改任何 loss 系数**。

---

## 6. 本次推送内容清单

- `HYPERPARAM_SEARCH_PROTOCOL_20261112.md`（新增，预注册协议）
- `diagnose_interaction_contribution.py`（新增，参数化诊断 CLI）
- `results/repeat_cancer_v1..v4.json`（新增，合成回执，无患者数据）
- `AGENTS.md`（追加本轮交接段落）
- `EXPERIMENT_REVIEW_20261112.md`（追加本轮补充）

<!-- 该段落由推送方追加，请勿删除。 -->
