# 提示词：服务器端拉取并运行 BLCA fold0 单折对照（项目臂 6 条 ＋ 上游对照臂 4 条）

> **本文件副本**：`Conditional_DeCUR_Surv/PROMPT_SERVER_BLUCA_FOLD0_RANKING.md`（与工作区根目录
> 的 `PROMPT_服务器端BLCA_fold0排名运行.md` 内容相同）。放在仓库内是为了让服务器
> `git pull` 之后能直接拿到本提示词。
>
> **代码基线 commit**：`12820a5bc28dbe746c17948e0fb416f4f454abe3`

> **用途**：交给服务器窗口的 AI 执行。目标是**跑出分数排名**，不是复述文档。
> **代码已就绪**：适配工作已完成并推送。服务器上 `git pull --ff-only origin main` 之后
> **直接跑即可，不需要写任何新代码**。
>
> **本提示词只依据本仓库已推送的代码和文档**，不含任何本机编造的预期分数。

---

## 0. 先看懂这批要跑的是什么

### 0.1 总体设定

- **队列**：BLCA（膀胱癌）legacy fold0，患者级划分，train/val = 303/77。
- **任务**：WSI 病理 patch 特征（UNI v1）＋ bulk RNA → 患者生存预测（终点 DSS，单位 months）。
- **统一协议** `blca_fold0_full_dense_patientwise_v3`：
  - 4 箱时间离散化（分位数，**在训练折上拟合**）、每患者 4096 patch 预算（不足则零填充）、
    30 epoch、AdamW、lr/wd = 5e-4、Cosine(30)、seed 3、batch 20（尾批按 rank bound 合并）。
  - **生存损失**：逐患者离散 hazard NLL `patientwise_clamped_discrete_v1`，alpha = 0.5。
    风险分 = `risk = -Σ S_j`。
  - **指标**：Harrell C↑、IPCW C↑、生存 NLL↓、Brier↓、IBS↓、AUC↑。
  - **选轮**：验证集 Harrell C 最大轮，另报 epoch 30（"last"）。**这是验证集选轮，不是独立测试。**
- **第一组六条臂共用同一份 plan、同一套 patch 抽样、同一批次调度、同样的 optimizer 步数**，
  所以它们之间是可比的。`summary` 会强制校验这些一致性，不一致就拒绝汇总。

**上游对照臂（第二组 4 条）不在这个 plan 里**：它们由 `comparison_5fold.py` 跑，
用的是**另一个协议字符串**，但终点、单位、分箱函数、patch 预算、patch 重抽种子、
训练种子、似然版本、指标实现**完全同源**。最后用 `comparison_merge.py` 把两组并成一张表：
**所有指标都由合并器用同一套代码从两边保存的生存矩阵重算**，不会把一个 runner 的
NLL 和另一个 runner 的 NLL 放在一起。详见 §0.6。

### 0.2 本批要跑的是**两组**臂，共 10 条

**第一组（项目臂，6 条）**：`comparison_fold0.py` 跑，协议 `blca_fold0_full_dense_patientwise_v3`。

| 臂名 | 这是什么 | 怎么算 | 参数量（V2 实测） |
|---|---|---|---|
| `mdecur_full_dense` | **本项目主方法 MDeCUR-Surv**。病理全局+通路局部编码，RNA 通路编码，11 个旧辅助项 + 4 个 dense token 辅助项。 | 主损失 = 逐患者离散 NLL + 15 个辅助项 | 464,645 |
| `mdecur_dense_aux_off` | **同一模型，辅助项全关**。只留生存损失。 | 主损失 = 逐患者离散 NLL | 464,645 |
| `camr_2m` | **CAMR 协议适配臂**。官方 CAMR 的共享/私有编码器 + 双线性跨模态门控 + 低秩融合 (LMF) + 重建头 + 判别器，两模态（病理 + RNA），第三模态用常量占位。生存头换成 4 箱离散 hazard logits。 | 主损失 = 逐患者离散 NLL + `.6·adversarial + .8·reconstruction + .05·orthogonality`（上游权重），判别器独立 optimizer | 2,742,361 |
| `camr_2m_official` | **CAMR 上游生存契约臂**。与 `camr_2m` **只有生存头和生存损失不同**，其它结构完全一样。生存头是上游原样的 `Sigmoid` 标量（缩放到 `[-4,4]`），训练损失是**连续 Cox 偏似然**，风险集按**生存时间**定义（不复用上游按行号下三角的 `R_set`）。 | 主损失 = Cox 偏似然 + 同一组上游辅助权重 | 与 `camr_2m` 同量级 |
| `morphid` | **新候选方法 MorphID-Surv**。冻结的无标签形态原型库 + 患者级交叉拟合 RBF 核岭 RNA 探针 + 离散生存头。路径：冻结 patch 特征 → 固定原型 → 组成与 CLR → 冻结 RNA → 交叉拟合核岭得 RNA 可预测部分 c → 残差 u = p − c → [RNA, c, CLR, u, u×RNA] → 生存头 → 离散 hazard logits。 | **只有生存头可训练**；形态原型、标准化、探针全冻结 | 更小（见运行时报告） |
| `morphid_no_interaction` | **同一个 MorphID，关掉残差×RNA 乘积项**，用于归因"交互项有没有用"。 | 同上 | 同上 |

**第二组（上游对照臂，4 条）**：`comparison_5fold.py` 跑**同一个 fold 0**，
协议 `blca_5fold_dss_uni_v1_p4096_dnll_v1`。

| 臂名 | 官方仓库 | 上游 commit | 执行的上游创新模块 | 适配点 |
|---|---|---|---|---|
| `pibd_2m` | `zylbuaa/PIBD` | `bd5bd94e` | PIB / PoE / MITransformerLayer / MIAttention / MIEstimator(CLUB) / SNN_Block / NLLSurvLoss | 上游 `.cuda()` 采样器换设备中立副本；主损失换共享离散 NLL，上游 proxy/KL/CLUB 保留为辅助项 |
| `camr_2m`（第二个副本） | `wxq-ustc/CAMR` | `8fbae0a` | 共享-私有编码、跨模态门、LMF、重建、DiffLoss | 三模态降两模态（CNA 常量占位）；80 维患者向量改 attention MIL + RNA 投影；Cox 标量改离散 logits |
| `deref_2m` | `ZJUMAI/DeReF` | `6bdbf81` | Transformer_P/G、Private/Share encoder、Gate、expert、Nyström attention、BilinearFusion | 6 个 omic 组来自上游 `csv/signatures.csv` 六个功能家族（**只按基因名匹配，不看表达值**）；上游随机特征重组保留 |
| `dimaf_2m` | `Trustworthy-AI-UU-NKI/DIMAF` | `3fe8123` | MultiSNN/SNN_Block、CrossAttentionLayer、DIMAF、PANTHERBase/DirNIWNet、mog_eval、DistanceCorrelation/Orthogonal/HSIC | 原型改在**当前折训练 patch** 上聚类（不用上游 pickle）；slide embedding 改走协议 bag |

> **`camr_2m` 会被跑两次**（两个 runner 各一次，同一批患者）。这是**故意的**：
> 它是**桥接臂**，用来量化"跨 runner 偏移"。合并表里它的两行差值，就是所有跨组对比要继承的偏移量。

**不可用方法（不建占位模型，表中不留空行）**：MurreNet（MICCAI 2025）、MoHD
—— 均未找到可核验的官方实现。

### 0.3 两条 CAMR 臂的可比性边界（务必读）

`camr_2m` 与 `camr_2m_official` **不共享训练目标**：

- `camr_2m` 用逐患者离散 NLL，所以它的 NLL 与其它臂同口径。
- `camr_2m_official` 用连续 Cox 偏似然。为了能被同一套评估代码打分，它的标量风险会经
  **训练折上的 Breslow 基线**做比例风险投影 `S(t)=exp(-H0(t)·exp(risk))` 回到 4 个协议箱。
  该投影对风险单调，所以排序指标（C、IPCW C）不受影响；但 NLL/Brier/IBS 衡量的是
  "该模型在共享指标下的标定"，**不是** Cox 目标的值。
- `summary` 会在 `arm_comparability_note` 与 `training_loss_versions` 中写明每条臂的训练似然。
- 两臂都**不是 CAMR 论文复现**：BLCA 无 CNA（第三槽常量占位）、4096 patch 需 attention MIL
  汇总成上游 80 维向量、优化器与调度用协议统一版本、判别器在本工程有独立 optimizer。
  **任何 CAMR 臂的绝对分数都不能与 CAMR 论文的 0.650 直接对比。**
- `camr_2m_official` 另有一处 **batch 风险集**限制：Cox 偏似然在 batch 内计算，
  风险集只含 batch 内患者。这是分批 Cox 的固有性质。

### 0.4 已有的 V2 结果：不能进本批排名表

服务器上 `/data1/decur_runs/BLCA_FULL_DENSE_FOLD0_20261114_V2/` 是**广播版似然**下训练的
（协议 `blca_fold0_full_dense_v2`，2026-11-14 的提交）。当时 `discrete_nll` 有**跨患者广播 bug**：
删失权重 `[B]` 未 squeeze 就乘上 `[B,1]` 的逐患者项，得到 `[B,B]` 矩阵，混合事件/删失 batch 里
一个患者的删失标记会缩放另一个患者的 log 项。修复在 `5e8ac3f` 落地。

因此：
- V2 的**训练权重被污染**，Harrell C / IBS / AUC 虽由预测分数算出、与似然无关，但权重来自污染训练
  → **只能当历史记录，不能当结论**；
- V2 的 **NLL 列无效**（评估也是整批 `[B,B]` 求均值）；
- V2 的 "`full_dense` vs `dense_aux_off` 打平" 这条**也必须重测**；
- V2 的 `report.json` **没有 `loss_version` 字段**，事后**无法自动识别**为广播版，只能靠目录名和
  协议 ID 人工区分；
- **禁止**复用/混入/覆盖/删除 V2 产物，也**禁止**把 V2 数字当"参照"写进本批表格。
- 本批 `summary` 会检查 `protocol` 与 `loss_version`，V2 产物过不了这个守卫。
  **不要**为了让它通过而改协议或改字段。

### 0.5 关于 MorphID 的两点（不要误判为 bug）

1. **MorphID 头训练读取的是缓存的患者级 OOF 特征**，所以**每轮 patch 重采样对它无效**。
   这是冻结式设计的必然结果，不是故障。报告元数据里
   `per_epoch_patch_resampling_affects_this_arm: false` 就是记录这件事。
2. **MorphID 的形态基底在训练折拟合一次**，所以它**不是完全嵌套的交叉验证**。
   报告元数据 `probe_diagnostics` 里会给出每簇的 `oof_r2`、`valid_patients`、
   `fitted`/`reason`，以及 `fit_patient_ids`（应全部是训练患者，**不含** val/test）。

### 0.6 两组臂怎么合成一张表（`comparison_merge.py`）

合并器**不信任任何一个 runner 报出来的指标**。它做的是：

1. **核对两份 plan 描述的是同一批患者**：患者数、事件数、患者顺序哈希、标签哈希、
   分箱边界必须逐项相同，否则**拒绝合并**。
2. **从两边保存的 `val_predictions_{best,last}.csv` 读出生存矩阵**，
   **反解 hazard logits**，用**同一套** `parity_metrics` + 患者级离散 NLL **重算所有指标**。
3. **交叉验证**：重算值与 runner 自己存的值一旦不一致（容差 1e-6），**拒绝出表**。
4. **桥接臂偏移**：`camr_2m` 两行（两个 runner）的 Harrell C 与 NLL 差值写进
   `bridge_arm`，这就是所有跨组对比继承的偏移量。
5. **NLL 统一**：上游对照 runner 自己的 NLL 是拿 `log(1-S)` 当 logits 算的，
   语义与项目臂不同。合并表里的 `NLL` 列**一律**是统一重算值；
   runner 原生 NLL 另存为 `runner_reported_NLL` 供审计，**不与统一值混用**。

**合并器会拒绝的情况**：两份 plan 数据不一致、重算与 runner 不一致、smoke 产物、
用过 outer test、似然版本不是 `patientwise_clamped_discrete_v1`、重复臂、
各臂 epoch 数不一致或不足 30 轮。

**必须如实说明的两处 runner 差异**：
- 上游对照 runner **不做尾批 rank bound 合并**，因此上游臂每轮比项目臂多一个 optimizer 步；
- 两个 runner 的 **runner 原生 NLL 定义不同**（已由合并器统一）。

---

## 1. 第一步：只读核对现状（不修改任何东西）

```bash
cd /data1/MDeCUR-Surv-ldt
pwd
git status --short
git rev-parse HEAD
git log --oneline -5
echo "--- 远端 ---"
git fetch --no-tags origin main
git rev-parse origin/main
git log --oneline origin/main -3
echo "--- 远端比本地多的提交 ---"
git log --oneline HEAD..origin/main
git diff --stat HEAD..origin/main
echo "--- 本地比远端多的提交 ---"
git log --oneline origin/main..HEAD
git diff --stat origin/main..HEAD
```

**判读规则**：

- 若 `HEAD` 落后 `origin/main` 且工作树干净 → 直接 `git pull --ff-only origin main`，继续 §2。
- 若服务器 HEAD 多出提交（`origin/main..HEAD` 非空）→ **先把这些提交的内容看清楚**
  （`git show`），确认它们只是文档/审计脚本，还是改过训练/评估/模型文件。
  - 若改过训练或评估文件：**不要 reset，不要 rebase 覆盖，不要强推**。
    保留它们（`git branch backup/server-<sha>`），汇报改动内容与影响，再决定合并方式。
    若与本批 `origin/main` 的改动冲突，**停下来汇报**，不要自行合并。
  - 若只是文档/脚本：可 `git pull --ff-only`（无冲突时），但要在汇报里写明保留了哪些提交。
- 若工作树有未提交改动：**不要 reset、不要 clean**。先 `git stash list` 与 `git status` 汇报，
  确认那是谁的工作、是否与本批冲突。

**必须同时回报**：
- `git rev-parse HEAD`、`git rev-parse origin/main` 的完整 SHA
- 服务器上是否已有正在运行的训练进程（`nvidia-smi`、`ps aux | grep -E "python.*(comparison|train)"`）
- 是否有 `warm_daemon` 或其他后台任务在跑（V1 报告提到它仍运行）
  **不要杀进程、不要清队列、不要重复启动已在跑的任务**
- `/data1/decur_runs/` 下是否已存在同名或同协议的实验目录（避免重复跑）

**不要做的事**：不安装/升级 conda 环境，不动 `papers/comparators/` 四个上游 checkout，
不启动 folds 1–4，不打开 outer test。

---

## 2. 第二步：环境与 CPU 回归

```bash
cd /data1/MDeCUR-Surv-ldt
conda activate trisurv
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m pytest -q --disable-warnings \
  test_comparison_fold0.py test_comparison_morphid.py test_comparison_camr.py \
  test_comparison_camr_official.py test_comparison_merge.py \
  test_cancer_parity.py test_cancer_parity_bag.py test_discrete_nll_patientwise.py \
  test_cancer_morphid.py test_morphid_probe.py test_decur_losses.py
```

**期望**：全部通过。本地等价全套是 `408 passed, 3 skipped`（3 项跳过是本机缺 DCT 参考仓库；
服务器上若有该仓库则不应跳过）。本批新增 `test_comparison_morphid.py` 8 项、
`test_comparison_camr_official.py` 15 项与 `test_comparison_merge.py` 9 项。

> `test_comparison_merge.py` 会真实训练两个 runner 的少量轮次，**在 CPU 上约需 2–3 分钟**，
> 比其它测试慢，属正常。

**若失败**：**停在故障点**，把完整 traceback 贴出来。不要通过改测试或改超参绕过。
特别留意 `ModuleNotFoundError: No module named 'model'` —— 本批删除了 `model.py`，
三个诊断脚本的源码哈希清单已改为 `decur_losses`；若服务器上是旧文件，`git pull` 后应自动解决。

---

## 3. 第三步：生成 plan

```bash
set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/data1/MDeCUR-Surv-ldt
ROOT=/data1/decur_runs/BLCA_FOLD0_RANKING_20261116
# 若目录已存在，先核对内容再改名，不要覆盖
test ! -e "$ROOT" || { echo "EXISTS: $ROOT"; exit 1; }
mkdir -p "$ROOT"

python "$REPO/comparison_fold0.py" plan \
  --manifest /data1/decur_runs/PARITY_DCT_FOLD0/manifest.csv \
  --survival-reference /data1/decur_runs/PARITY_DCT_FOLD0/survival_reference.csv \
  --reference-run /data1/decur_runs/PARITY_DCT_FOLD0/run_fold0_30ep \
  --out "$ROOT/plan.json"
PLAN_SHA=$(sha256sum "$ROOT/plan.json" | cut -d' ' -f1)
echo "PLAN_SHA=$PLAN_SHA"

# 确认 plan 里包含全部臂与相关源码
python - <<'PY'
import json
plan = json.load(open('/data1/decur_runs/BLCA_FOLD0_RANKING_20261116/plan.json'))
print('available_models:', plan['available_models'])
print('protocol:', plan['protocol'])
print('loss_version:', plan['settings'].get('loss_version'))
for name in ('comparison_morphid', 'cancer_morphid_model', 'morphid_probe'):
    print(name, 'in source_sha256:', name in plan['source_sha256'])
print('morphid_fit_scope:', plan['settings']['morphid_fit_scope'])
PY
```

**必须确认**：`available_models` 含 `camr_2m_official`、`morphid` 与 `morphid_no_interaction`，
`loss_version` 是 `patientwise_clamped_discrete_v1`，且三个 MorphID 源码都在 `source_sha256` 里
（否则 plan 锁不覆盖新代码路径）。

**注意**：`plan` 会校验参考 run 的 `report.json`（DSS / months / 4 bins / alpha .5 /
4096 patches / seed3 / 30 epochs / aux 全关 / rna policy none）。
若该参考 run 不存在或与当前代码生成的协议不符，**会直接报错**——那是正确行为，
按错误信息处理，不要改协议去迁就。

---

## 4. 第四步：正式训练（6 条臂，串行或按空闲 GPU 并行，设备类型必须一致）

```bash
for MODEL in mdecur_full_dense mdecur_dense_aux_off camr_2m camr_2m_official \
             morphid morphid_no_interaction; do
  echo "=== $MODEL ==="
  python "$REPO/comparison_fold0.py" run \
    --plan "$ROOT/plan.json" --plan-sha256 "$PLAN_SHA" \
    --model "$MODEL" --device cuda:0 \
    --out "$ROOT/$MODEL" \
    > "$ROOT/$MODEL.log" 2>&1 || { echo "FAILED: $MODEL"; tail -40 "$ROOT/$MODEL.log"; exit 1; }
done
```

**可选**：先用独立目录加 `--smoke` 跑 1 轮 CUDA 工程检查。
`--smoke` 产物带 smoke 标记，**不能进入正式排名**。

**若某臂报 `--model` 不在 choices 里**，说明服务器上的代码不是 `0b3763a` 之后的版本——
回到 §1 核对，**不要换用别的臂顶替**。

---

## 4b. 第四步之二：拉取上游对照源码（只在第一次需要）

`papers/comparators/` 下的四个上游仓库**不在本仓库里**（第三方产物，未随 GitHub 发布）。
服务器上必须先拉一次，**固定 commit**，并让逐文件 SHA256 校验通过。

```bash
cd /data1/MDeCUR-Surv-ldt
bash scripts/fetch_upstream_comparators.sh
```

> **路径说明**：`comparison_sources.py` 从**代码目录的上一级**推导上游根目录，
> 即 `/data1/MDeCUR-Surv-ldt/../papers/comparators` = `/data1/papers/comparators`。
> 脚本用自身位置推导同一路径，两边必须一致。

**期望输出**：四个仓库各打印 `HEAD`，最后打印 `all upstream checkouts match the recorded hashes`。

**若脚本报 "has uncommitted changes" 或 "does not match the pinned commit"**：
**停下来汇报**，不要 reset、不要强推、不要删目录重 clone。

---

## 4c. 第四步之三：上游对照臂训练（同一个 fold 0）

需要 4 个额外输入（**只按名字匹配，不看表达值**）：有序基因名 `genes.txt`、
GMT `combine_signatures.gmt`、DeReF `csv/signatures.csv`。

```bash
set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/data1/MDeCUR-Surv-ldt
ROOT=/data1/decur_runs/BLCA_FOLD0_RANKING_20261116
FF_ROOT="$ROOT/fivefold_fold0"
mkdir -p "$FF_ROOT"

python "$REPO/comparison_5fold.py" plan --fold 0 \
  --manifest /data1/decur_runs/PARITY_DCT_FOLD0/manifest.csv \
  --survival-reference /data1/decur_runs/PARITY_DCT_FOLD0/survival_reference.csv \
  --genes /data1/decur_runs/PARITY_DCT_FOLD0/genes.txt \
  --gmt /data1/decur_runs/PARITY_DCT_FOLD0/combine_signatures.gmt \
  --deref-signatures "$REPO/../papers/comparators/DeReF/csv/signatures.csv" \
  --device cuda:0 \
  --out "$FF_ROOT/plan.json"
FF_SHA=$(sha256sum "$FF_ROOT/plan.json" | cut -d' ' -f1)
echo "FF_SHA=$FF_SHA"

for MODEL in pibd_2m camr_2m deref_2m dimaf_2m; do
  echo "=== $MODEL ==="
  python "$REPO/comparison_5fold.py" run \
    --plan "$FF_ROOT/plan.json" --plan-sha256 "$FF_SHA" \
    --model "$MODEL" --device cuda:0 \
    --out "$FF_ROOT/$MODEL" \
    > "$FF_ROOT/$MODEL.log" 2>&1 || { echo "FAILED: $MODEL"; tail -40 "$FF_ROOT/$MODEL.log"; exit 1; }
done
```

**注意**：这份 plan 用的是**另一个协议字符串**（`blca_1fold_dss_uni_v1_p4096_dnll_v1`），
它与项目臂的 plan **不同**——这是预期的，不是错误。合并器会核对两份 plan
描述的是同一批患者（患者数/事件数/顺序哈希/标签哈希/分箱边界）。

**折数说明（2026-11-16）**：调度已由五折削减为**只跑 fold 0**，项目臂与上游臂
**都在同一个 fold 0** 上。`comparison_5fold.py` 这个文件名与"five-fold"字样是
历史命名，实际的折范围由 `comparison_protocol.FOLDS` 决定，现在是 `(0,)`；
`--fold 1..4` 会被代码拒绝（argparse choices 只有 `0`）。

**若 `plan` 报基因名维度不匹配**：说明 `genes.txt` 与本批 RNA 维度/顺序不一致，
**停下来汇报**，不要改 RNA 维度去迁就。

---

## 4d. 第四步之四：合并成一张排名表

```bash
python "$REPO/comparison_merge.py" \
  --fold0-plan "$ROOT/plan.json" \
  --fivefold-plan "$ROOT/fivefold_fold0/plan.json" \
  --fold0-runs "$ROOT/mdecur_full_dense" "$ROOT/mdecur_dense_aux_off" \
                "$ROOT/camr_2m" "$ROOT/camr_2m_official" \
                "$ROOT/morphid" "$ROOT/morphid_no_interaction" \
  --fivefold-runs "$ROOT/fivefold_fold0/pibd_2m" "$ROOT/fivefold_fold0/camr_2m" \
                  "$ROOT/fivefold_fold0/deref_2m" "$ROOT/fivefold_fold0/dimaf_2m" \
  --survival-reference /data1/decur_runs/PARITY_DCT_FOLD0/survival_reference.csv \
  --required-epochs 30 \
  --out "$ROOT/merged_ranking.json"
```

输出 `merged_ranking.json` 含 `rows`（每条臂的 best/last 两行，全部指标为统一重算值）、
`Harrell_C_rankings`、`bridge_arm`（桥接臂偏移）、`comparability_limits`。

---

## 5. 第五步：汇总排名

```bash
python "$REPO/comparison_fold0.py" summary \
  --runs "$ROOT/mdecur_full_dense" "$ROOT/mdecur_dense_aux_off" "$ROOT/camr_2m" \
         "$ROOT/camr_2m_official" "$ROOT/morphid" "$ROOT/morphid_no_interaction" \
  --out "$ROOT/comparison.json"
```

`summary` 会**拒绝**这些情况：plan SHA 不一致、protocol 或 loss_version 不同、
smoke 产物、不足 30 轮、评估过 test、重复臂、训练抽样哈希不同、训练调度哈希不同、
optimizer_steps 不同、设备配置不同。**这些拒绝是保护机制，不要绕过。**

---

## 6. 回报格式（必须照此给用户）

### 6.1 分数排名表

**主表用 `merged_ranking.json` 的 `rows`**（10 条臂 × best/last），因为那里的指标
是统一重算的。`comparison.json`（项目臂自汇总）作为交叉核对一起给出。

| 臂 | 组 | 选轮 | epoch | Harrell C↑ | IPCW C↑ | 生存NLL↓ | Brier↓ | IBS_legacy_grid↓ | AUC_legacy_times↑ | 参数量 | 耗时 |

**规则**：

- 每个模型给 **best** 与 **last(epoch30)** 两行，两行来自**同一个检查点**，
  **不要**为每项指标各挑一个最高轮。
- Harrell C 峰值排序与 epoch30 排序**分别给**。
- 其他指标逐项说明方向（NLL/Brier/IBS 越低越好，AUC 越高越好），**不要**说成"越高越好"。
- **若某指标存在方向冲突**（例如 CAMR 在 V1 中 Harrell 更高但 IPCW 与 NLL 更差），
  **如实并列呈现**，不要宣布"整体更好"。

### 6.2 必须同时给出

- plan 的完整 SHA256
- 各臂的 `training_sampling_sha256`、`training_schedule_sha256`（必须彼此相同）
- 各臂的 `optimizer_steps`
- 源码 commit（完整 SHA）
- 实际患者数与事件数（train/val）
- 全部 metric errors 与 finite-edge 审计
- **`merged_ranking.json` 的 `bridge_arm`**：桥接臂 `camr_2m` 两个 runner 的
  Harrell C 与 NLL 差值（这是所有跨组对比继承的偏移量，**必须写进结论**）
- 上游四个仓库的 URL、完整 commit、逐文件 SHA256 校验结果
- **MorphID 特有**：`probe_diagnostics.probe_clusters`（每簇 `fitted`/`reason`/
  `valid_patients`/`oof_r2`）、`frozen_state_unchanged`、`fit_patient_ids`
  （确认**不含** val/test 患者）、`realised_latent_dim`
- GPU 型号、`CUDA_VISIBLE_DEVICES`、耗时

### 6.3 必须写明的局限

- 单折、验证集选轮，**不是独立测试**，**不是显著性结论**，**不能宣称总体优越或临床有效**。
- **本批是 1 折（fold 0）**：调度由五折削减为单折以适配时间预算。
  因此**没有任何折间方差估计**，无法排除折间波动，排名**只能是单折上的点估计顺序**。
  表里出现的 `mean` 就是该折的值、`std` 恒为 0，行内 `fold_count=1`；
  **不得**把这些数字说成"五折平均"或"稳定优于"。
- 旧封存 76 人与本 303/77 的 ID 交集仍未核实。
- 归档的 DCT 数字（0.7266/0.6371）与旧 MDeCUR（0.7361/0.7076）来自**广播版似然**，
  **不能与本批并列**；若要并列必须在同一 plan 下重跑。
- **V2 结果（协议 `blca_fold0_full_dense_v2`，2026-11-14 目录）同样来自广播版似然**，
  **不得**出现在本批排名表里，也不得与本批数字合并、平均或"参考"。
  本批数字必须全部来自 `blca_fold0_full_dense_patientwise_v3` 新目录。
- **`camr_2m` 与 `camr_2m_official` 都是已披露适配的双模态比较臂，不是 CAMR 官方复现**；
  它们的绝对分数不能与 CAMR 论文的 0.650 对比。两条臂**训练目标不同**
  （离散 NLL vs 连续 Cox 偏似然），`comparison.json` 的 `training_loss_versions`
  与 `arm_comparability_note` 会写明；`camr_2m_official` 的 NLL/Brier/IBS 来自 Breslow
  比例风险投影，衡量的是标定而非 Cox 目标值。
- **若 MorphID 参与**：  - 冻结形态基底**非完全嵌套交叉验证**；
  - 探针重建的是**冻结病理表示**而非标签，**低 R² 不是独立性证据**；
  - 预测残差是 **RNA 条件化估计**，不是已识别的独有成分；乘积项**不是 PID synergy**；
  - CLR 组成**自由度 K−1**，各系数不是独立的绝对组织效应；
  - 簇数 K=4、latent_dim=8 是事先固定常量，**未用验证集标签选择**；
  - **每轮 patch 重采样对该臂无效**（设计使然）。
- **两组臂来自两个 runner，合并表由 `comparison_merge.py` 统一重算指标**：
  - 上游对照 runner **不做尾批 rank bound 合并**，上游臂每轮比项目臂多一个 optimizer 步；
  - 两个 runner 的**原生 NLL 定义不同**，合并表 NLL 是统一重算值，
    `runner_reported_NLL` 仅供审计；
  - `bridge_arm` 给出的偏移量**必须随结论一起报告**，不能只报排名。
- **四个上游臂都是已披露适配，不是原论文复现**；**任何一臂的绝对分数都不能与
  论文表格（CAMR 0.650 等）直接对比**。CAMR 上游执行程度在四者中相对最低，结论中单列。
- **MurreNet / MoHD 无可核验官方实现**，不得推断"它们表现如何"，表中不留空行。
- **归档的 DCT 五折数字（0.7005±0.0333）不在本表**，它来自广播版似然且是另一套评估协议。

---

## 7. 禁止事项

1. 不为提高分数临时改超参、不加折、不开 outer test。
2. 不把历史数字（V1 的 0.735341/0.786846 等，**以及 V2 的 0.781300/0.679873/0.685420 等**）
   填进新表冒充本批结果；**也不得**把 V2 数字作为"参照/对照"混入本批排名。
3. 不改写 `comparison_morphid.py` / `comparison_morphid` 臂的实现——它已就绪并有测试。
4. 不 reset / clean / 强推，不覆盖服务器上已有的提交或未提交工作。
5. 不 kill 已在运行的进程，不重复启动同一实验，不装/升环境。
6. 任何一步失败**停下来汇报**，不要用改协议、改测试的方式让它"通过"。
7. 不声称临床有效性、显著性、方法首创性。
8. 不把"合成检查通过"当作患者效果证据。
9. 不修改 `papers/comparators/` 下的任何第三方文件；上游仓库只在**固定 commit** 处使用。
10. 不跳过 `comparison_merge.py` 直接把两个 runner 的数字手工拼成一张表——
    合并器统一重算指标并量化桥接臂偏移，手工拼接会丢掉这两层保护。
