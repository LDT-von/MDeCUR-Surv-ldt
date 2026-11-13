# BLCA 单折同输入、同生存协议比较：服务器交接

本批只做 **legacy BLCA fold0、seed3、30 epochs**。同时报告每个模型的 **val Harrell C 最高轮**与 **epoch30 末轮**；“最后一个”指这一折的最后一个 epoch，不是 fold4。不自动补其余4折，不使用 outer test，不调参搜索。

## 代码与源码状态

目标仓库为 `https://github.com/LDT-von/MDeCUR-Surv-ldt`，服务器现有检出路径应先核对为 **`/data1/MDeCUR-Surv-ldt`**。本地对应 `C:/Users/栋栋/Desktop/DeCUR-main/Conditional_DeCUR_Surv`。不要去父目录或旧 clone 运行。

| 方法 | 服务器拉取后可用的文件 | 状态 |
|---|---|---|
| MDeCUR aux-off | `/data1/MDeCUR-Surv-ldt/comparison_fold0.py`，调用 `cancer_pathway.py` 和 `cancer_pathway_stream.py` | 与既有0.7361配置对应的预测架构，重新训练，辅助项全部关；不把历史数字填进新表 |
| CAMR-2M-P4096-DNLL | `/data1/MDeCUR-Surv-ldt/comparison_camr.py`，经同一个 `comparison_fold0.py` 运行 | 已实现的双模态适配版；不是原论文三模态Cox结果复现 |
| MurreNet | 尚无入口，`--model murrenet` 明确报错 | 官方会议页 Code N/A；作者主页未给该模型代码，本次未找到可验证作者仓库 |
| MoHD | 尚无入口，`--model mohd` 明确报错 | 未找到可验证作者源码；原方法还需要 patch/region 层次输入 |

CAMR 作者源码来自 `https://github.com/wxq-ustc/CAMR`，固定 commit `8fbae0a0bc4e9da31547b5eab5d773e6b2049e93`。本机稀疏检出在 `C:/Users/栋栋/Desktop/DeCUR-main/papers/comparators/CAMR`，仅取源码，未检出或读取附带患者pkl。来源及源码哈希在仓库 **`comparison_sources.json`**。服务器不需要这个外部检出目录；所需适配代码已包含在本仓库。

MurreNet 核查源：`https://papers.miccai.org/miccai-2025/0624-Paper0057.html`、作者主页 `https://mingxinliu.top/` 与其 GitHub `https://github.com/lmxmercy`。不要把同作者 MGCT 改名为 MurreNet。MoHD 核查源：`https://pubmed.ncbi.nlm.nih.gov/42176461/`，DOI `10.1016/j.media.2026.104098`。取得官方链接前，两者保持缺失状态，不填模拟模型或分数。

## 输入和共同协议

所有可运行比较臂共用一个 `plan.json` 和 SHA256。计划锁定清单、RNA文件、参考报告和检查点、患者顺序、双精度标签、bag内容指纹、训练拟合分箱、代码及运行库版本；执行前、第二次训练读取后、执行结束均核对。含 test 行的清单在读取特征前拒绝。

| 项 | 共同值 |
|---|---|
| 清单 | `/data1/decur_runs/PARITY_DCT_FOLD0/manifest.csv` |
| 双精度标签 | `/data1/decur_runs/PARITY_DCT_FOLD0/survival_reference.csv` |
| 完成的参考运行 | `/data1/decur_runs/PARITY_DCT_FOLD0/run_fold0_30ep/` 下 `report.json`、`checkpoint.pt` |
| 人群 | 预期303 train /77 val、101/27事件；由参考报告及内容指纹核对，不能悄悄改人群 |
| 病理 | 同一组外部1024维特征；患者多slide先串接，统一4096 patch预算；训练每轮无放回重抽，验证等间隔，不足补零；各臂训练行索引哈希必须一致 |
| RNA | 同一17025维表达、同基因顺序、policy none；不加CNA、mutation或新特征 |
| 生存 | DSS，月，4箱，train未删失时间qcut，`alpha_surv=.5` 离散NLL，`risk=-sum(S)` |
| 训练 | seed3，30轮，batch8，AdamW lr5e-4/wd5e-4，CosineAnnealingLR(T_max=30) |
| 选轮 | val Harrell C最高，精确平局取最早；末轮固定epoch30 |

新增入口固定进程启动 `PYTHONHASHSEED=0`，因为既有patch采样器使用 Python hash。历史0.7361运行的hash种子未知，**本批重跑MDeCUR参考臂**，不保证逐位复现历史分数。所有臂共用本批抽样与训练顺序。固定输入和主生存目标不意味着模型结构、参数量及各自辅助正则完全相同；这些差异正是方法比较的内容。

CAMR 改写点逐项记在 `CAMR2M.adaptation_metadata()`：原三路80维患者向量改为病理attention MIL＋RNA输入适配、删除CNA、三路融合改双路、Cox标量改4箱logits。保留shared/private编码、门控、低秩融合、DiffLoss、重建及显式D/G更新。辅助默认 adv=.6、rec=.8、orth=.05；重建目标为学习得到的80维输入表示，不是全部原始RNA或所有patch。公开源码训练器没有实际更新独立判别器；本适配版显式更新，属于已披露差异。

指标完全调用现有 `cancer_parity.py`，不会为新模型单独修口径。报告 C/IPCW C/NLL/Brier/IBS/AUC 以及 errors。旧网格只有约8.09、15.12月两点，当前 IAUC 实际使用 `grid[1:]`（此时仅15.12月单点）；IBS只覆盖该短区间。表中标 `AUC_legacy_times`、`IBS_legacy_grid`，不可说成全随访IAUC/IBS。IPCW异常保留，不能因某次null高就宣布指标无效，也不能把AUC视为不受删失权重影响。

## 服务器窗口按此执行

先只读核对：`pwd`、`git status --short`、当前commit、相关训练进程与GPU。保留已有path_only、Stage1队列和其他任务；不清队列、不重复启动、不杀进程。仅在跟踪文件无冲突时 `git pull --ff-only origin main`。如果远端被其他提交推进，确认本批文件已包含即可，不能reset/强推覆盖对方工作。

使用现有 **trisurv** 环境，不安装/升级依赖。先跑CPU工程测试；若失败停在故障点。下面用绝对路径写清所有入口：

```bash
cd /data1/MDeCUR-Surv-ldt
conda activate trisurv
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m unittest test_comparison_camr test_comparison_fold0
```

后续在可用GPU上运行，不抢占已有作业。可通过 `CUDA_VISIBLE_DEVICES` 选择一块空闲GPU，各臂使用相同GPU设备类型。命令放进同一个带 `set -euo pipefail` 的shell，任何一步失败不继续训练下一步。新目录名示例，不覆盖已有目录；若已存在，先核对产物和进程，再选新的批次名。

```bash
set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
COMPARE_REPO=/data1/MDeCUR-Surv-ldt
COMPARE_ROOT=/data1/decur_runs/BLCA_FOLD0_COMPARATORS_20261113_V1
test ! -e "$COMPARE_ROOT"
mkdir "$COMPARE_ROOT"
python "$COMPARE_REPO/comparison_fold0.py" plan \
  --manifest /data1/decur_runs/PARITY_DCT_FOLD0/manifest.csv \
  --survival-reference /data1/decur_runs/PARITY_DCT_FOLD0/survival_reference.csv \
  --reference-run /data1/decur_runs/PARITY_DCT_FOLD0/run_fold0_30ep \
  --out "$COMPARE_ROOT/plan.json"
COMPARE_PLAN_SHA=$(sha256sum "$COMPARE_ROOT/plan.json" | cut -d ' ' -f 1)
python "$COMPARE_REPO/comparison_fold0.py" run \
  --plan "$COMPARE_ROOT/plan.json" --plan-sha256 "$COMPARE_PLAN_SHA" \
  --model mdecur_aux_off --device cuda:0 --out "$COMPARE_ROOT/mdecur_aux_off" \
  > "$COMPARE_ROOT/mdecur_aux_off.log" 2>&1
python "$COMPARE_REPO/comparison_fold0.py" run \
  --plan "$COMPARE_ROOT/plan.json" --plan-sha256 "$COMPARE_PLAN_SHA" \
  --model camr_2m --device cuda:0 --out "$COMPARE_ROOT/camr_2m" \
  > "$COMPARE_ROOT/camr_2m.log" 2>&1
python "$COMPARE_REPO/comparison_fold0.py" summary \
  --runs "$COMPARE_ROOT/mdecur_aux_off" "$COMPARE_ROOT/camr_2m" \
  --out "$COMPARE_ROOT/comparison.json"
```

此批正式训练只有 **2条臂×1 fold×30轮**。不要调用 search orchestrator，不跑 folds1–4，不启动未提供源码的两篇，不为提高分数临时改超参。可先用独立输出目录加 `--smoke` 跑1轮CUDA工程检查；该目录会标smoke，不能进入正式比较。

每条臂输出 `checkpoint_best.pt`、`checkpoint_last.pt`、`val_predictions_best.csv`、`val_predictions_last.csv`、`report.json`。最后两份指标取各自同一个检查点；不能分别挑每项指标的最高轮。汇总拒绝抽样不同、计划不同、设备配置不同、未完成30轮、smoke、test或重复臂。

回报一张4行表：模型×best/last，列epoch、Harrell C↑、IPCW C↑、NLL↓、Brier(2点)↓、IBS_legacy_grid↓、AUC_legacy_times↑，再列参数量、耗时、实际患者事件数、计划SHA、训练抽样SHA、源码commit及全部metric errors。回答“谁高”时，Harrell C峰值和epoch30分别排序，其他指标逐项说明方向，不把每项独立峰值拼在一行。

旧DCT 0.7266/0.6371、旧MDeCUR 0.7361/0.7076、旧RNA-only/ridge可以另列“历史转发参考”，**不混入本批锁定计划排名**。本批没有自动重跑DCT/RNA-only/path-only；尤其不要重复启动仍在跑的path_only。单折验证峰值只是探索性点估计，不能宣称总体优越、临床有效或上游完全无泄漏；旧封存76人与本303/77的ID交集仍需另行核实。
