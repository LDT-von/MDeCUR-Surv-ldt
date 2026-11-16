# 已完成V1单折结果的审查与安全接续

本页依据用户2026-11-16转发的服务器报告，**不是本机读取真实患者产物后的复核**。用户称两条臂11月13日16:32结束，best/last检查点、预测、30轮history及comparison.json完整保留；两训练进程退出，warm_daemon仍运行。不要重复启动V1或停止warm_daemon。

产物根：`/data1/decur_runs/BLCA_FOLD0_COMPARATORS_20261113_V1/`。
转发的plan短SHA：`8c68026a…48f3`；抽样短SHA：`6da86298…e594`，两臂相同。完整值仍须读取原文件，不能用短SHA占位或重新生成V1计划。

## 数值与方向

| 配置 | epoch | Harrell C↑ | IPCW C↑ | 生存NLL↓ | Brier↓ | IBS legacy↓ | AUC legacy↑ |
|---|---:|---:|---:|---:|---|---:|---:|
| aux-off best | 8 | .735341 | .521562 | .796395 | .109663 / .166142 | .137902 | .852127 |
| aux-off last | 30 | .698891 | .508145 | .879383 | .104583 / .156933 | .130758 | .801435 |
| CAMR best | 20 | .786846 | .375193 | .908270 | .096239 / .149177 | .122708 | .894954 |
| CAMR last | 30 | .759113 | .359790 | 1.021068 | .102867 / .141168 | .122018 | .888317 |

CAMR−auxoff：最佳轮C **+.051505**、IPCW **−.146369**、NLL **+.111875**；末轮C **+.060222**、IPCW **−.148355**、NLL **+.141685**。NLL更高是变差；Brier/IBS更低才是变好，不能把所有指标都说成“更高更好”。

当前只支持：CAMR双模态适配版在该验证集的Harrell排序点估计更好，而IPCW与生存似然表现更差，存在明确指标冲突；不能宣布整体生存预测能力更好。best和last来自同一批患者及同一训练轨迹，不是独立重复；差值未经配对不确定性评估，不能称显著。RNA-only及DCT旧数字没有进入同一锁定plan，不混排。

本批 **mdecur_aux_off不是完整方法**。已发布的 `ae17edc` 才将默认主方法设为 `mdecur_full_dense`（11旧＋4dense辅助），本次转发没有它的患者成绩。V1 batch8和新V2 batch20不可混排；V2中的同架构aux-off用于归因，不能用本次旧架构aux-off替代。

## 必须核实的评估矛盾

旧生存列/时间网格错位会影响Brier、IBS与动态AUC；legacy值原样保留，不能因metric errors为空就认为时间语义正确。AUC旧实现实际仅约15.12月单点，IBS旧短网格约8.09–15.12月。

Harrell与IPCW都使用`-sum(S)`，上述时间网格问题不解释它们的排序反转。IPCW均低于.5只描述当前权重下的配对排序，不能直接推断人群真实性能低于随机，也不能宣布指标坏。需看训练删失KM的G、平方逆权重、真正有后续可比较患者的事件贡献及最大单事件占比；不要事后按分数选tau。

新增脚本 **`scripts/audit_v1_comparison_predictions.py`** 专门适配V1的两条臂×best/last：

- 只读manifest、双精度reference、原plan、comparison、两份report、四份预测；checkpoint只做字节哈希，不加载模型、RNA数组或patch。
- 核对plan/输入哈希、同一训练抽样、患者身份/标签、训练拟合分箱、完整history、最早峰值及epoch30。
- 复算四份预测的Harrell/IPCW和legacy时间指标，核对表格与报告，另列finite-edge审计和KM权重集中度。
- NLL只核对report/history/summary一致；没有logits时不声称从累计生存CSV精确复原原训练似然。
- 不覆盖旧文件。失败写failed及原因，不能把已核对的部分伪装成全批通过。通过也不证明上游预处理、特征来源完全无泄漏或Git版本已经核实。

## 先保存服务器两个提交

本机2026-11-16实查GitHub main=`ae17edc76a9c5173a9945bcf299f80b1a9738f3b`；本地main与其一致。服务器报告HEAD=`111a241`、在8455875上有两个提交，但当前GitHub API及main获取不到该对象。因此尚不知道这两提交内容，不能推定它们只有文档、也不能假装已整合。

请服务器窗口先只读返回：

```bash
cd /data1/MDeCUR-Surv-ldt
git status --short
git rev-parse HEAD
git log --oneline 8455875..HEAD
git diff --stat 8455875..HEAD
```

为该HEAD建立保留分支；不要reset/clean/force push，不先rebase覆盖，也不要把未知患者产物提交公开。可先 `git fetch --no-tags origin main`，再 `git show origin/main:V1_RESULTS_REVIEW_20261116.md` 阅读新文档而不改工作树。核对训练/GPU与后台任务，确认无人依赖该工作区后，再审阅两提交的具体改动、以保留它们的方式合入当前main，处理冲突并核验。若改变过训练/评估文件，要解释V1源码指纹与“训练commit8455875”的关系。

## 安全更新后只读复算V1

使用现有trisurv环境，不安装升级；入口绝对路径如下：

```bash
python /data1/MDeCUR-Surv-ldt/scripts/audit_v1_comparison_predictions.py \
  --manifest /data1/decur_runs/PARITY_DCT_FOLD0/manifest.csv \
  --survival-reference /data1/decur_runs/PARITY_DCT_FOLD0/survival_reference.csv \
  --comparison-root /data1/decur_runs/BLCA_FOLD0_COMPARATORS_20261113_V1 \
  --out /data1/decur_runs/BLCA_FOLD0_COMPARATORS_20261113_V1/saved_four_prediction_audit_20261116.json
```

默认读取根目录`plan.json`。若实际归档plan另有路径，使用`--plan 原始计划文件`；不可生成新plan替代。输出必须是不存在的新文件。如果labels/risk/表格/历史指标不一致，先排查该问题；不启动新的训练来掩盖。

本批只增加上述只读审计及文档，没有修改模型、损失、采样或训练器，也没有启动任何患者任务。后续完整模型的一折训练继续按 `FULL_DENSE_TOKEN_PROTOCOL_20261114.md`，保持全部辅助默认开启、不扩五折；先解决代码版本与评估核查，不等待MurreNet/MoHD源码来阻塞已实现主方法。
