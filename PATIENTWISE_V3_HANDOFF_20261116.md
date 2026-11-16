# 逐患者 NLL V3：本地修复与 BLCA fold0 服务器交接

本次接续范围为本地修复、生成数据验证与普通 Git 发布。服务器命令由服务器窗口另行执行，只运行 BLCA legacy fold0，不运行五折、outer_test，不停止 warm_daemon 或重跑历史实验。已有五折源码保留，不属于本次执行范围。

## 修复与历史结果边界

旧 `discrete_nll` 的删失向量 `[B]` 与 gather 后概率 `[B,1]` 广播为 `[B,B]`，使混合事件/删失 batch 的患者梯度串扰。当前 `patientwise_clamped_discrete_v1` 逐患者运算，none 返回 `[B]`；保留 clipping 和 alpha 语义：事件权重 1，删失权重 `1-alpha`。独立手算、梯度、拆 batch、患者置换和更改他人删失状态的回归覆盖此漏洞。

旧 C 仍是旧预测的排序结果，不能作为正确逐患者 NLL 下训练的证据；重评分不能修复旧权重。旧 V1/V2 分数与产物保留，不与 V3 排名合并。本次没有验证 DCT 原训练器是否同错，没有取得新患者结果。

本次发现训练 `main()` 没有调用已存在的 `_configure`，仍执行旧重复分支；已接通实际入口：

- 所有 overrides/common-mode 先应用，再最终校验。默认保留原 11 项＋4 项 dense 辅助；单项置零、common-mode 改动或 aux-off 均记录为非 full 参考。
- 冻结输入必须显式 aux-off，训练与推理共用 `apply_checkpoint_controls`。RNA 使用保存的训练 scaler 后再归零；冻结病理保留原始 patch 数、padding、provenance。
- best、last、历史 best 别名 checkpoint 均保存 controls、预算、chunk、loss 版本和输入哈希；未知历史契约保持 unverified。
- `--batch-tail-minimum` 用于实际批次。legacy 对照 303 人/batch17/minimum17 为 `16×17＋31`；两个臂更新步数一致。主 V3 比较仍 batch20、共同 rank bound 合并尾批。
- 报告记录实际批量/步数/抽样哈希；比较器另存患者顺序、schedule 摘要、主 optimizer 与 CAMR discriminator 步数，汇总拒绝不同抽样/schedule/主更新步数。

接续基线已含 padding-only 分块长度修复和 SHA-256 sampler，本次保留并验证。诊断器按本次实际交互强度恢复 base/计算贡献，finally 恢复；支持 None，拒绝覆盖输出，使用保存 scaler/输入哈希并拒绝不支持的 dense checkpoint。observed-follow-up-bin ridge 是标准化观测随访箱的最小二乘，采用 dual 求解，不能称删失感知 NLL 生存基线。

## V3 共同协议

协议 ID `blca_fold0_full_dense_patientwise_v3`。必须生成新 plan、新目录，从头训练；不复用 V1/V2 plan/权重。

| 项目 | 固定值 |
|---|---|
| 划分 | BLCA legacy fold0 train/val；val 选轮，非独立测试 |
| 终点/种子 | DSS、months、seed3 |
| 训练 | 30 轮、AdamW、lr/wd5e-4、Cosine30 |
| 输入/目标 | 4096 patch、raw RNA policy none、4 箱逐患者离散 NLL、alpha=.5 |
| 共同批次 | batch20、同一患者顺序/patch 抽样、共同 full rank bound 合并尾批 |
| 三臂 | mdecur_full_dense / mdecur_dense_aux_off / camr_2m |
| 选轮 | 最高 val Harrell C，精确并列取最早；另报 epoch30 |
| 指标 | legacy 保留，finite-edge 时间对齐审计单列 |

full 的 local common/private 直接进入预测；token 交互是任务协同候选，不是已识别 PID synergy。高匹配相关不代表生物学共同因素或空间 RNA 对应。CAMR 是已披露适配的双模态比较，不是原论文结果复现。

## 服务器命令（本机未执行）

仓库 `https://github.com/LDT-von/MDeCUR-Surv-ldt`，目录 `/data1/MDeCUR-Surv-ldt`。先检查 Git、正在读取源码的任务和 GPU 占用；确认可以安全更新后 fast-forward。存在分叉或本地改动须保留核对，不 reset/clean/force。使用现有 trisurv，不安装升级。

```bash
cd /data1/MDeCUR-Surv-ldt
git status --short
git log -5 --oneline
git fetch origin
git merge --ff-only origin/main
git rev-parse HEAD
conda activate trisurv
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pytest -q test_discrete_nll_patientwise.py test_cancer_parity_bag.py \
  test_parity_checkpoint_controls.py test_parity_entrypoint_controls.py \
  test_parity_diagnostics.py test_parity_shell_safety.py test_comparison_fold0.py
```

旧 reference/checkpoint 仅锁定输入身份与文件指纹，不载入其权重继续训练。先核对现场是否已有相同新协议实验，避免重复。下面在同一个 shell 中创建一个 plan、顺序运行三个臂；确认 cuda:0 可供本任务使用，不杀占用进程。目录与日志必须是新的，失败后保留产物。

```bash
set -euo pipefail
set -o noclobber
COMPARE_REPO=/data1/MDeCUR-Surv-ldt
COMPARE_ROOT=/data1/decur_runs/BLCA_FULL_DENSE_FOLD0_20261116_V3
test ! -e "$COMPARE_ROOT"
test ! -L "$COMPARE_ROOT"
mkdir -- "$COMPARE_ROOT"
git -C "$COMPARE_REPO" rev-parse HEAD > "$COMPARE_ROOT/code_commit.txt"
python "$COMPARE_REPO/comparison_fold0.py" plan \
  --manifest /data1/decur_runs/PARITY_DCT_FOLD0/manifest.csv \
  --survival-reference /data1/decur_runs/PARITY_DCT_FOLD0/survival_reference.csv \
  --reference-run /data1/decur_runs/PARITY_DCT_FOLD0/run_fold0_30ep \
  --out "$COMPARE_ROOT/plan.json"
COMPARE_PLAN_SHA=$(sha256sum "$COMPARE_ROOT/plan.json" | cut -d ' ' -f 1)
printf '%s\n' "$COMPARE_PLAN_SHA" > "$COMPARE_ROOT/plan.sha256"
# CUDA 工程检查独立目录，不进入正式汇总。
python "$COMPARE_REPO/comparison_fold0.py" run \
  --plan "$COMPARE_ROOT/plan.json" --plan-sha256 "$COMPARE_PLAN_SHA" \
  --model mdecur_full_dense --device cuda:0 --smoke \
  --out "$COMPARE_ROOT/smoke_full_dense" > "$COMPARE_ROOT/smoke_full_dense.log" 2>&1
for COMPARE_MODEL in mdecur_full_dense mdecur_dense_aux_off camr_2m; do
  python "$COMPARE_REPO/comparison_fold0.py" run \
    --plan "$COMPARE_ROOT/plan.json" --plan-sha256 "$COMPARE_PLAN_SHA" \
    --model "$COMPARE_MODEL" --device cuda:0 --out "$COMPARE_ROOT/$COMPARE_MODEL" \
    > "$COMPARE_ROOT/$COMPARE_MODEL.log" 2>&1
done
python "$COMPARE_REPO/comparison_fold0.py" summary \
  --runs "$COMPARE_ROOT/mdecur_full_dense" "$COMPARE_ROOT/mdecur_dense_aux_off" "$COMPARE_ROOT/camr_2m" \
  --out "$COMPARE_ROOT/comparison.json"
```

回报三模型×best/epoch30 共六行，不能跨指标各挑峰值。包含 C、NLL、IPCW C、Brier/IBS/AUC、finite-edge 审计，另附共同 plan/源码/抽样/schedule SHA、实际批量与步数、辅助原始/加权项、common 跳过情况、dense 有效比例及梯度摘要。val 参与选轮，不报为独立测试或临床收益。

## 本地回执与保留文件

最终测试数量/源码 SHA 见 `results/patientwise_release_validation_20261116.json`；三臂各30轮合成 best/last 重载见 `results/patientwise_three_arm_validation_20261116.json`。仅 CPU 生成数据。本机未 SSH、未读患者输入、未启动患者训练/CUDA/五折/outer_test。

四个既有未跟踪文件保留原位：`local_patientwise_focused_20261116.log`、`results/_hash_seed_wrapper.py`、`scripts/capture_hash_draw_failure.py`、`scripts/check_draw_sensitivity.py`。其中后三个在运行期间消失，当前任务及子任务未删除它们；已按开场读取的完整文本恢复，未声称原始字节级校验。脚本为旧 hash sampler 故障复现/敏感性诊断，wrapper 显式 monkeypatch 回旧 sampler；未用于当前生产验证，不纳入本次发布。
