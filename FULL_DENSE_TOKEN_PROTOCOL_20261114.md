# 完整辅助默认开启、token 稠密解耦与历史高分核查

本文件取代上一批V1中把 `mdecur_aux_off` 当主比较臂的安排。用户2026-11-14明确要求：主方法的辅助必须默认开启，增加DenseTwins式token稠密解耦，并研究局部token的跨模态协同。只做BLCA legacy fold0，不补其他4折，不在本地窗口执行患者作业。

## 主方法及实际训练路径

默认入口 `/data1/MDeCUR-Surv-ldt/comparison_fold0.py run` 不写 `--model` 时，选择 **`mdecur_full_dense`**。显式可选 `mdecur_dense_aux_off` 是同预测架构的辅助消融；`mdecur_aux_off` 是历史架构消融，均不能冒充主方法。CAMR使用 `camr_2m`，MurreNet/MoHD仍无已核实官方源码入口。

主模型实现为 `cancer_dense_model.py:DensePathwaySurvival`，继承旧 `PathwayLocalSurvival` 并保留原病理attention、全局RNA、通路token、条件patch匹配及交互预测。新增两条预测路径：

1. 对匹配的病理表示与RNA通路token做局部投影，得到 `ZP=[CP,UP]`、`ZR=[CR,UR]`；`mean_k([(CP+CR)/2, UP, UR])` 直接进入 `decomposition_head` 并加到hazard logits。因此共同/独有坐标接受生存NLL的直接梯度，而不再仅仅作为辅助投影器。
2. `token_cooperation` 用训练期参考均值中心化的 `UP,UR`，经过各自无偏置线性映射、逐元素乘积、token均值及无偏置预测头，形成附加交互项。参考均值仅在训练前向更新，验证及重载评估冻结；与原交互项一样乘 `interaction_strength=.2`。

这种乘法项可以表达XOR式非加性关系，但不是严格信息论synergy/PID估计。当前预测病理token来自RNA条件匹配，因此整个预测式不能声称分离出了纯单模态主效应和纯协同效应。常见信息量的“联合减两个边际”混合了协同与冗余，也不能直接叫synergy。临床增量需要同协议重新训练对照；只看乘积结构、热图或固定检查点关闭分支不足以证明。

## 辅助默认全部开启

`DensePathwayConfig` 默认开启原有11项：global_path_within、global_rna_within、global_common、global_unique、patch_within、pathway_path_within、pathway_rna_within、pathway_common、pathway_unique、matched_common、matched_unique；其中global_common默认stability加权。

另外默认开启4项：

| 新项 | 默认基础权重 | 目标 |
|---|---:|---|
| dense_path_within | 0.0005 | 病理两视图的完整`[C,U]` token相关趋向单位阵 |
| dense_rna_within | 0.0005 | RNA通路两视图的完整`[C,U]` token相关趋向单位阵 |
| dense_common | 0.001 | 跨模态匹配后的共同坐标相关趋向单位阵 |
| dense_unique | 0.001 | 跨模态匹配后的特有坐标相关趋向零阵 |

跨模态dense两项以 `min(1,epoch/5)` 乘基础权重；第1轮即为正，5轮后达到基础权重。所有15项都有实际objective调用和日志，不以填入配置代替参与训练。

代码区分 **enabled** 与 **estimable**：默认开启并不保证每个批次每个方向都有有效方差。global_common若退化并被底层跳过，完整运行会明确失败，不会静默删loss继续给“完整模型成绩”。Dense低方差仍保留有限惩罚并逐患者标 `valid=False`，不把低方差零相关当解耦成功。服务器必须回报这些比例、unique matched token数、损失及梯度，不能只回报“15项开了”。

## DenseTwins如何作用于token

来源：<https://arxiv.org/html/2611.04710v1>，§4、附录B；官方<https://github.com/CarloSgaravatti/DenseTwins>本次仍仅README。此处是论文启发的WSI/bulkRNA适配，非作者代码数值复现。

`cancer_dense_tokens.py` 对每位患者分别计算：模态内先在各自编码特征中做单向cosine最近邻，再对匹配投影的完整`[C,U]`做BT；跨模态在共同子空间做单向最近邻，对匹配的common做BT、unique做DEC，平均四种视图组合。索引不求导，匹配值求导，多对一允许、数量与低方差均记录。采用当前项目的中心化和epsilon相关，而非声称与正文未中心化式子逐字相同。

病理dense项每患者最多取 **256个确定性等间隔真实patch token**，排除预算补零；RNA使用全部通路token。限制token预算是工程选择，不得称全部4096个patch都进入dense NN。损失先在每患者内部计算，再患者等权平均；token不是新增独立患者。原有pathway项仍沿患者轴计算，二者统计轴必须分开描述。

Dense匹配前的两侧token各自只读本模态，避免用RNA条件病理表示去制造共同目标。最近邻选择本身仍会产生选择偏差，不能作为真实空间对应或共同生物因素的证据。Bulk RNA不是空间转录组，通路token没有与patch逐位置的实测配对。现有PT双视图是编码后特征噪声，不是WSI图像增强。

本地已知答案负控 `scripts/check_dense_matching_null.py`：两组完全独立生成的token云，按原索引相关均值约-0.00693，最近邻匹配后约0.99976。这证实高post-matching相关不能单独作为“学到共同信息”的证据；不会据此删掉用户要求的dense训练目标，而是把它作为必须保留的失效边界。

`scripts/check_token_cooperation_synthetic.py`另外对XOR、冗余、单模态、零信号各做3个固定seed、300步训练，独立新测试8192样本，仅结束后评估。XOR的加性头测试NLL均值约0.69383，加入生产token交互组件约0.32918；完整RNA错配置换约1.17492。冗余/单模态/零信号没有稳定交互增益。它是无删失、两类事件时间的categorical toy，不是临床hazard目标；交互臂132参数、加性臂84参数，也不是容量匹配的临床效应估计。

## V2共同协议

同一BLCA患者、特征、RNA、DSS终点、4箱训练分位数、4096patch、raw RNA policy none、离散NLL alpha=.5、seed3、30轮、lr/wd5e-4、AdamW与Cosine30保持。**患者batch统一从V1的8改为20**：全局within维数16需要必要条件B>=17；303人按20分批，末3人并入前批，得到14批20＋1批23。所有比较臂使用完全相同的顺序与合并规则。不能把V1 batch8结果混入V2排名，也不保证复现历史0.7361。

新协议ID `blca_fold0_full_dense_v2`，需要重新生成plan，不能复用V1 plan。主比较为full dense、同架构aux-off、CAMR，各一折30轮；不额外启动历史aux-off或RNA-only。若已有V1任务运行，先保留其进程/结果；**不要在读取工作区源码的旧任务运行中pull本批**，否则旧计划的源码守卫应拒绝发布结果。等安全后更新，另用新结果目录。

新架构不是旧0.7361检查点的继续训练。旧固定R专用诊断器不能直接假定能加载DensePathwaySurvival；本批使用新比较器的保存与重载路径。旧parity的`final_validation`保留为best复评兼容字段，新增明确的`last_validation`、best/last检查点及预测，不再把这个旧字段误当epoch30。

## 先核查0.7361，再训练新主方法

高Harrell C本身不能证明bug。本地已经用已知答案确认：`-sum(S)`风险方向正确；旧代码有生存列/评估时间错位，会影响Brier/IBS/AUC，但**不影响当前Harrell C路径**。旧comparable_pairs来自另一实现，时间并列时也可能与主C分母不同。IPCW未设tau，少数带大权重的晚期配对可使它远离Harrell，不能直接宣布IPCW失效。

新增 `cancer_parity_audit.py`：S第j列对应有限内部训练边界`edges[j+1]`，保留支持范围内的有限边界，不给开放末箱伪造val_max时间。旧legacy指标保留，另列 `finite_edge_audit`；该修正不能消除val选轮偏差，也不能重写历史C。新增报告真实sksurv配对数和训练删失KM权重集中度。`train_pathway_parity.py`还补重复/非法reference标签和manifest对齐校验，存reference哈希；其默认也是dense full，旧架构需显式 `--legacy-architecture`。

服务器先只读核对旧预测，输入文件如下；入口只读CSV/report和checkpoint字节哈希，不反序列化权重、不读patch/RNA数组、不训练：

```bash
conda activate trisurv
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python /data1/MDeCUR-Surv-ldt/scripts/audit_parity_saved_predictions.py \
  --manifest /data1/decur_runs/PARITY_DCT_FOLD0/manifest.csv \
  --survival-reference /data1/decur_runs/PARITY_DCT_FOLD0/survival_reference.csv \
  --reference-run /data1/decur_runs/PARITY_DCT_FOLD0/run_fold0_30ep \
  --out /data1/decur_runs/PARITY_DCT_FOLD0/saved_prediction_audit_20261114.json
```

输出必须是新文件。若患者/标签/risk/报告C任一不一致，先回报failed及原因，不继续启动新训练，也不替换原产物。通过只说明本次CSV/标签/报告自洽，**不证明上游预处理、特征来源、患者匹配完全无泄漏**；旧封存76人与新303/77的交集仍需由运行现场另外核实。

## 服务器新实验精确入口

仓库 `https://github.com/LDT-von/MDeCUR-Surv-ldt`，实际代码目录 `/data1/MDeCUR-Surv-ldt`；先只读检查Git、进程、GPU，安全时ff-only更新，不reset/clean/杀其他任务。使用现有trisurv，不安装升级。

先工程验证：

```bash
cd /data1/MDeCUR-Surv-ldt
python -m unittest test_cancer_dense_tokens test_cancer_dense_model \
  test_cancer_parity_audit test_saved_parity_audit test_comparison_fold0
```

旧预测核查通过、无其他任务冲突后，在同一个带`set -euo pipefail`、前述环境变量的shell中执行：

```bash
COMPARE_REPO=/data1/MDeCUR-Surv-ldt
COMPARE_ROOT=/data1/decur_runs/BLCA_FULL_DENSE_FOLD0_20261114_V2
test ! -e "$COMPARE_ROOT"
mkdir "$COMPARE_ROOT"
python "$COMPARE_REPO/comparison_fold0.py" plan \
  --manifest /data1/decur_runs/PARITY_DCT_FOLD0/manifest.csv \
  --survival-reference /data1/decur_runs/PARITY_DCT_FOLD0/survival_reference.csv \
  --reference-run /data1/decur_runs/PARITY_DCT_FOLD0/run_fold0_30ep \
  --out "$COMPARE_ROOT/plan.json"
COMPARE_PLAN_SHA=$(sha256sum "$COMPARE_ROOT/plan.json" | cut -d ' ' -f 1)
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

新GPU/主模型先可用独立输出目录加`--smoke`做一轮CUDA工程检查；smoke不能进入正式表。完整运行失败应保留新目录和log，不删除/覆盖来掩盖退化。只有一折，不五折，不为更高C更换seed/超参。

最终回报6行（3模型×best/epoch30），包含C、NLL、IPCW C、legacy Brier/IBS/AUC与finite-edge修正指标、真实配对数和KM权重摘要；每项来自同一选定checkpoint，不各挑峰值。还需报告15项实际raw/weighted损失、global_common跳过、dense valid比例、token匹配数量、local projector/decomposition/cooperation梯度、参数量、耗时、共同plan及patch抽样SHA。主方法分数是否提高、是否超过RNA信号，等待真实结果，不预先承诺。
