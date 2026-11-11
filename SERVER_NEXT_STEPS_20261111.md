# 5090 拉取更新与下一批实验

这是给服务器助手的可执行交接。用户已要求在核实运行情况后推送Windows更新，随后由服务器窗口拉取并开展预定实验。Windows助手本次没有连接服务器。以下现状来自用户转发的服务器只读回报，不是本机独立复核。

## 1. 已知服务器状态与结果

核查时间为2026-11-11 14:26 UTC（北京时间22:26）。项目 `/data1/MDeCUR-Surv-ldt` 为main，HEAD `c5c476e3d51164e928b15a3821acfc3dda335eae`；当时无本项目训练进程。仓库内仅4个未跟踪合成结果 `results/repeat_cancer_v1.json` 至v4，须保留。实验产物位于 `/data1/decur_runs`，不覆盖。

| 既有244/60、seed17实验 | 最终/最佳轮 | val NLL | Harrell C：1−S12 | Brier12 | 共同项跳过批次 |
|---|---|---:|---:|---:|---:|
| 完整参考 | 23/8 | 2.1695327759 | 0.6455026455 | 0.1667254852 | 2 |
| matched-off | 23/8 | 2.1722288132 | 0.6433862434 | 0.1670644836 | 5 |
| uniform-common | 23/8 | 2.1695315838 | 0.6455026455 | 0.1667250828 | 0 |

三者正常结束，外测未评估；输入指纹、RNA mask、主要配置据报告已核对一致。query置换、配对bootstrap和interaction-off尚未运行。单次小验证集的微小差异不足以建立有效或无效结论。

对回报的一处解释作校正：stability和uniform的共同相关都用A视图p1/r1；前者额外用p2/r2估计权重，并可能退化跳过。不是从“两视图共同相关”变为“单视图共同相关”。两模式实际改变权重和退化处理。

当时物理GPU0有无关KIRC任务，物理GPU1空闲；GPU占用随时会变，执行前重新检查。不得停止或影响其他任务。内折正确目录为 `/data1/decur_runs/DEVELOPMENT_INNER_FOLDS_SEED17_20261110/folds/`，回报中的`20260710`为笔误。

## 2. 更新与工程核验

1. 核实当前无本项目训练、无新增跟踪修改；保存HEAD、status和4个未跟踪文件的路径/哈希。若状态已变，保留修改并报告具体情况，不reset/clean/stash覆盖。
2. 允许给旧HEAD加本地tag `checkpoint-pre-windows-20261111`；若tag已存在先核对目标，不覆盖既有引用。
3. 在原目录执行 `git pull --ff-only origin main`。与Windows最终消息给出的完整提交SHA核对，一致后继续；若远端又变化，先核对，不盲目运行。
4. 使用已验证的trisurv解释器，复跑CPU合成检查。完整本地套件的命令为 `python -m unittest discover`，另加 `python -m unittest scripts.test_make_development_folds`；后者可能未被discover自动收集。最终发布前两者合计147项通过（0失败/错误/跳过），见 `results/server_release_validation_20261111.json`。145项为此前阶段回执，发布版又补了报告NaN守卫和指标依赖指纹的检查。保存目标机实际结果，不能用Windows结果替代目标环境验证。
5. 库外旧脚本保留；后续明确调用新仓库脚本，避免误用 `/data1/decur_runs/QUERY_DIAGNOSTIC_20261110/` 或 `PAIRED_PATHWAY_ANALYSIS_20261110/` 中的旧拷贝。

本版新增跳过原因、H风险诊断和指标口径；shuffled改用独立RNG。差值稳定权重仍是研究模块，未启用。合法输入下历史C/Brier公式未改变。新报告/stream_config增加协议元数据，不能删除守卫来强行比较新旧版本。

## 3. 先分析既有实验，不启动五折扩张

在 `/data1/decur_runs` 下创建全新阶段目录，例如 `POST_WINDOWS_AUDIT_20261111_<提交短SHA>`；若存在则使用新编号，不覆盖已有目录。保存命令、源码SHA、输入哈希、开始/结束状态和失败日志。

### 3.1 两项配对比较

在仓库根目录，使用训练解释器执行下列命令。`${RUN}`为上述新阶段目录的绝对路径，先创建。此阶段不加载特征或RNA文件，仅读取train/val manifest标签、report与保存预测。

```bash
python paired_pathway_validation.py \
  --manifest /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv \
  --reference /data1/decur_runs/BLCA_PATHWAY_LOCAL_FULL_FOLD0_SEED17_NEW \
  --candidate /data1/decur_runs/ABLATION_MATCHED_OFF_FOLD0_SEED17_20261110 \
  --comparison matched-off --bootstrap 2000 --seed 20261110 \
  --out "${RUN}/paired_matched_off.json"

python paired_pathway_validation.py \
  --manifest /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv \
  --reference /data1/decur_runs/BLCA_PATHWAY_LOCAL_FULL_FOLD0_SEED17_NEW \
  --candidate /data1/decur_runs/ABLATION_UNIFORM_COMMON_FOLD0_SEED17_20261110 \
  --comparison uniform-common --bootstrap 2000 --seed 20261110 \
  --out "${RUN}/paired_uniform_common.json"
```

任何原预测重算、配置或指纹守卫失败都先定位并保留失败，不改容差或改report绕过。返回candidate−reference差值、各指标95%区间及有效重采样次数。区间条件于固定模型和训练删失KM，不包含训练/选轮不确定性；同60人已用于选轮。

### 3.2 固定参考checkpoint的5次query置换

预定seed20261110、5次无固定点置换。病理、全局RNA、原患者RNA通路value保留，只替换用于patch评分的query。使用原参考checkpoint，先与归档provenance核对SHA；若归档没有checkpoint哈希，至少独立记录此次运行前SHA，并明确它只校验本次文件一致性，不能自动证明历史来源。

```bash
python diagnose_pathway_query_permutation.py \
  --manifest /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv \
  --run-dir /data1/decur_runs/BLCA_PATHWAY_LOCAL_FULL_FOLD0_SEED17_NEW \
  --checkpoint-sha256 "${EXPECTED_CHECKPOINT_SHA256}" \
  --device "${DEVICE}" --permutations 5 --seed 20261110 \
  --out "${RUN}/query_permutation_seed20261110.json"
```

GPU1仍空闲且适用时，按目标机已验证方式设置 `CUDA_VISIBLE_DEVICES=1`，此时`${DEVICE}`为`cuda:0`；CUDA上下文初始化沿用既有wrapper。先检查wrapper实际接口与原命令，不猜测参数。CPU执行也可，但记录实际设备/耗时，不能因此换模型或采样patch。原始预测复现检查不通过时先停诊断并查原因，不继续解释置换效果。

报告每次NLL/C/Brier差值及逐患者生存概率变化汇总。5次置换不作为5个独立样本；这是模型依赖诊断，不是因果效应，也不是空间表达测量。

## 4. 然后做同一新版本的参考与交互消融

不能将旧c5c476e参考与新版本interaction-off直接交给严格配对脚本：新stream_config和指标协议元数据不同。使用同一新提交、同一GPU、同一环境依次训练两臂，并各自按val NLL选轮。旧三臂结果保留为历史阶段。

- R：新版本完整参考，interaction_strength=0.2、11项辅助目标开启、common_mode=stability。
- I：只将interaction_strength改为0，其余与R一致；保留所有辅助目标。它检验预测交互分支的联合贡献，不能称为移除了全部局部/匹配信号。

共用参数如下；分别设置全新out目录与interaction_strength，不传evaluate-test：

```bash
python -u train_pathway.py \
  --manifest /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv \
  --genes /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/genes.txt \
  --out "${NEW_OUTPUT_DIRECTORY}" \
  --view-source feature_noise --feature-noise 0.05 --rna-noise 0.05 \
  --common-mode stability --strength 2 --min-weight 0.1 \
  --interaction-strength "${INTERACTION_STRENGTH}" \
  --hidden 16 --common 8 --local-dim 8 --local-common 4 \
  --global-path-within 0.0005 --global-rna-within 0.0005 \
  --global-common 0.001 --global-unique 0.001 --patch-within 0.001 \
  --pathway-path-within 0.0005 --pathway-rna-within 0.0005 \
  --pathway-common 0.001 --pathway-unique 0.001 \
  --matched-common 0.0005 --matched-unique 0.0005 \
  --patient-batch-size 20 --patch-chunk-size 2048 --device cuda:0 \
  --epochs 80 --patience 15 --seed 17 --starts 0 --horizon 12 --time-unit months \
  --endpoint 'Dataset-provided OS in months; time origin unverified'
```

使用trisurv原有命令级LD_LIBRARY_PATH和已验证CUDA初始化wrapper；GPU选择必须先实时确认。先保存实际两条命令和配置差异，再执行。不安装/替换环境，不占用正在运行其他任务的GPU，不静默关闭损失。新原因字段允许解释跳过；若一开始连续跳过或出现非有限值，保留日志诊断，不盲目跑满。

两臂完成后用相同脚本 `--comparison interaction-off` 做2000次配对bootstrap。同时比较新R与旧参考的轨迹，作为版本复现诊断；不以旧新任一版本最好值选择性报告。新R无需精确复现旧CUDA浮点轨迹，但明显偏差需解释。

## 5. 阶段边界与回报

本批范围为：目标环境工程核验、两项旧结果配对分析、一次5置换诊断、新版本R/I两次训练及其配对分析。先完成这些再决定内部五折候选，不自动扩展种子、内折或时间区间。

76名outer_test继续封存，train/val清单不得出现test行。内折fold_0为243/61，与本批244/60不同；未来五折必须每折配套重新训练参考，RNA scaler/mask各训练折独立拟合。

汇报包括提交完整SHA、实际命令和GPU、工程检查结果、每阶段成功/失败状态、所有输出路径/哈希、R/I最终与best epoch、验证NLL/C/Brier/H-C、共同项跳过原因、配对区间及query变化。失败和未完成同样报告；不只列最好指标。
