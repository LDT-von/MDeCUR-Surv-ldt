# 癌症代码仓库的会话接续

2026-11-13推送完成（完整SHA a2925c3e...）：已push `main`到origin，远端新增提交c08dcac（服务器端AI的梯度审计复核与只读提取器）已先fetch并rebase整合，无冲突、无强推。origin已从HTTPS改为SSH（`git@github.com:LDT-von/MDeCUR-Surv-ldt.git`），此前HTTPS无凭据无法推送。变基后完整测试176项+28子测试通过、0失败。`search_harness/`为本次新增（Stage1–3调度器，原先只在运行目录），全部路径经`SEARCH_*`环境变量可覆盖，已在真实队列上验证`progress.py`/`reconcile_queue.py`/claim-release逻辑/命令行生成与原运行一致；仓库版`run_pathway_with_measure.py`已在真实fold0跑通1轮（5090，352s）。

2026-11-13服务器端交接与推送：先读`SERVER_HANDOVER_20261113.md`。本轮push `HYPERPARAM_SEARCH_PROTOCOL_20261112.md`（预注册协议）、`diagnose_interaction_contribution.py`（参数化诊断CLI，本地复现原数字：完整C=0.645503、base-only C=0.524868、交互logit跨度占比1.3547）、`results/interaction_contribution_diagnostic_20261113.json`、`results/repeat_cancer_v1..v4.json`（合成，无TCGA标识）及`SERVER_HANDOVER_20261113.md`。注意raw交互5.34×0.2=1.068与base2.86不同尺度不可直接比，可比量是logit跨度占比1.3547；这是单检查点选轮验证的机制诊断，不是重训interaction-off的效果估计。协议：Stage1单折31配置家族配额筛选→Stage2五折OOF(SE≈0.011)→Stage3 seed29/43方差分解；patch_chunk_size 2048→8192提速约3倍、数值差≤2e-5。机器曾重启致队列中断，已重启5 worker+warm_daemon并reconcile；本机未跑患者训练/CUDA，未触碰outer test，未改训练实现与loss系数。

2026-11-12真实R训练批次梯度入口交付：`diagnose_fixed_r_gradients.py`，先读`FIXED_R_GRADIENT_AUDIT_20261112.md`。用户已要求本地验证后push并由服务器窗口运行；本轮代码/合成结果与此前本地API一并纳入交付。CPU计划→按计划哈希审计两步，固定seed20261112的前3训练批次、每批repeat11/22；真实R为3×20人、6记录，全patch原chunk。只加载train，val/test特征/RNA/标签不读，无optimizer/step；逐记录state/.grad和结束RNG/输入守卫，逐项梯度重构校验。旧checkpoint无独立训练内容哈希，报告明确历史合并指纹未重算，另锁定本次plan→run训练指纹。新增8项CLI检查通过；完整170项CPU检查0失败/错误/跳过，回执`results/fixed_r_gradient_release_validation_20261112.json`。未在本机运行患者或CUDA；不五折、不改loss系数。发布状态和完整提交SHA以本轮最终回报为准，下方“尚无CLI/未推送”为此前阶段。

2026-11-12固定R服务器结果已收到并审阅：见`EXPERIMENT_REVIEW_20261112.md`最新补充。用户转发689e2af批次156项通过、9干预完成；新R SHA为6c21fcf5…1f4bde84b，与前批归档一致（据报告）。冻结R关闭交互C降至0.524868，但训练固定query仍0.645503；注意力L1约0.7不等于匹配表示巨变（相对L2约3.5%–3.7%），raw交互5.34需乘0.2为1.068，不能直接与base2.86比较后称主导。重训I仍0.647619，旧配对区间含0；当前只是选轮验证，未证明机制增量。真实R逐项梯度尚未计算；本地API/162项回执仍未提交/push，本次10源码SHA核对一致，未改程序或重跑测试、未SSH、不五折。以下“服务器继续跑/尚未运行”固定R表述为历史。

2026-11-12本地逐项梯度审查：见 `GRADIENT_AUDIT_20261112.md`。新增 `pathway_gradient_audit.py` 与合成脚本；生产objective可选返回同一次前向的原始分项，默认损失/RNG/backward逐元素一致已验证。完整162项CPU检查通过；本轮未提交/推送、未连接服务器，真实R训练批梯度尚无结果。随机初始化合成例中局部common可能负内积，不能按此给患者损失排名或删项。服务器继续跑689e2af固定R分解，不展开五折。

2026-11-12固定R诊断交付：用户要求先做固定R分解，不展开五折。入口 `diagnose_fixed_r_decomposition.py`，先读 `FIXED_R_DIAGNOSTIC_20261112.md`，使用新R而非旧参考/I；原query、5置换、训练固定query、均匀权重、冻结R关闭交互共9种情况。训练固定query仅用训练RNA；原预测守卫/模型状态/输入哈希均检查，注意力L1两遍流式计算。本地156项CPU检查通过；目标机患者结果仍未运行本项。训练实现未改变，旧批次已经完成，不按旧文档重复启动R/I或自动五折。

2026-11-12最新回报：b4d9537批次据用户转发已完成，目标环境147项通过；旧结果配对、5次query诊断、新版本R/I及配对均完成，见 `EXPERIMENT_REVIEW_20261112.md`。所有比较仍是同60名选轮验证患者；辅助/交互增量尚无明确证据。不要重复启动上一批，亦不自动扩展五折。两条旧消融report实际在`model_output/`下，旧交付文档已更正。以下“尚未运行”段落为此前历史。

2026-11-11服务器交付：先读 `SERVER_NEXT_STEPS_20261111.md`。用户转发核查确认旧参考/matched-off/uniform三臂均已完成，结果仅为同60名患者的选轮验证；interaction-off/query/配对分析尚未运行。下一批先做旧结果分析，再在同一新版本上配套训练参考与interaction-off；不混用新旧协议、不自动扩展五折。新shuffled随机流和指标元数据已记录，差值权重仍未启用。发布前147项CPU合成/回归检查通过，真实目标环境仍须核验；所有既有患者实验产物与未跟踪合成文件须保留。

最新用户明确要求完整新框架而非继续停留于基线：新增 `train_pathway.py`、`cancer_pathway.py`、`cancer_pathway_stream.py` 和固定Reactome顶层GMT，见 `PATHWAY_LOCAL_FRAMEWORK.md`。包含RNA通路编码、条件patch匹配与交互、病理独立锚点和条件匹配的局部共同/独有、患者级DeCUR和病理局部一致性；支持单视图特征噪声或真实cached_uni D4，前者不能当成图像增强。目标机尚无新框架患者运行回执，不可宣称有效或首创；旧single_view基线保留。

最新阶段：用户报告39b0563的fold0 mask融合基线已完整运行（19轮、best4、val NLL2.186943/C-index0.6307，非测试），8个RNA坐标屏蔽有效，外测未加载。下一步显式 `--modality both|pathology|rna` 对照已实施，旧both初始化/路径保持兼容，单模态仅生存目标、跳过未选分支；完整102项CPU检查通过，见 `MODALITY_CONTROLS.md`。新单模态目标机结果尚未收到。

最新：用户已转发5090单视图fold0 244训练/60验证两轮CUDA成功，仍非独立测试/局部目标证据。训练常量基因导致验证标准化值约2.174e8，后续已实现可选 `--rna-scale-policy mask --rna-min-std 1e-6`，保存train-only mask、推理/辅助噪声共用，见 `RNA_PREPROCESSING.md`。完整93项合成/回归检查通过。旧clamp行为兼容；旧的“尚无患者训练回执”段落为历史状态。

本仓库是WSI＋RNA的**癌症生存预测研究实现**，不是原始DeCUR仓库，也不是旧槽机制项目。

若以整个工作区打开：先阅读父目录的 `AGENTS.md`、`START_HERE.md`、`PROJECT_HANDOFF.md`，并核对 `WORKSPACE_SNAPSHOT.json` 的日期。完整交接以父目录的 `PROJECT_HANDOFF.md` 为准，用户新指令优先。

如果只复制／打开本子仓库而父目录文件不存在，至少先读 `CANCER_MODEL_README.md`、`START_UNI_VIEWS.md`、`uni_views/README.md`、`uni_views/VALIDATION.json`。这些文件不包含完整的DenseTwins讨论；需要取得整个工作区交接，不能假装已看过。

- 当前主模型是 `cancer_model.py: CancerDeCUR`；`model.py`还保留旧原型类，但其中辅助损失仍被新模型复用。
- 已实现：同patch图像双视图的冻结UNI v1提取，及`--path-views cached_uni`缓存接入。此模式保留全部patch，要求同次提取的reference＋A/B。
- 2026-11-09 后续已实现逐patch局部相关目标（`--lambda-local`，要求cached_uni）、预测与辅助投影分离（`--prediction-space encoder`）、病理／RNA独立within系数。完整公式、论文及开关见 `LOCAL_GLOBAL_IMPLEMENTATION.md`。默认保留历史投影预测、局部项关闭。
- 后续已实现 `--training-backend stream`，包括lazy H5、患者小批量、checkpointed patch分块与完整局部中心矩。见 `STREAM_TRAINING.md`。固定患者batch下分块与完整计算在容差内一致；患者minibatch会改变相关统计覆盖范围，不能称与原全队列目标完全等价。
- 2026-11-10新增 `--path-views single_view`：安全mmap现有PT的完整bag普通融合，强制关闭辅助项，不声明UNI来源。stream专用 `--validation-only`支持train/val清单，不加载test特征/RNA或解析其标签。见 `SINGLE_VIEW_TRAINING.md`，最新完整85项合成/回归检查通过；5090数据与基础CUDA kernel信息仅来自用户转发，尚无本项目患者训练回执。
- 最新一次完整70项合成／回归检查通过（既有51＋本轮19），回执为 `results/stream_validation_20261109.json`，另有CPU合成峰值内存／耗时基准；旧回执保留历史状态。用户已确认真实数据仍在另一台电脑，尚无真实UNI、SVS、患者训练或CUDA成功回执。
- 稳定性加权只是候选。对增强稳定不等于预后可靠，零相关不等于独立。解释网络时同时交代预测和辅助路径。
- 用户要求独立思考，不从旧SlotSPE／槽机制方法引入设计。不因另一个项目里有脚本就把其方法并入本仓库。
- 当前有未提交／未跟踪工作；检查Git状态，保留已有改动。不要reset、clean或强制覆盖，不声称新功能已push。
- 另一台有WSI的电脑此前断电；历史报告不等于当前状态。实际患者提取／训练与环境替换按用户明确授权的范围进行，不默认全量启动；已给出的授权不必重复询问。
- 结构性代码查询遵循父目录的CodeGraph规则；若工具未启用，使用现有搜索读取并如实说明。
