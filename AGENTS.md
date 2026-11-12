# 癌症代码仓库的会话接续

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
