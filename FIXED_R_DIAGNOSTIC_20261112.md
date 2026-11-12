# 固定R检查点分解诊断：服务器执行交接

范围：仅冻结检查点推理与诊断；不训练、不选新轮次、不做五折、不加载outer_test。使用已完成的新R：`/data1/decur_runs/NEW_VERSION_R_FOLD0_SEED17_20261111/`，不是旧参考或重新训练I。沿用原244/60 manifest和checkpoint内的模型/RNA预处理参数。

## 1. 诊断对象与固定协议

脚本为 `diagnose_fixed_r_decomposition.py`。一次运行9种情况：原query；seed20261110的5次无固定点置换；训练集固定query；均匀patch权重；固定R仅关闭交互输出。5次置换是原 `numpy default_rng` 协议，不据结果重抽或选择种子。

所有query替换均保持目标患者病理、全局RNA、原RNA通路value不变。训练固定query只使用训练患者RNA，经过保存的scaler/tokenizer/match映射，先平均各患者单位query，再对每通路均值归一化。均值范数不大于1e-6时预定使用零query，产生该通路的均匀权重，并记录路径索引；不会按验证表现选择fallback。

均匀情况在评分处用零query，不改病理value。关闭交互情况直接使用同一个R模型的基础头输出，原始匹配和原始交互仍作为诊断记录，但实际交互贡献为0。它不是重新训练I，不能把它与前一批I臂混为同一种干预。

## 2. 输入与输出守卫

- manifest必须只有train/val，发现任何test行，在打开RNA或特征前拒绝。
- 显式核对checkpoint SHA，checkpoint/report的manifest与cohort指纹、结构、配置、scaler和划分标记。
- 从训练RNA重新拟合scaler作一致性检查；不以其替换保存值。
- 原预测必须复现存档的val_predictions与NLL/C/Brier；通过后才执行干预。
- 模型eval、参数关闭梯度、全程inference_mode；检查前后state字节哈希相同，输入文件哈希也必须保持不变。
- 输出JSON只能写新路径；已有文件直接报错。失败时保存日志，不改report或放宽容差绕过。

## 3. 执行步骤

1. 在 `/data1/MDeCUR-Surv-ldt` 核实本项目进程、git状态及当前GPU占用，保护4个未跟踪合成结果和全部旧实验目录。拉取Windows最终消息指定的提交，核对完整SHA。若已有新任务或跟踪修改，保留并先报告，不强制覆盖。
2. 使用现有trisurv解释器，至少运行 `python -m unittest test_fixed_r_decomposition test_diagnose_pathway_query_permutation`；完整回归为 `python -m unittest discover` 及 `python -m unittest scripts.test_make_development_folds`，本地合计156项通过。检查均为CPU合成数据，目标机须记录自己的实际结果。
3. 在 `/data1/decur_runs` 创建全新目录，如 `FIXED_R_DECOMPOSITION_20261112_<短SHA>`。独立记录新R的checkpoint SHA，并与已有provenance/归档哈希核对。若只有本次事前SHA，明确它仅固定本次文件一致性；不要误用旧参考的`c48e3d...`哈希。
4. 核对实际新R目录直接包含checkpoint.pt/report.json/val_predictions.csv。本批不需要旧消融`model_output/`目录。
5. 优先使用实时空闲GPU；没有空闲GPU可用CPU推理，不影响其他项目，不改chunk、不采样patch。沿用现有trisurv和命令级LD_LIBRARY_PATH。

仓库根目录命令如下（`${RUN}`为新输出目录，`${R_SHA}`为上述新R checkpoint完整SHA，`${DEVICE}`为明确设备）：

```bash
python -u diagnose_fixed_r_decomposition.py \
  --manifest /data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/fold0_train_val.csv \
  --run-dir /data1/decur_runs/NEW_VERSION_R_FOLD0_SEED17_20261111 \
  --checkpoint-sha256 "${R_SHA}" \
  --device "${DEVICE}" \
  --out "${RUN}/fixed_r_decomposition.json"
```

若物理GPU1确认可用，可设置 `CUDA_VISIBLE_DEVICES=1` 并使用 `--device cuda:0`；CPU则明确 `--device cpu`。若CUDA需要预先建立上下文，检查已验证wrapper如何初始化，然后为本脚本准备新包装。**原 `run_pathway_with_measure.py` 若固定runpy执行train_pathway.py，不能直接套用**；包装必须执行本诊断脚本，记录其源码/命令，不启动训练、不改核心模型或环境。

## 4. 输出如何解读

每种情况报告：NLL/C/Brier及相对原预测差值，H风险C/风险数值审计；匹配token、基础logit、原始交互、实际加权交互、最终logit、H12、S12的变化；patch注意力L1；逐通路注意力变化和匹配token L2变化。

变化分布以患者为单位等权汇总均值、中位数、P90、最大值。相对L2是每名患者完整张量差范数除原张量范数；原范数不大于1e-8者不计算比值，单独计数。跨通路或patch不构成新增患者。原始交互在off情况下保留，实际贡献明确置零，避免误以为raw不变说明干预失败。

注意力L1用两遍完整分块计算：第一遍各query/通路的全bag logsumexp，第二遍归一化权重并累加绝对差。只保留[情况,通路]统计，不物化全bag注意力表；检查每组权重总和为1，L1理论范围[0,2]。计算使用float64归一化。所有reference patch均参与，仍按checkpoint记录的chunk大小执行。

解释按证据区分：匹配/注意力与预测都变化小，支持该检查点对query不敏感；匹配变但交互/预测变化小，提示后续映射、平均或分支敏感性值得检查；固定R关闭交互的变化不能等同重新训练I的效果。固定query或均匀在当前验证集更好，也不能直接选为最终模型，需后续预定训练对照。

本诊断仍复用选轮验证患者和一个固定模型，不是独立测试、因果效应或空间表达测量。先回报本批结果，再决定后续研究，不自动训练或扩展五折。

## 5. 回报与追溯

回报完整提交SHA、实际设备/解释器/命令/wrapper、新R checkpoint SHA、原预测最大复现差、训练固定query低范数fallback、模型前后哈希、全部9种情况的匹配/交互/logit/H/S变化与指标表。记录耗时、日志、失败及输出SHA；患者原始文件和模型权重不上传GitHub。

新增9项针对性检查通过，随后完整156项CPU合成/回归检查通过（0失败/错误/跳过，约106秒）；两者不能相加为165个不同检查。脚本及测试的本地回执见 `results/fixed_r_release_validation_20261112.json`。训练实现未改变。既有文档路径修复和上一批证据解释一并交付，不能据阅读旧协议再次启动已完成的R/I训练。
