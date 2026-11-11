# 生存输出、删失指标与消融随机流审查

2026-11-11；接续本地机制审查。仅源码分析、已知答案计算和 CPU 合成/回归验证。未连接服务器，也没有重算既有患者结果。

## 1. 已确认的问题与修复

### 1.1 NaN 评估时间会产生虚假的零 Brier

修复前对 `ipcw_brier([1,3],[1,1],[1,3],[1,1],[.2,.8],NaN)` 的实际输出为 `value=0.0, g_horizon=1.0, at_risk=0, n=2`。原因是与 NaN 的时间比较全部为 False，没有样本贡献，最后返回零。该问题位于底层指标函数；完整模型 CLI 已验证有限 horizon，因此不能推断它影响过现有正常 CLI 实验。

已在 `cancer_metrics.py` 拒绝非有限/负时间点及非法 eps，并让 Harrell 指标拒绝负观察时间。合法输入下保留原公式。修复前 metrics 文件 SHA256：`310e66f708521d7fe33615b8856b2d1dc9d123a7bf93f2eec76894ee24bce033`。修改后已知答案和非法输入检查通过。

### 1.2 1−S 的舍入能改变 C-index

风险为 `1−S(h)` 时，float32 的减法可以在 S 尚未下溢前就将风险舍入为 1。设两人危险率为 `[3,2]`、观察时间 `[1,2]`、均发生事件：

| horizon | H(h) | 理论 S(h) | float32 1−S(h) | 原风险 C | H 风险 C |
|---|---|---|---|---:|---:|
| 1 | [3,2] | [0.04979,0.13534] | 可区分 | 1 | 1 |
| 12 | [36,24] | [2.32e−16,3.78e−11] | [1,1] | 0.5 | 1 |

新 `cancer_survival_diagnostics.py` 从 logits 直接在 float64 计算

\[
H_i(h)=\sum_k[\operatorname{softplus}(\ell_{ik})+10^{-8}]\,e_k(h).
\]

在精确数学中 H 与 `1−exp(−H)` 同序。诊断报告保留原 C，同时增加 H 风险 C、恰好0/1的风险数、唯一风险值数量，以及只因概率风险并列而丢失排序的可比对数。仍采用项目的严格时间可比对和精确风险相等规则；重算为float64可能暴露原来低精度舍入隐藏的细小差别。

`train_pathway.py` 的新报告加入 `evaluation_risk_numerics`，预测 CSV 追加 `cumulative_hazard_at_horizon`。原字段和值计算方式保留。未来可以比较二者判断饱和，但不能由这个反例宣布历史患者 C 被低估；旧记录曾报告某些既有模型风险没有饱和，需要真实检查点重新核验才能确定当前模型情况。

### 1.3 shuffled 对照会扰动后续 RNA 噪声

旧 `stability_weighted_common` 的 `torch.randperm` 使用全局 RNG，而 `rna_noise_view` 也消耗全局 RNG。因而相同 seed 下，开启 shuffled 除了重排权重，还会改变下一批 RNA 扰动。使用固定输入和同一初始 RNG，下一组随机数最大差为 2.0710。

已为该函数增加可选 `shuffle_generator`；完整通路训练的 `pathway_objective` 使用由当前 `view_seed` 派生、位于同一设备的独立 generator。完整路径的两批集成检查确认 stability 与 shuffled 的四组 RNA 视图逐元素相同，最终全局 RNG 状态也相同。有效权重排列仍发生变化，比较的目标因素得到保留。

`stream_config` 记录 `auxiliary_rng_scheme=rna_global__weight_shuffle_isolated_v1`。其他旧原型调用若不传 generator，仍使用原API行为；本轮修复覆盖当前完整 Pathway 训练入口，没有声称统一修复所有历史实验入口。过去的 stability 与 uniform 两臂都没有执行权重随机排列，因此这个问题不直接证明那两臂受影响。

旧版与新版 shuffled 训练轨迹不可混作同一随机协议。已有严格配对脚本会检查 stream_config；新增指标口径检查也会拒绝一边有新口径、一边缺失或不同口径的比较，不能手动删掉元数据以绕过。

## 2. 必须明确的指标约定

当前 Harrell 实现仅比较 `t_j > t_i && event_i=1`，风险精确相等算0.5。它排除所有同时间对；常见库还可将同时间事件—删失对作为可比对，并使用风险并列容差。参见 [scikit-survival 官方定义](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_censored.html)。因此不同代码的“C-index”未必完全同口径。此次保留现有算法，将约定记录进 `metric_conventions`。

IPCW 还需同时说明删失 KM 的 tied-time 风险集、事件使用 G(t) 还是 G(t−)，以及平均分母。取以下明确例子：

```text
train_time=[1,1,2], train_event=[1,0,1]
eval_time=[1,2], eval_event=[1,1], S(h)=[0.2,0.7], h=1.5
```

| 约定 | 事件权重分母 | 生存者权重分母 | Brier |
|---|---:|---:|---:|
| 项目 legacy_v1：同时间事件仍在删失风险集中；事件左极限 | 1 | 2/3 | 0.0875 |
| reverse KM，事件先移出风险集；事件左极限 | 1 | 1/2 | 0.11 |
| sksurv 0.28 的 reverse KM＋事件右连续值 | 1/2 | 1/2 | 0.13 |

项目值已实际运行核验，另外两值是依据固定版本源码作的手算，**本机没有运行 scikit-survival 得到它们**。依据：[reverse KM 文档](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.nonparametric.kaplan_meier_estimator.html)、[v0.28 nonparametric 源码](https://github.com/sebp/scikit-survival/blob/v0.28.0/sksurv/nonparametric.py)、[v0.28 metrics 源码](https://github.com/sebp/scikit-survival/blob/v0.28.0/sksurv/metrics.py)。这属于联合约定差异，未据此更换历史公式。

Brier 的分母保留全部评估样本 n，提前删失者贡献为0但仍包含在 n；新增检查核对添加一名提前删失样本后的变化。[官方 Brier 定义](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html)也使用全部样本平均。以后真实评估还需记录训练删失 KM 的支持范围和权重集中度；当前函数只检查所需 G 是否过小，不能把超出训练随访范围的平坦尾部自动当成有充分支持。

## 3. 单区间模型的表达限制

默认 `starts=[0]` 给每名患者一个恒定危险率。因此对任意正 h，H(h)、S(h)和发生概率排序都由同一个危险率决定；“用1−S(12)算出的Harrell C”不是12月截断或time-dependent C。h=0 时所有风险为0，排序没有区分能力。

用现有 `survival_at` 验证两区间已知答案：切点12，A的危险率 `[.08,.015]`，B为 `[.015,.08]`。

| 时间 | S_A | S_B |
|---|---:|---:|
| 12 | 0.382893 | 0.835270 |
| 24 | 0.319819 | 0.319819 |
| 36 | 0.267135 | 0.122456 |

单区间正危险率无法同时表达这种前后逆转。这里没有训练，更没有证明真实患者存在交叉曲线。改进候选是使用现有多区间实现，并按预先确定的临床时间点或各训练折独立拟合的规则确定切点；控制模型参数量，检查各区间事件与暴露量，再进行相同开发划分下的比较。

## 4. 还需要继续审查的部分

当前验证选轮代码采用严格 `val < best`，相同最小值保留最早轮次；patience按未改善轮数累计。它符合记录的新协议，不是SlotSPE的并列覆盖规则。每轮训练NLL是参数持续变化时各批损失的患者加权均值，验证NLL则来自轮末固定模型；二者有统计口径差异，不能把差值直接解释成精确泛化间隙。

下一项有价值的本地工作是逐项辅助梯度的范数和方向审计：用同一前向和同一随机视图分别计算生存与辅助梯度，检查二者的方向关系及梯度是否只停留在辅助投影。总梯度非零和标量损失小都不足以回答该问题。尚未在本轮把这个诊断实现为新训练策略。

## 5. 产物与证据

- `research_survival_audit.py` 与 `results/survival_protocol_audit_20261111.json`：已知答案、源代码SHA256、CPU环境和边界。
- `cancer_survival_diagnostics.py`：H风险及数值诊断，完整 Pathway 入口已接入。
- `test_survival_audit.py`：6项已知答案/契约检查；既有通路测试新增跨两批RNA视图一致性，并补新报告字段验证。
- `paired_pathway_validation.py`：新增指标口径一致性守卫。
- 针对性30项CPU检查通过；随后完整145项CPU合成/回归检查通过，0失败、0错误、0跳过，约98.9秒。回执见 `results/survival_protocol_validation_20261111.json`，包含本轮源码哈希。145包含此前已有检查，不能与138或30相加。没有新的患者指标、服务器状态或外测结论。
