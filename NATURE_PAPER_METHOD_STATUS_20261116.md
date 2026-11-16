# 自然论文（Nature Communications 2025，s41467-025-57541-y）与本项目的对应关系

> 用途：把"看了那篇论文之后做了什么"讲清楚，区分**已实现**、**已接入待跑**、**未实现**。
> 论文：Liu B, Polack M, Coudray N, et al. *Self-supervised learning reveals clinically
> relevant histomorphological patterns for therapeutic strategies in colon cancer.*
> Nature Communications 16:2328 (2025). DOI 10.1038/s41467-025-57541-y
>
> 本文件只记录**算法对应关系与实现状态**，不含任何患者分数。

---

## 1. 论文的方法是什么

论文的完整链条是五步：

| 步骤 | 论文做法 | 关键参数 |
|---|---|---|
| ① 自监督编码 | Barlow Twins 训练 patch 编码器 | 250,000 tiles，128 维，10x，224×224 |
| ② 聚类 | 在 tile 向量的 KNN 图上跑 **Leiden community detection** | K=250 近邻，γ=1.5，最终 **47 个 HPC** |
| ③ 质控 | 去掉含气泡/异物/失焦的簇（125 → 剔除 12 → 重跑） | 病理学家双人评估 |
| ④ 组成表示 | 每张 WSI = 各 HPC 的 **tile 占比**（组成向量），再做 **CLR** 变换 | 47 维，自由度 K−1 |
| ⑤ 生存建模 | 对 CLR 组成做 **L2 正则 Cox 回归**，5 折 CV 选 γ 与 α | α=0.1842，外部 AVANT 对照组 c-index **0.65**（临床基线 0.58） |

关键点：**论文的生存模型只用病理组成，不用 RNA**。它是一个"病理图谱 → 预后"的
可解释路径，而不是多模态融合方法。

---

## 2. 本项目对应实现了什么

| 论文步骤 | 本项目 | 状态 |
|---|---|---|
| ① 自监督编码 | **不复现**。本项目用的是已冻结的 **UNI v1** patch 特征，不是 Barlow Twins | **未实现**（且本机无 tile 图像） |
| ② Leiden 聚类 | **不复现**。用 **标准化 + PCA + 确定性 k-means** 得到 K=4 个特征空间原型 | **已实现（替代方案）**，`FrozenMorphologyBank` |
| ③ 质控 | 用 mass cutoff 标记"患者—簇表示不可估计"，不硬投影 | **已实现** |
| ④ 组成 + CLR | **已实现**，含乘法零替换与闭合性质检验 | `composition_clr` |
| ⑤ Cox on CLR | **不复现 Cox**。用**离散 hazard 生存头** + **RNA 条件化的预测残差** | **已实现（改造版）** |

### 本项目真正的差异化（论文里没有的）

论文是**病理单模态**的组成 → Cox。本项目在组成之外加了一层：

- **c = f̂(RNA)**：用 RNA 预测冻结病理池化 p，得到"RNA 能解释的病理部分"。
- **u = p − c**：病理的 RNA 预测残差。
- 生存头输入 `[RNA, c, CLR, u, u×RNA]`，其中 u×RNA 是显式的交互项。

RBF 核岭 + 患者级交叉拟合探针是 `morphid_probe.py` 的实现。

---

## 3. 与论文的差异必须如实说明

1. **不是 HPL/Leiden 复现**。论文的 47 个 HPC 来自结直肠癌的 Barlow Twins + Leiden
   (γ=1.5)，KNN 传播到外部队列。本项目用 K=4 的 k-means 原型，
   **簇数、簇语义、聚类算法都不同**，不声称复现论文 HPC，也不声称簇可被病理学家辨认。
2. **编码器不同**。论文自训 Barlow Twins；本项目用冻结 UNI。两者特征空间不可比。
3. **生存模型不同**。论文是纯 Cox on CLR；本项目是离散 hazard 头 + 残差交互。
4. **CLR 自由度 K−1**。系数不是独立的绝对组织效应；K=4 时只有 3 个自由度。
5. **低探针 R² 不是独立性证据**。它是"这个探针族解释不了多少"的陈述。

---

## 4. 那篇论文的旧草稿去哪了

早期 `我的新尝试/cluster_conditional_decur.py` 是按该论文思路（HPL/Leiden + CLR + Cox）
写的草稿。**2026-11-16 目录精简时它被删除**，且**没有进入任何 Git 提交**
（`Conditional_DeCUR_Surv` 的全部提交历史里都没有 `cluster_conditional_decur` 或
`我的新尝试` 路径）。工作区内已不存在该目录，本机也**没有**找到它的工作区外备份 ZIP
（`Downloads` 里只有 `stability_weighted_decur_architecture_bundle.zip`，是另一件东西）。

所以：**那个草稿无法直接拿来跑**。当前可运行的是 MorphID 路线。

---

## 5. 要不要跑，跑什么

### 已经在这一批里（不需要额外决定）

`morphid` 与 `morphid_no_interaction` 两条臂已接入 `comparison_fold0.py`，
会随 BLCA fold0 的六臂批次一起跑。这两条臂承载的就是"论文思路 + RNA 残差改造"。

### 想真正对齐论文，需要额外做（目前**未实现**）

一个**纯病理组成 → L2 正则 Cox** 的对照臂。它不需要深度训练，很便宜：

1. 用训练折的冻结 patch 特征做 Leiden 聚类（或先用 k-means 作为无依赖替代）
   → 每患者的簇占比组成 → CLR。
2. 在训练折上拟合 L2 正则 Cox（风险集按**时间**定义，不复用上游 `R_set` 的按行号下三角）。
3. 在验证折算 Harrell C / IPCW C。

它**没有** RNA 融合、**没有** 离散 hazard 头，因此**不能**直接进现有 `summary` 表
（那一列是逐患者离散 NLL 的口径），需要单独的评估路径和单独的报告。

依赖现实：本机 `leidenalg` / `igraph` **都没装**，只有 `networkx` / `sklearn` / `scipy`。
服务器上是否装了需要先查；没装则先走 k-means 替代，或在服务器侧安装
（这需要你授权改环境）。

---

## 6. 结论

- 论文的**思想**（离散形态簇 → 组成 → CLR → 生存建模）已经进入本项目，形态部分是 MorphID。
- 论文的**具体实现**（Barlow Twins、Leiden、47 HPC、纯 Cox）**没有**进入本项目，
  也**不能**声称复现。
- MorphID 已有本地合成证据（25 项）与单折接入（8 项契约测试），
  **没有**患者训练、**没有** CUDA、**没有**患者分数。
- 是否要补一个"纯组成 Cox"臂，取决于你是否需要一个"和论文同形"的锚点；
  它很便宜，但需要新的评估路径，且 Leiden 需要额外的环境依赖。
