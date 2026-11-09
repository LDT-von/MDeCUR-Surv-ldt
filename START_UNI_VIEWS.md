# UNI 双视图部分已实现

入口是 `extract_uni_views.py`，独立源码在 `uni_views/`。不需要原始 DeCUR、Tri-Surv 或 SlotSPE 仓库才能运行提取工具。

交付包请解压到独立目录，先阅读本文件；不覆盖另一台电脑现有的 Tri-Surv 项目或数据目录。

这部分做的是：WSI＋已有 level-0 坐标 → 同一个完整 patch 的两种旋转/镜像 → 冻结的本地 UNI v1 → 两份逐行对应的特征。它不重新分割组织、不生成 patch 图片文件、不改患者划分。

1. 给目标电脑的 AI 阅读 [接入提示词](uni_views/HANDOFF.md)。
2. 从 [提取清单示例](uni_views/slides.csv.example) 建立真实切片清单。
3. 按 [完整说明](uni_views/README.md) 先运行 plan、doctor，再进行范围明确的试提取。
4. 使用新的癌症训练接口时，选择 `--path-views cached_uni`，它强制保留全部 patch。当前严格接口需要额外的未增强 reference，故提取时加 `--include-reference`；不会把顺序未经证实的旧缓存与新视图直接混用。

仅两份增强特征新增空间约为单份特征的2倍；加 reference 是3倍。按照报告的457张切片单份28.79GB，分别约57.58GB、86.37GB；实际436张交集由 plan 统计，不预先猜。

验证：35个不同的检查已通过，其中新增模块最终17项通过、既有模型与数据18项回归通过。新检查仅使用合成像素、伪 WSI、伪编码器及临时文件；没有读取患者像素、运行真实UNI或进行患者训练。验证细目见 `uni_views/VALIDATION.json`。

尚需目标电脑验证：OpenSlide原生库、实际SVS解码、UNI权重严格加载、GPU架构/显存、真实吞吐和存储。当前训练器仍是全批次padding，不能据此宣称数百万patch的生存训练已经具备扩展性。
