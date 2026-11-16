# 当前癌症生存模型

当前完整模型、七个模型源码文件及四种对照的路径统一见 [README.md](README.md)。

默认主模型为 `DensePathwaySurvival`，预测使用病理＋RNA，训练保留11旧＋4dense辅助。`cancer_model.py` 和 `cancer_pathway.py` 是上层实际复用的基础组件。

RNA-only、WSI-only 和早期 masked-mean 原型已删除，不再作为可运行方法。共享辅助损失现在位于 `decur_losses.py`。

单折计划与服务器命令见 [PATIENTWISE_V3_HANDOFF_20261116.md](PATIENTWISE_V3_HANDOFF_20261116.md)。源码清单变更后必须重新生成plan；历史验证峰值不是独立测试结果。
