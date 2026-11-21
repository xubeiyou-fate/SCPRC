# Paired fold-level uncertainty: Original PhaseFormer vs SCPRC

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](../../../MIGRATION_SCOPE_ZH.md)。


Status: **completed**; 18 city×seed folds; 10,000 paired fold bootstrap replicates.

The primary estimator is the equal-weight mean of the 18 fold-level metrics. Each bootstrap draw resamples complete city×seed folds and uses the same indices for Original and Full. The intervals condition on the six observed cities, three fixed seeds, frozen checkpoints, and the retrospective forecast origins.

| Metric | Original | Full | Full−Original | 95% CI for difference | Relative change | 95% CI relative | Better folds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global RMSE | 35.067526 | 32.190739 | -2.876787 | [-4.787159, -1.287374] | -8.204% | [-11.234%, -5.674%] | 18/18 |
| Global MAE | 8.281722 | 7.049821 | -1.231901 | [-1.807680, -0.766768] | -14.875% | [-23.563%, -9.637%] | 18/18 |
| Macro-site RMSE | 11.067534 | 9.659218 | -1.408316 | [-2.042971, -0.896063] | -12.725% | [-19.574%, -8.847%] | 18/18 |
| Macro-site MAE | 8.281722 | 7.049821 | -1.231901 | [-1.792947, -0.765095] | -14.875% | [-23.446%, -9.646%] | 18/18 |

Negative differences favor SCPRC. These are fold-level uncertainty intervals, not guarantees over unseen cities, retraining runs, or future prospective data.

Input lineage was checked against [lineage_check.json](../../../../results/experiments/evaluation/efficiency_and_uncertainty/lineage_check.json); per-fold values are in [paired_fold_metrics.csv](../../../../results/experiments/evaluation/efficiency_and_uncertainty/paired_fold_metrics.csv).
