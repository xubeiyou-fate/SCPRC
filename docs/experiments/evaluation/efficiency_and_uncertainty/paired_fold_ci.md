# Paired fold-level uncertainty: Original PhaseFormer vs SCPRC

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
