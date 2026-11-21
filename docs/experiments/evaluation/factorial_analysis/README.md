# Formal paired comparison and 2x2 interaction

Status: **completed**; all four arms have 18 aligned city×seed folds and 10,000 paired bootstrap replicates.

The primary comparison remains Original PhaseFormer versus SCPRC. Exact-PCGrad is a mechanism control. Full-no-PCGrad closes the factorial design so the relation effect, PCGrad effect and their interaction are separately estimable.

## Paired comparisons

| Comparison | Metric | Left | Right | Right−Left | 95% CI | Relative change | Better folds |
|---|---|---:|---:|---:|---:|---:|---:|
| exact_vs_full | RMSE | 32.541714 | 32.190739 | -0.350976 | [-0.800073, -0.036832] | -1.079% | 13/18 |
| exact_vs_full | MAE | 7.022667 | 7.049821 | +0.027155 | [-0.047233, +0.109199] | +0.387% | 11/18 |
| original_vs_exact | RMSE | 35.067526 | 32.541714 | -2.525811 | [-4.298371, -1.054673] | -7.203% | 18/18 |
| original_vs_exact | MAE | 8.281722 | 7.022667 | -1.259056 | [-1.867552, -0.753867] | -15.203% | 17/18 |
| original_vs_full_no_pcgrad | RMSE | 35.067526 | 36.046891 | +0.979365 | [-1.407193, +4.039221] | +2.793% | 9/18 |
| original_vs_full_no_pcgrad | MAE | 8.281722 | 8.215484 | -0.066239 | [-0.461318, +0.346611] | -0.800% | 10/18 |
| full_no_pcgrad_vs_full | RMSE | 36.046891 | 32.190739 | -3.856152 | [-7.601869, -0.933310] | -10.698% | 18/18 |
| full_no_pcgrad_vs_full | MAE | 8.215484 | 7.049821 | -1.165662 | [-1.807726, -0.599987] | -14.189% | 16/18 |

## Factorial interaction

| Metric | Difference-in-differences | 95% CI |
|---|---:|---:|
| RMSE | -1.330341 | [-4.261121, +0.884267] |
| MAE | +0.093393 | [-0.301508, +0.480691] |

Negative differences favor the right-hand model in pairwise rows. The interaction is a descriptive factorial attribution under the locked protocol; it does not establish that PCGrad is a universally causal improvement.

Machine-readable results: [factorial_ci.json](../../../../results/experiments/evaluation/factorial_analysis/factorial_ci.json); fold-level values: [factorial_fold_metrics.csv](../../../../results/experiments/evaluation/factorial_analysis/factorial_fold_metrics.csv).
