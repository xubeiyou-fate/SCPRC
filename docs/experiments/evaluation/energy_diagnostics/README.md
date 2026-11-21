# R12 Energy-operation diagnostics

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](../../../MIGRATION_SCOPE_ZH.md)。


Status: **completed**. This directory contains a read-only post-processing replay of the canonical Original PhaseFormer and SCPRC forecasts. It does not retrain a model and does not modify any checkpoint, dataset, or paper matrix.

## Scope and analysis unit

The evaluation contains six cities (`AMS`, `JHB`, `LOA`, `MEL`, `SPO`, `SZH`), three seeds (`13`, `17`, `29`), 18 `city×seed` folds, 721 rolling origins per fold, and a 24-hour forecast horizon. The independent comparison unit is the fold. Origins, hours, and stations are repeated observations nested inside each fold; they are not counted as independent replicates.

The input lineage is:

- Original PhaseFormer: [paper2/experiments/rebuild_20261030/p2_no_pcgrad_phaseformer/target_*/*/predictions.npz](../../../SOURCE_ARTIFACT_INDEX.md#src-101368b7b3846fb6), key `pred_phaseformer_no_pcgrad_raw`.
- SCPRC: [paper2/experiments/rebuild_20261030/canonical_h24_anchor/target_*/*/predictions.npz](../../../SOURCE_ARTIFACT_INDEX.md#src-3e41da61b4b1ce83), key `pred_scpcr_candidate_raw`.
- Labels and station identifiers are checked against the local CHARGED v3 [Hourly.zip](../../../SOURCE_ARTIFACT_INDEX.md#src-3bb4240477855cd5) archive.

All model forecasts are clipped at zero, matching the existing U1/O1 replay convention. Seasonal naive rules are causal: the 24-hour rule copies the preceding daily cycle and the 168-hour rule copies the preceding weekly cycle. No August label is used to construct those baselines.

## Diagnostics

For both the city-total layer and the station layer, the script reports:

- peak level error: the 24-hour-window maximum prediction minus the realized maximum;
- peak timing error: argmax-hour difference within the 24-hour window;
- ramp MAE: absolute error of the first hourly difference;
- ramp-threshold exceedance: actual and predicted exceedance rates, rate gap, and hit rate;
- 24-hour total-volume error: absolute error, signed bias, positive-actual MAPE, and WAPE.

The city-total layer sums all station series before computing the peak, timing, ramp, and total-volume diagnostics. The station layer retains every station as a separate series and averages the station-level observations inside each fold. Because some station-hour actual values are zero, relative errors are reported as `*_mape_pct_positive_actual` with a valid-positive denominator rate, together with WAPE. Absolute errors remain defined on all observations.

Ramp thresholds are fixed from the pre-August observed prefix only: the city-total threshold is the 90th percentile of absolute hourly city-total ramps, and the station threshold is calculated separately for each station. These are stress thresholds for diagnostics, not physical operating limits and not test-set-tuned parameters.

## Capacity-reserve replay

The primary scenario preserves the existing normalized rule:

```text
reserved capacity = max over 24 predicted city-total volumes × (1 + safety margin)
overload volume    = positive(actual hourly city-total volume − reserved capacity)
proxy cost         = reserved capacity + λ × overload volume
normalized proxy   = proxy cost / realized 24-hour city-total peak
```

The primary setting is a 10% a-priori safety margin and `λ=10`. Sensitivity files cover margins `0%, 5%, 10%, 20%` and penalties `λ=1, 5, 10, 20`. The replay compares Original PhaseFormer, SCPRC, Seasonal naive 24 h, Seasonal naive 168 h, and an `Oracle actual peak` lower-bound diagnostic.

The oracle reserves the realized 24-hour peak. It is unavailable before the outcome is observed and is never a deployable policy or a tuning target. The capacity values remain in the CHARGED dataset's native aggregate-volume units. The normalized proxy is dimensionless; it is not electricity cost, tariff savings, transformer loading, queueing performance, dispatch feasibility, grid reliability, or revenue.

The older U1 table reports point capacity at multiple horizons without the primary 10% margin. R12's primary capacity file uses the fixed 10% margin and therefore must not be numerically compared to U1's zero-margin point rows without matching `safety_margin_pct` and horizon definitions.

## Main output files

- [operational_metrics_per_fold.csv](../../../../results/experiments/evaluation/energy_diagnostics/operational_metrics_per_fold.csv): 18-fold summaries for the two analysis layers and four models.
- [operational_metrics_equal_fold.csv](../../../../results/experiments/evaluation/energy_diagnostics/operational_metrics_equal_fold.csv): equal-fold means for manuscript reporting.
- [operational_metrics_per_origin_city_total.csv](../../../SOURCE_ARTIFACT_INDEX.md#src-d5ae7bc8cfe6d436): city-total metrics for every fold and origin.
- [ramp_thresholds_by_city.csv](../../../SOURCE_ARTIFACT_INDEX.md#src-22129871bee46e33): pre-August city and station threshold metadata.
- [capacity_reserve_risk_per_fold.csv](../../../SOURCE_ARTIFACT_INDEX.md#src-89d9eef2faf47549): fold-level capacity replay for all margins and penalties.
- [capacity_reserve_risk_equal_fold.csv](../../../../results/experiments/evaluation/energy_diagnostics/capacity_reserve_risk_equal_fold.csv): equal-fold sensitivity summaries.
- [capacity_reserve_risk_per_origin.csv](../../../SOURCE_ARTIFACT_INDEX.md#src-a2b4f88dc6b775a6): origin-level capacity, overload, idle-capacity, and normalized proxy values.
- [capacity_reserve_risk_primary_equal_fold.csv](../../../../results/experiments/evaluation/energy_diagnostics/capacity_reserve_risk_primary_equal_fold.csv): primary 10%/`λ=10` comparison.
- [run_r12_energy_diagnostics.py](../../../../experiments/evaluation/energy_diagnostics/run_r12_energy_diagnostics.py): deterministic recomputation script.
- [protocol.json](../../../../results/experiments/evaluation/energy_diagnostics/protocol.json): full protocol, lineage, parameter grid, output list, and claim limits.
- [manifest.json](../../../../results/experiments/evaluation/energy_diagnostics/manifest.json): compact artifact manifest with hashes and quality checks.
- [STATUS.json](../../../SOURCE_ARTIFACT_INDEX.md#src-dd8f6dfe2745a3c1): completion and validation status.
- [summary.json](../../../../results/experiments/evaluation/energy_diagnostics/summary.json): compact numerical summary.

## Recompute and quality boundary

From the SCPRC release repository, the scoring entry point is:

```bash
python experiments/evaluation/energy_diagnostics/run_r12_energy_diagnostics.py
```

This scoring entry point requires its separately indexed frozen inputs; those arrays are not included in the code release. Configure external input and output locations as documented by the entry point before execution. A completed run is expected to contain 18 folds, 144 operational fold rows (4 models × 2 layers × 18 folds), 51,912 city-total origin rows, 1,440 capacity fold rows, and 1,038,240 capacity origin rows. The checked artifacts contain no numeric NaN values.

These results support a retrospective operational-diagnostic statement: forecasts can be translated into a transparent capacity-reserve scenario, and the model/baseline trade-off can be inspected by city, seed, horizon, margin, and penalty. They do not establish a real energy-management benefit without tariff data, physical capacity constraints, queue/dispatch dynamics, or a prospective blind deployment evaluation.
