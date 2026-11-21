# Supplementary design controls: evaluation rules

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](../../MIGRATION_SCOPE_ZH.md)。


This is a retrospective evaluation of frozen predictions, not a blind test.
The evaluator never imports or starts the runner, trains models, fits target data,
or changes prediction, checkpoint, fold-data, or prior energy-evaluation files.
Only the four same-batch variants are compared: full_same_run (Full),
phase_shared_strict, stationwise_matched, and concat_project.

## Population and readiness

The fixed matrix is AMS, JHB, LOA, MEL, SPO, SZH × seeds 13, 17, 29 × four
variants: 72 full-budget runs, or 18 city–seed folds per variant.
Read root/runs/target_<citylower>_seed<seed>/<variant>/result.json and
prediction_raw.npy. Fold truth and identifiers come from ../fold_data.npz,
including symlinks. --runs selects a different runs directory; --fold-data-root
may select an alternate directory containing the same per-fold subdirectories.
Prediction discovery only searches for prediction_raw.npy within an expected
variant directory. It never scans smoke, benchmark, checkpoint or NPZ prediction
archives to fill missing runs. Smoke status completed_smoke is never eligible,
even with --allow-partial or when --runs points to a smoke directory.

Require status completed, steps 2048, window_exposures 8192, optimizer_steps 410,
matching city/seed/variant, positive parameter_count, and initialization, training,
metrics and fold_protocol objects. Preserve the result metadata in the manifest.
Verify a supplied predictions_sha256 and fold_data_sha256. Runner metrics are
provenance only; all reported errors are recomputed from frozen predictions.
Optional diagnostics and configuration are copied best-effort, with read failures
reported rather than interpreted as evidence against a model.

Require native float32 truth and aligned [721,24,N] prediction, consecutive
label_starts 2928 through 3648 and unique ordered site_ids. Optional origin_time
and target_time must be datetime64, hourly, finite and mutually consistent.
Within a city, all seeds must identify exactly the same truth, sites and origins.
Within a fold, the common fold protocol and any supplied backbone initialization
hash must agree across models. Reject nonfinite input arrays and metrics.

## Metric definitions and precision

Use the exact pure origin_scores and metric_rows definitions in
../energy_value_20261104/run_energy_evaluation.py. Parse only those function ASTs;
never import or invoke that module's hardcoded evaluator. Record its SHA-256.
The self-test compares the streaming implementation to those exact definitions.

Convert each truth/prediction chunk to FP64 before any arithmetic or reduction.
Truth remains native float32 on disk. Memory-map prediction NPYs; stream the truth
NPY member out of NPZ into temporary disk storage and memory-map it. Do not load
all site arrays or all model predictions into RAM. Keep only per-origin scalar
sufficient statistics for paired resampling. No predictions are rounded.

For each origin with 24 leads and N stations:

- Station SSE = sum of squared prediction-minus-truth errors over leads/stations;
  SAE = sum of absolute errors; count = 24N. Fold station_rmse is
  sqrt(sum SSE / sum count); station_mae is sum SAE / sum count.
- City trajectories sum all stations separately at each lead, in FP64.
- Peak magnitude error is max(predicted city trajectory) minus max(truth city
  trajectory). peak_magnitude_mae averages its absolute value across origins.
- Ramp errors compare the 23 adjacent within-horizon differences. There is no
  transition between origins and no hour-24-to-hour-1 wrap. ramp_mae pools
  absolute ramp errors; ramp_rmse = sqrt(pooled ramp SSE / pooled ramp count).
- total_24h_mae averages abs(sum predicted city leads minus sum true city leads).
- peak_timing_mae_hours averages abs(first argmax(predicted city trajectory)
  minus first argmax(truth city trajectory)). Lead indices are zero-based, ties
  choose the first lead, and time distance is noncircular (0 vs 23 = 23 hours).
- Additional city/peak/total RMSE and city MAE use the same pooled definitions.
  Every RMSE is derived from SSE/count; a mean of per-origin RMSE is never used.

Raw is primary. nonnegative_clip is a separate operational sensitivity, applying
max(prediction,0) at the station/lead level before station aggregation. Truth is
never clipped. Do not mix raw and clipped rows or select the more favorable mode.

## Estimands and interpretation

The primary estimator is the equal mean of the 18 fold metrics per variant,
not a station-, exposure-, city-size- or pooled-origin-weighted mean. City
summaries equally average their three seed folds. Report paired control − Full
differences, so positive values mean Full has lower error. Pair by city and seed.

For each error metric, report separately:

- Difference: mean(control − Full).
- Ratio-of-means relative gain for Full: 100 × (mean(control) − mean(Full)) /
  mean(control), with the control as denominator.
- Mean-fold relative gain for Full: mean[100 × (control − Full) / control].

A zero denominator gives an undefined (null) relative quantity with an explicit
count; do not introduce epsilon denominators or silently drop zero folds to
claim the 18-fold mean relative gain. Paired fold tables retain each difference.

Intervals are exploratory pointwise 95% percentile sensitivity intervals.
There are only six city clusters; seeds and overlapping hourly windows are not
independent city replications. Shared training sources also limit independence.
No default p values, significance stars, familywise claims, causal interpretation,
or generalization to unseen cities or fresh retraining follows from intervals.

## Resampling

City-cluster bootstrap: 10,000 draws, RNG seed 20261117. Resample six cities with
replacement, retaining all three paired seed folds inside each selected city.
Use the same city draws for all models, metrics and postprocessing modes.
Bootstrap intervals describe absolute paired differences. The ratio-of-means
and mean-fold-relative quantities are reported separately as point estimates.

Temporal sensitivity: paired noncircular moving blocks of 168 consecutive origins,
10,000 draws by default, RNG seed 20261117 in an independent generator stream.
For each city/draw choose five starts uniformly from 0..553 inclusive, concatenate
their blocks, and truncate only the last block to retain 721 origins. Use the
same chosen origin indices across all three seeds, all four variants and both
postprocessing modes within that city; draw cities' temporal indices separately.
Keep the six cities and three seeds fixed. Resample scalar SSE/SAE/count and
recompute each fold metric (especially RMSE), then average the 18 fold values.
This diagnoses overlapping-window sensitivity, not training randomness. Record
draw counts and seed. A nondefault draw count is explicitly identified.

## Monitoring, exports and figures

Default output is root/analysis. All JSON is finite-checked and atomically
replaced with allow_nan=False. The manifest identifies its input snapshot,
expected/eligible/evaluated counts, missing/invalid runs, checksums, settings,
software, optional diagnostics and whether completion has actually been reached.
The output manifest is written last after successful exports; an initial running
manifest prevents stale complete status while a new snapshot is evaluated.

Without --allow-partial, an incomplete snapshot returns a nonzero exit status.
With --allow-partial it writes available metrics for monitoring and returns zero,
but never claims complete. Missing/invalid runs remain visible. Unbalanced point
estimates are labelled equal_available_fold_mean_monitoring. Intervals require
all 18 aligned paired folds for that particular comparison; incomplete pairs
receive no interval. Readiness can change while the runner writes new folds;
rerun the evaluator to capture a new snapshot. No waiting loop starts training.

Export per_fold.csv, per_origin.csv.gz, summaries.csv, paired_per_fold.csv,
paired_summary.csv and uncertainty.csv, plus small per-origin primitive caches.
CSV column headers are stable even when a monitoring snapshot has zero valid runs.
Metric names are station_rmse, station_mae, city_total_rmse, city_total_mae,
peak_magnitude_mae, peak_magnitude_rmse, ramp_mae, ramp_rmse, total_24h_mae,
total_24h_rmse and peak_timing_mae_hours. Per-fold rows also include all SSE, SAE
and count sufficient statistics. Per-origin rows include the reference's signed
errors, first-argmax indices, actual/predicted peaks and totals, and negative counts.
summaries.csv is wide (one row per city/variant/postprocess; city ALL means global),
with n_folds, expected_folds, n_cities, complete, estimator and the metric columns.
Paired summary and uncertainty tables are long (one metric per comparison row).
uncertainty.csv interval_method values are
six_city_cluster_bootstrap_all_three_seeds and
paired_168_origin_noncircular_moving_block_fixed_cities_seeds; join on control,
reference, postprocess and metric. There are 66 paired summary rows and 132
interval rows when complete (3 controls × 2 modes × 11 metrics × 2 schemes).
Main figure: four raw energy error panels showing paired control − Full effects,
city means across seeds, equal-fold means and city-cluster intervals when ready.
Tradeoff figure: station RMSE difference vs each energy error difference, with raw
and clipped symbols, connecting segments and the same city-cluster intervals.
Both are descriptive quantitative grids. Source data are the paired tables;
figure captions state available n and the pointwise interval definition. Export
editable PDF/SVG and PNG; zero reference lines and canonical labels show direction
without significance claims. Partial figures are labelled MONITORING.
Every figure passes a 1.5-point final rendered 2×2 plot-area alignment gate, saved
as an alignment JSON. The supplied synthetic verification figures also undergo
external PDF glyph-size and collision audits. These checks do not pre-approve
future figures rendered with new numerical ranges. PNG is a 300 dpi preview,
not a submission TIFF. Intervals use explicit line segments; synthetic generators
exist solely inside self_test and never supply production metrics.

## Execution and verification

    python evaluate_controls.py
    python evaluate_controls.py --allow-partial
    python evaluate_controls.py --runs /path/to/runs --out /path/to/analysis
    python evaluate_controls.py --self-test

--self-test uses synthetic data in a system temporary directory outside paper2,
does not train, and tests reference equivalence, clipping, timing ties/noncircular
distance, all 23 transitions, pooled RMSE, bounded discovery, budget/smoke rejection,
partial/strict status, deterministic paired resampling, and finite JSON behavior.
Synthetic outputs must never be mixed into paper tables or described as findings.
