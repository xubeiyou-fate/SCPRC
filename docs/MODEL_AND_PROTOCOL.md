# Model and Experimental Protocol

Protocol ID: `SCPRC-PAPER-20261031-v2`. This document describes the model, data split, training budget, and evaluation protocol. Per-run configuration and source records are listed with the experiment results.

## Algorithm Route

SCPRC uses a PhaseFormer forecasting backbone with a history-only, phase-conditioned station-relation adapter. PhaseFormer is the inherited forecaster; the contribution under evaluation is the constrained relation adaptation for a variable station set.

For each 168-hour input, RevIN-normalized history is arranged as seven daily cycles by 24 within-day phases. Shared profile/query/key layers form positive phase-wise station summaries using `ELU + 1` features and low-rank sufficient statistics. The implementation does not materialize a station-by-station adjacency matrix. A zero-initialized residual correction is centered across stations, preserving the city-level mean component. The correction is constrained using station scale, peer reliability, and source-derived disagreement signals.

SCPRC uses multi-source city training with PCGrad gradient projection, gradient clipping, and AdamW. PCGrad provides the gradient-projection step in multi-source training. The Original PhaseFormer control uses the same backbone and training budget without SCPRC relation gates and without PCGrad. Exact-PCGrad keeps PCGrad but disables the relation gates; it isolates a training-mechanism step and is not the main paper baseline. Full-no-PCGrad keeps the complete relation adapter and disables PCGrad; it closes the 2×2 relation-on/off × PCGrad-on/off attribution design.

The optional source-support screen, Exact fallback, and `SCPRC-Online(84,.25)` run after offline forecasts exist. They change which prediction is released; they do not define the offline candidate or replace its point-accuracy score.

## CHARGED Training and Evaluation

| Item | Protocol |
|---|---|
| Data | CHARGED standard hourly archive; volume only, kWh per site per local hour |
| Cities | AMS, JHB, LOA, MEL, SPO, SZH |
| Evaluation | Leave one city out; five source cities and one held-out city per fold |
| Time split | Source April-July 2023; held-out evaluation August 2023 |
| Input / horizon | 168 hours / 24 hours for the main matrix; separate real h=48 head for H1 |
| Replication | Seeds 13, 17, 29; 18 city-by-seed folds |
| Scaling | Source-city-only scaler; do not fit on held-out test labels |
| Training budget | 2,048 source minibatches per fold, batch size 4, 8,192 sampled windows |
| Objective | Standardized MSE plus 0.15 multi-resolution consistency at 2, 3, 4, 6, 12 |
| Optimizer | AdamW; PCGrad variants aggregate source-city gradients, about 410 optimizer updates |
| Test origins | 721 rolling forecast origins per canonical h=24 fold |

The Original and Full-no-PCGrad runs average source-city gradients without projection. The Full and Exact-PCGrad runs use the recorded PCGrad protocol. The method and baseline share the data, split, loss, configured minibatch budget, seeds and fold definition. R9 uses 10,000 paired city×seed bootstrap replicates and a descriptive difference-in-differences interaction. Implementation settings, hashes, and hardware are recorded in the corresponding run manifests; availability is listed in [release contents](MIGRATION_SCOPE_ZH.md).

## Metrics and Uncertainty

Compute raw-volume RMSE and MAE separately within each city-by-seed fold, then report the arithmetic mean of the 18 folds. Rolling origins are dependent; do not treat all hourly origins as IID replicates. The primary Original-vs-Full uncertainty table uses 10,000 paired bootstrap resamples of complete folds. It is conditional on the six observed cities, three fixed seeds, checkpoints and retrospective test periods.

U1 calibrates each fold's interval residuals on 361 target-history origins from 2023-07-16 through 2023-07-31, then reports empirical 80/90/95% coverage, width, interval score, WIS and capacity diagnostics at horizons 6/12/24. The U1 point metrics apply a nonnegative forecast truncation; keep those metrics inside U1 and do not replace the raw-prediction P0 headline. Empirical coverage is not a distribution-free guarantee.

U2 evaluates six fixed corruptions: 10% point missingness, 24-hour block missingness, 10% site outage, +20% scale shift, Gaussian noise at 10% source standard deviation, and 1% positive spikes at 3x. These are retrospective synthetic stress tests, not prospective reliability evidence.

H1 trains a separate 48-hour output head; it is not an inference-time extension of the h=24 checkpoint. E1 times model-forward inference in FP32 with CUDA synchronization, 12 warm-ups, 80 repeats and batch sizes 1/4/16. It does not measure an end-to-end serving SLA.

For the optional Online controller, use its own frozen 84-origin delayed-feedback protocol and moving-block uncertainty. Do not conflate that temporal-origin bootstrap with the primary fold bootstrap.

## External and Operational Protocol Boundaries

Report Jiaxing, Shanghai and UrbanEV separately. Jiaxing is a session-allocated hourly kWh proxy; Shanghai is hourly mean power in kW; UrbanEV is a rated-power-derived Shenzhen zone-volume series and is same-city cross-dataset evidence, not a seventh independent city. The external Exact/SCPRC rows are from the legacy v2.2 matrix and do not provide a current protocol-matched Original PhaseFormer comparison.

U1 and O1 are different capacity analyses: U1 compares Original with Full using calibrated intervals and a target-history protocol; the 2026-10-30 O1 replay uses Exact-PCGrad, Full, seasonal rules and an oracle under a fixed normalized capacity scenario. Neither uses real tariffs, transformer constraints, queues or grid dispatch data. Report normalized risk proxies only; do not claim monetary savings or grid reliability.

## Interpretation

- The primary comparison is Original PhaseFormer versus SCPRC at h=24.
- Exact-PCGrad is a mechanism control; Original PhaseFormer is the main baseline.
- Full-no-PCGrad is the relation-on/PCGrad-off factorial arm. Its current equal-fold mean is RMSE 36.0469 and MAE 8.2155, so the relation adapter alone is not uniformly beneficial without PCGrad.
- Relation operators exhibit RMSE/MAE trade-offs; SCPRC does not have the lowest error on every metric.
- The Shanghai daily-mean baseline has lower errors than the reported learned models, and external results vary by seed.
- The spatial diagnostic is fixed-distance Graph WaveNet. Its 18-fold run is complete, but provenance and quality questions remain unresolved. No accepted adaptive GWN, MTGNN, DCRNN, or faithful Energy dynamic-graph reproduction is reported.
- The reported evaluations are retrospective; they are not blind or prospective confirmation.

For the full row-by-row matrix and numeric tables, see [MAIN_EXPERIMENT_MATRIX.md](MAIN_EXPERIMENT_MATRIX.md). The source algorithm description is [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

