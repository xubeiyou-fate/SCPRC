# Model and Experimental Protocol

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](MIGRATION_SCOPE_ZH.md)。


Freeze ID: `SCPRC-PAPER-20261031-v2`. This note is the portable paper-writing description; exact implementation lineage remains recorded in each server-side run manifest.

## Algorithm Route

The offline candidate is a PhaseFormer forecasting backbone with a history-only, phase-conditioned station-relation adapter. PhaseFormer is the inherited forecaster; the contribution under evaluation is the constrained relation adaptation for a variable station set.

For each 168-hour input, RevIN-normalized history is arranged as seven daily cycles by 24 within-day phases. Shared profile/query/key layers form positive phase-wise station summaries using `ELU + 1` features and low-rank sufficient statistics. The implementation does not materialize a station-by-station adjacency matrix. A zero-initialized residual correction is centered across stations, preserving the city-level mean component. The correction is constrained using station scale, peer reliability, and source-derived disagreement signals.

SCPRC uses multi-source city training with PCGrad gradient projection, gradient clipping, and AdamW. PCGrad is an established training component, not a novelty claim. The Original PhaseFormer control uses the same backbone and training budget without SCPRC relation gates and without PCGrad. Exact-PCGrad keeps PCGrad but disables the relation gates; it isolates a training-mechanism step and is not the main paper baseline. Full-no-PCGrad keeps the complete relation adapter and disables PCGrad; it closes the 2×2 relation-on/off × PCGrad-on/off attribution design.

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

The Original and Full-no-PCGrad runs average source-city gradients without projection. The Full and Exact-PCGrad runs use the recorded PCGrad protocol. The method and baseline share the data, split, loss, configured minibatch budget, seeds and fold definition. R9 uses 10,000 paired city×seed bootstrap replicates and a descriptive difference-in-differences interaction. Exact implementation details, hashes and hardware live in the corresponding result manifests on the server.

## Metrics and Uncertainty

Compute raw-volume RMSE and MAE separately within each city-by-seed fold, then report the arithmetic mean of the 18 folds. Rolling origins are dependent; do not treat all hourly origins as IID replicates. The primary Original-vs-Full uncertainty table uses 10,000 paired bootstrap resamples of complete folds. It is conditional on the six observed cities, three fixed seeds, checkpoints and retrospective test periods.

U1 calibrates each fold's interval residuals on 361 target-history origins from 2023-07-16 through 2023-07-31, then reports empirical 80/90/95% coverage, width, interval score, WIS and capacity diagnostics at horizons 6/12/24. The U1 point metrics apply a nonnegative forecast truncation; keep those metrics inside U1 and do not replace the raw-prediction P0 headline. Empirical coverage is not a distribution-free guarantee.

U2 evaluates six fixed corruptions: 10% point missingness, 24-hour block missingness, 10% site outage, +20% scale shift, Gaussian noise at 10% source standard deviation, and 1% positive spikes at 3x. These are retrospective synthetic stress tests, not prospective reliability evidence.

H1 trains a separate 48-hour output head; it is not an inference-time extension of the h=24 checkpoint. E1 times model-forward inference in FP32 with CUDA synchronization, 12 warm-ups, 80 repeats and batch sizes 1/4/16. It does not measure an end-to-end serving SLA.

For the optional Online controller, use its own frozen 84-origin delayed-feedback protocol and moving-block uncertainty. Do not conflate that temporal-origin bootstrap with the primary fold bootstrap.

## External and Operational Protocol Boundaries

Report Jiaxing, Shanghai and UrbanEV separately. Jiaxing is a session-allocated hourly kWh proxy; Shanghai is hourly mean power in kW; UrbanEV is a rated-power-derived Shenzhen zone-volume series and is same-city cross-dataset evidence, not a seventh independent city. The external Exact/SCPRC rows are from the legacy v2.2 matrix and do not provide a current protocol-matched Original PhaseFormer comparison.

U1 and O1 are different capacity analyses: U1 compares Original with Full using calibrated intervals and a target-history protocol; the 2026-10-30 O1 replay uses Exact-PCGrad, Full, seasonal rules and an oracle under a fixed normalized capacity scenario. Neither uses real tariffs, transformer constraints, queues or grid dispatch data. Report normalized risk proxies only; do not claim monetary savings or grid reliability.

## Paper Claim Boundaries

- Lead with the Original PhaseFormer vs SCPRC h=24 comparison.
- Use Exact-PCGrad only to explain the PCGrad mechanism step; keep Original PhaseFormer as the main comparison baseline.
- Report Full-no-PCGrad as the relation-on/PCGrad-off factorial arm. Its current equal-fold mean is RMSE 36.0469 and MAE 8.2155, so the relation adapter alone is not uniformly beneficial without PCGrad.
- Report RMSE/MAE trade-offs in relation-operator results; Full does not win every metric.
- Preserve the Shanghai daily-mean advantage and seed-level external variation.
- Do not claim adaptive GWN/MTGNN/DCRNN or Energy dynamic-graph reproduction from audit-only local code. The active spatial row is fixed-distance Graph WaveNet and remains separate until its 18-fold run is complete.
- Describe every result as retrospective; none is blind or prospective confirmation.

For the full row-by-row matrix and numeric tables, see [MAIN_EXPERIMENT_MATRIX.md](MAIN_EXPERIMENT_MATRIX.md). The source algorithm description is [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

