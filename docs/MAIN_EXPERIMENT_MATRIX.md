# SCPRC Paper Experiment Matrix

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](MIGRATION_SCOPE_ZH.md)。


**2026-11-17 matrix update:** Three supplementary design controls are complete and integrated into the overview table and audited results below. The same-batch Full reference has its own run identity; historical canonical scores are not silently relabeled. Read the [results and conclusions](experiments/design_controls/RESULTS_AND_CONCLUSIONS_ZH.md) and [72-run independent audit](../results/experiments/design_controls/audit/completed_runs_audit.json).

Historical freeze ID: `SCPRC-PAPER-20261031-v2`  
Status authority: [experiments/rebuild_20261031/FINAL_SUBMISSION_AUDIT_V2.json](../results/experiments/evaluation/FINAL_SUBMISSION_AUDIT_V2.json)  
Previous freeze: [MAIN_EXPERIMENT_MATRIX_V1_20261031.md](SOURCE_ARTIFACT_INDEX.md#src-43b5751ca4dc823e)  
Scope: paper-facing claims and results. All evaluations are retrospective.

**2026-11-05 extension:** the main comparison and headline below remain fixed.
The completed [energy-value extension matrix](ENERGY_EXPERIMENT_MATRIX.md)
adds four strict controls (72 independent trainings), 16-model energy scoring,
gate/phase diagnostics and paired uncertainty. Read its
[results and bounded conclusions](experiments/energy_value/RESULTS_AND_CONCLUSIONS.md)
for current evidence. `Full-drop-*` controls are distinct from the old independent switches.

Historical status checked on 2026-11-01: R10 training finished at 2026-10-31 20:18:02 +08:00 (18/18 folds). Its post-run numerical and alignment checks pass, but paper acceptance remains under review. See [experiments/rebuild_20261031/r10_spatial_baselines/POSTRUN_AUDIT.json](../results/experiments/evaluation/spatial_baseline_review/POSTRUN_AUDIT.json). The v2 transfer ZIP remains a dated snapshot of its build time.

## Fixed Comparison Roles

The primary offline model comparison is **Original PhaseFormer (no-PCGrad) vs SCPRC**. Original PhaseFormer is the main baseline. SCPRC is the paper method and contains both the PCGrad training component and the phase-conditioned relation adapter. `Exact-PCGrad` is an attribution control that adds PCGrad while leaving SCPRC relation gates off; it is not the primary baseline. `Full-no-PCGrad` retains the relation adapter and disables PCGrad to close the factorial design. The historical `PhaseFormer Exact` row is a separate legacy run and must never be relabeled as Original PhaseFormer.

The candidate forecaster, its multi-source training procedure, and the optional `SCPRC-Online(84,.25)` release controller are separate objects. Report direct offline forecasts in the main results; report released forecasts in a separate deployment table.

## Canonical CHARGED Protocol

| Item | Frozen setting |
|---|---|
| Dataset | CHARGED hourly volume; six cities AMS, JHB, LOA, MEL, SPO, SZH |
| Split | Leave one city out; source April-July 2023, held-out city August 2023 |
| Replication | Seeds 13, 17, 29; 18 city x seed folds |
| Input / output | 168 hourly observations -> 24 hourly forecasts |
| Features / scale | Volume only; source-city scaler; native volume units |
| Budget | 2,048 source minibatches, batch size 4; 8,192 sampled source windows |
| Main aggregation | Arithmetic mean of the 18 fold-level metrics; equal fold weight |
| Status | Retrospective; August labels were inspected during development; no project-level blind test |

The main paired uncertainty analysis resamples the 18 complete city x seed folds. Its intervals are conditional on these six cities, three fixed seeds, frozen checkpoints and retrospective origins; they do not quantify unseen-city, future-period or retraining uncertainty.

## Full Experiment Matrix

| ID | Experiment | Current comparison / contents | Status | Paper placement |
|---|---|---|---|---|
| P0 | h=24 main accuracy | Original PhaseFormer vs SCPRC; PatchTST, TimeMixer and rule baselines | Complete, 18/18 folds | Main results |
| A1 | PCGrad and component ablation | Original -> Exact-PCGrad -> Scale-only / Peer-only / Disagreement-only -> Full | Complete, 18/18 folds | Ablation |
| A2 | Relation/operator controls | Daily relation, Phase relation, Full, parameter-matched Attention and DeepSets | Complete | Method attribution |
| U1 | Coverage, WIS and capacity | Original vs Full; same-fold target-history calibration; nominal 80/90/95%; horizons 6/12/24 h | Complete, 18/18; lineage match true | Uncertainty and decision diagnostics |
| U2 | Six-city input robustness | Original vs Full; six fixed perturbations, 18 folds per model | Complete, 216 combined model x scenario x fold rows | Robustness |
| H1 | Long horizon | Independently trained h=48 Original, Exact-PCGrad and Full | Complete, 18/18 folds | Horizon extension |
| E1 | Model efficiency | Original, Exact-PCGrad and Full; synchronized FP32 CUDA; batch 1/4/16 | Complete, 162 measurements | Efficiency |
| S1 | Paired uncertainty | Original vs Full; 10,000 paired fold-bootstrap replicates | Complete | Statistical reporting |
| X1 | External datasets | Jiaxing, Shanghai, UrbanEV; each kept separate | Existing matrix complete; legacy model lineage | External case studies |
| O1 | Capacity reservation replay | Exact-PCGrad, Full, seasonal rules and oracle in normalized native units | Completed scenario replay; separate from U1 | Operational scenario, bounded claims |
| D1 | Optional Online release | `SCPRC-Online(84,.25)` causal replay, separate from offline forecast scores | Existing retrospective replay | Optional deployment table |
| N1 | Adaptive GWN / MTGNN | No protocol-matched inductive implementation in the local matrix | Audit only | Do not report as completed baseline |
| N2 | Energy dynamic behavior graph | No verified official code, preprocessing and matched training protocol | Audit only | Do not claim faithful reproduction |
| N3 | Real tariff / dispatch benefit | No local tariff, capacity, queue or grid-constraint inputs | Blocked by required data | Keep normalized scenario; no monetary claim |
| R8 | Full-no-PCGrad 2×2 arm | Full relation structure with ordinary source-gradient averaging; PCGrad disabled; same h=24 protocol, 18 folds and 3 seeds | Complete, 18/18 | Strict factorial ablation |
| R9 | Formal paired Exact/Full and interaction | Exact-PCGrad vs canonical Full paired bootstrap plus Original/Exact/Full-no-PCGrad/Full difference-in-differences | Complete, 18 folds, 10,000 replicates | Formal attribution and uncertainty |
| R10 | Real spatial baseline | Volume-only fixed-distance Graph WaveNet using measured distance supports; adaptive node embeddings disabled | Training complete, 18/18; numerical/alignment checks pass; result review pending | Spatial sensitivity table after result acceptance |
| R11 | Energy behavior-graph protocol comparison | Local point diagnostic compared item by item with the named Energy dynamic behavior-graph residual protocol | Audit only; not faithful reproduction | Protocol limitation / related work |
| R12 | Energy operation diagnostics | Peak, peak timing, ramp, 24 h total, capacity reserve proxy, seasonal references and oracle lower bound | Complete, 18/18 | Energy-management connection with bounded claims |
| R13 | Strict temporal extrapolation | Audit of untouched future period and frozen future protocol | Complete audit; no blind test available | Limitation and future validation protocol |
| A3 / EV10 | Strict relation/gate removal | Full-drop-scale, Full-drop-peer, Full-drop-source-envelope, Daily+full attenuation; same PCGrad/data/budget | Training complete, 18/18 folds x 4 arms = 72 checkpoints | Strict mechanism ablation; energy trade-offs |
| EV0-EV6 | Unified energy metrics and attribution | 16 model identities, raw and nonnegative scoring; Peak/Ramp/24h Total/Timing; all 18 folds | Complete, 288 aligned model-fold identities, 576 metric rows | Energy value beyond PCGrad; comparison with simplified models |
| EV7 | Capacity parity and model extension | Original/Exact/Full match existing R12; retain seasonal 24/168/oracle, 16-model margin/penalty grid | Complete; parity maximum discrepancy below 1e-5 | Native-unit scenario trade-offs |
| EV8-EV9 | Activation, phase association and figures | Frozen Full gate diagnostics, input-only periodicity, all origins, paired city/fold figures | Complete; 7 main figures and 4 mechanism figures; QA passed | Mechanism conditions and boundaries |
| A4 / EV11 | Strict shared phase-relation operator | Same-batch Full vs `phase_shared_strict`; 18 paired folds | Complete; independent audit passed | Mechanism and energy attribution |
| A5 / EV12 | Parameter-matched stationwise control | Same-batch Full vs `stationwise_matched`; 18 paired folds | Complete; independent audit passed | Mechanism and energy attribution |
| A6 / EV13 | Concatenate-and-project fusion control | Same-batch Full vs `concat_project`; 18 paired folds | Complete; independent audit passed | Mechanism and energy attribution |

## P0: Main Accuracy and Baselines

Scores are equal-fold means; lower is better. PatchTST and TimeMixer use volume-only input and matched source-window exposure. Matching exposure does not make all optimizers or update counts identical.

| Model | RMSE | MAE | Role |
|---|---:|---:|---|
| **Original PhaseFormer (no-PCGrad)** | 35.0675 | 8.2817 | **Primary baseline** |
| **SCPRC** | **32.1907** | **7.0498** | **Proposed offline method** |
| PatchTST | 42.8268 | 8.1441 | Fair volume-only strong baseline |
| TimeMixer | 44.4249 | 8.3487 | Fair volume-only strong baseline |
| Seasonal naive, 24 h | 35.8339 | 7.5424 | Rule baseline |
| Seasonal naive, 168 h | 36.0085 | 8.2180 | Rule baseline |
| Persistence / last value | 78.6753 | 11.6569 | Rule baseline |
| Previous-24 h mean | 57.2497 | 9.9171 | Rule baseline |
| DLinear, two-variable | 70.9589 | 13.4285 | Qualified optimization match |
| N-BEATS + behavior-graph residual | 64.2645 | 12.4309 | Local diagnostic, not faithful reproduction |
| PhaseFormer Exact | 35.8956 | 8.4028 | Legacy Conventional Exact; provenance only |

SCPRC vs Original PhaseFormer: RMSE decreases 8.20%, MAE decreases 14.87% on the equal-fold mean estimand. The old 10.32% / 16.10% claim compares Full with legacy Conventional Exact and is not the current headline.

## A1-A2: Attribution Matrices

### PCGrad and independent components

| Variant | RMSE | MAE | Interpretation |
|---|---:|---:|---|
| Original PhaseFormer, no-PCGrad | 35.0675 | 8.2817 | Main baseline |
| Exact-PCGrad | 32.5417 | 7.0227 | PCGrad mechanism control; no relation gates |
| Scale-only | 32.1563 | 7.0385 | Independent scale component |
| Peer-only | 32.7614 | 7.1095 | Independent peer component |
| Source-disagreement-only | 32.6720 | 7.0840 | Independent disagreement component |
| SCPRC, independent-switch run | 32.1893 | 7.0497 | P2 component matrix; separate run identity |

The canonical four-arm run reports SCPRC as `32.1907 / 7.0498`. Keep that value separate from the independent-switch Full row; do not silently merge results from separate run identities. The interpretation chain is Original -> Exact-PCGrad for the PCGrad effect, then Exact-PCGrad -> Full for the added relation components, with individual switches used to inspect component behavior.

**Strict removal extension (2026-11-05):** A1's independent switches do not
establish Full-drop effects. The new A3/EV10 arms close this gap without
changing A1's archived values. Current energy evidence supports a phase/peak
benefit while showing trade-offs for Scale and Peer and a very small
source-envelope effect. Do not claim every mechanism is necessary or that
Full wins all energy metrics. Exact remains an attribution reference only.

### R8-R9 formal factorial completion

R8 completes the missing relation-on / PCGrad-off arm. Its equal-fold mean is RMSE `36.0469` and MAE `8.2155`. Relative to Original, RMSE is higher while MAE is slightly lower; therefore the relation adapter alone cannot be presented as uniformly beneficial without PCGrad.

R9 uses the same 18 aligned city×seed folds and 10,000 paired bootstrap replicates. Exact-PCGrad -> Full changes RMSE by `-0.3510`, 95% CI `[-0.8001, -0.0368]`; MAE changes by `+0.0272`, CI `[-0.0472, +0.1092]`. The PCGrad×relation difference-in-differences is `-1.3303` for RMSE, CI `[-4.2611, +0.8843]`, and `+0.0934` for MAE, CI `[-0.3015, +0.4807]`. These are descriptive factorial attribution results conditional on the six cities, three seeds and retrospective origins.

### Relation operator comparison

| Variant | RMSE | MAE | Role |
|---|---:|---:|---|
| Daily relation-PCGrad | 32.2066 | **6.9798** | Daily relation control |
| Phase relation-PCGrad | 32.6624 | 7.0384 | Phase relation control |
| **SCPRC** | **32.1907** | 7.0498 | Proposed relation method |
| Phase-wise generic Attention | 32.6658 | 7.0180 | Parameter-matched operator control |
| PhaseFormer + DeepSets adapter | 32.5458 | 7.0029 | Parameter-matched set control |

Full has lower RMSE than the tested operator controls, while they have lower MAE. Report a trade-off, not universal superiority.

## U1-U2: Uncertainty, Capacity and Robustness

U1 uses same-fold target-history calibration residuals (361 calibration origins per fold) and canonical Original/Full predictions. It reports empirical coverage at nominal 80%, 90% and 95%, interval width, interval score, WIS at horizons 6/12/24 h, and capacity risk. It applies nonnegative truncation before scoring point forecasts, so its Original/Full point metrics (`34.7548 / 7.7323` and `32.1760 / 7.0178`) differ from P0's raw-prediction headline. Keep the two estimands in their own tables. This is empirical retrospective coverage under the locked calibration protocol, not a distribution-free guarantee.

| U1 summary | Original PhaseFormer | SCPRC |
|---|---:|---:|
| 90% nominal coverage, h=24 | 93.32% | 93.14% |
| Mean interval width, h=24 | 29.91 | 26.80 |
| WIS, h=6 | 5.4361 | 5.0020 |
| WIS, h=12 | 5.5369 | 5.0881 |
| WIS, h=24 | 5.6833 | 5.2280 |
| 24 h point-capacity normalized cost | 7.6353 | 3.6150 |
| 24 h point-capacity overload-window rate | 87.00% | 72.01% |

Capacity costs are normalized scenario quantities in native volume units, not electricity bills or grid reliability measures. The U1 point summaries belong to this calibration/capacity analysis and must not replace the P0 main point-accuracy values.

U2 perturbations: 10% point missingness with mean imputation; 24 h block missingness; 10% site outage; +20% scale shift; Gaussian noise at 10% of source standard deviation; and positive spikes at 1% of cells scaled by 3. Full is not best on every perturbation: for positive spikes Original RMSE is 53.55 and Full is 53.77, while Full MAE is lower. Report scenario-specific values and avoid a blanket robustness claim.

## H1 and E1: Horizon and Efficiency

### h=48

These are independently trained 48-hour output heads, not h=24 checkpoints extended at inference. Do not compare h=48 numbers as if they replaced the h=24 main result.

| Model | RMSE | MAE |
|---|---:|---:|
| Original PhaseFormer | 34.9035 | 8.3396 |
| Exact-PCGrad | 33.1345 | 7.4613 |
| SCPRC | 32.9430 | 7.4448 |

### Inference efficiency

18 folds x 3 models x 3 batch sizes; FP32, CUDA-synchronized, 12 warm-ups and 80 timed repeats. Values below are means over fold-level batch p50 latency measurements.

| Model | Parameters | Batch 1 p50 (ms) | Batch 4 p50 (ms) | Batch 16 p50 (ms) |
|---|---:|---:|---:|---:|
| Original PhaseFormer | 36,360 | 2.786 | 5.282 | 15.534 |
| Exact-PCGrad | 36,360 | 2.579 | 5.015 | 16.729 |
| SCPRC | 37,263 | 5.320 | 7.791 | 20.196 |

This is model-forward timing on the recorded device, not an end-to-end serving SLA.

## X1: External Datasets and Baselines

The following are three-seed means, reported in each dataset's native target units. Never pool their raw RMSE/MAE. PhaseFormer Exact and SCPRC candidate are from the legacy 2026-10-24 external matrix; there is no current canonical Original PhaseFormer (no-PCGrad) external comparison in this bundle.

| Model / baseline | Jiaxing, kWh proxy | Shanghai, kW | UrbanEV, native volume |
|---|---:|---:|---:|
| DLinear | 26.0739 / 19.2442 | 288.5929 / 158.1101 | 208.4042 / 55.8891 |
| Daily-mean rule | 33.9683 / 26.0119 | **175.4674 / 82.8385** | 426.9106 / 106.7412 |
| NBEATS_GRAPH, local diagnostic | 26.9033 / 19.7756 | 593.7645 / 299.5258 | 255.3611 / 67.5440 |
| PatchTST | 27.2789 / 20.1866 | 317.6650 / 176.6518 | 218.0781 / 58.0634 |
| Persistence | 46.9571 / 33.7134 | 1328.5340 / 501.5145 | 591.3484 / 133.1669 |
| PhaseFormer Exact, legacy | 26.4107 / 19.2980 | 394.4765 / 196.5632 | 230.8918 / 60.2168 |
| SCPRC candidate v2.2, legacy | **25.0502 / 18.3736** | 333.0230 / 188.4495 | **202.1759 / 53.3197** |
| TimeMixer | 26.6191 / 19.6504 | 360.8646 / 182.0132 | 231.1178 / 62.0088 |
| Weekly-naive rule | 32.4004 / 22.7603 | 225.3904 / 105.0237 | 322.2413 / 77.1326 |

Jiaxing is session-derived hourly-energy proxy data; Shanghai is hourly mean power; UrbanEV is Shenzhen zone-level volume and is not an independent city from CHARGED-SZH. The Shanghai daily-mean result is a required negative/boundary result: it is substantially stronger than the listed learned models.

## O1 and Optional D1: Operational Boundaries

The 2026-10-30 O1 replay compares Exact-PCGrad, SCPRC, seasonal-naive baselines and an unavailable-in-practice oracle under a fixed normalized capacity rule. It is distinct from U1's newer Original-vs-Full calibrated capacity analysis. Keep the tables separate. No real tariff, transformer limit, queue, grid constraint or monetary saving was evaluated. The oracle uses realized peak and is diagnostic only.

`SCPRC-Online(84,.25)` is an optional causal release policy after offline forecasts. Its 27-case replay is a separate estimand; it is not another offline model row and must not replace P0 candidate accuracy. The archived 26-case replay is historical provenance only.

## R10-R13 new evidence and boundaries

R10 **fixed-distance Graph WaveNet** training completed for all 18 folds on 2026-10-31 at 20:18:02 +08:00. The 2026-11-01 post-run audit confirms artifact hashes, finite predictions, recomputed metrics, and exact station/origin/truth alignment with the canonical anchor. The primary-directory equal-fold result is RMSE 363.5875 and MAE 28.9003, with particularly high SZH error. These are provisional descriptive results. JHB/seed17 has a separate parallel-run result that differs despite matching recorded code, protocol and scaler; the exact cause remains unresolved. Both copies are retained without selecting by validation score. R10 remains outside the accepted main comparison pending this provenance and baseline-quality review. The local protocol records `adaptive_node_embedding=false` and `addaptadj=false`; it cannot be called adaptive Graph WaveNet. There is no completed protocol-matched MTGNN or DCRNN runner in this matrix.

R11 is an audit of the closest Energy dynamic behavior-graph residual paper. The local point diagnostic lacks the paper-matched quantile head, Pinball/WIS80 scoring, peak-risk task, graph prior, preprocessing and verified official protocol. It remains a qualified local diagnostic and cannot support a faithful superiority claim.

R12 connects forecasts to a transparent reserve decision: reserve the maximum predicted 24 h city-total volume with a fixed safety margin, then count overload and idle capacity. Full improves the city-total peak, timing, ramp and total-volume diagnostics over Original, but its normalized capacity proxy is `1.958` versus `3.339` for Original and `1.808` for Seasonal-24 h. This is native-volume scenario evidence, not tariff, dispatch, transformer, queue or grid-reliability evidence.

R13 confirms that the current project has no untouched future period for a blind evaluation. The paper must state that the results are retrospective and should use the frozen future protocol in [experiments/rebuild_20261031/r13_temporal_extrapolation/protocol.json](../results/experiments/evaluation/temporal_evaluation/protocol.json) if later data become available.

## Local Evidence Map

- Main CHARGED table and baselines: [figures_tables/table_strong_baselines.md](figures_tables/table_strong_baselines.md).
- Relation/operator table: [figures_tables/table_relation_operators.md](figures_tables/table_relation_operators.md).
- Machine-readable completion status: [experiments/rebuild_20261031/FINAL_SUBMISSION_AUDIT_V2.json](../results/experiments/evaluation/FINAL_SUBMISSION_AUDIT_V2.json).
- U1 summaries: [prediction_intervals/](../results/experiments/evaluation/prediction_intervals/).
- U2 summary: [experiments/rebuild_20261031/u2_original_robustness/original_vs_full_summary.csv](../results/experiments/evaluation/robustness/original_vs_full_summary.csv).
- h=48: [experiments/rebuild_20261031/h48_original/summary.csv](../results/experiments/evaluation/phaseformer_h48/summary.csv).
- Efficiency and paired uncertainty: [efficiency_and_uncertainty/](../results/experiments/evaluation/efficiency_and_uncertainty/).
- External results: [mandatory_supplement/external_strong_baselines_20261028/summary.md](mandatory_supplement/external_baselines/summary.md).
- Energy operation scenario: [mandatory_supplement/energy_scenario_20261030/README.md](mandatory_supplement/energy_scenario/README.md).
- Full-no-PCGrad arm: [experiments/rebuild_20261031/r8_full_no_pcgrad/summary.json](../results/experiments/evaluation/scprc_mean_gradient/summary.json).
- Formal factorial statistics: [experiments/rebuild_20261031/r9_factorial_stats/README.md](experiments/evaluation/factorial_analysis/README.md).
- Spatial baseline status: [experiments/rebuild_20261031/r10_spatial_baselines/RUN_STATUS.md](experiments/evaluation/spatial_baseline_review/RUN_STATUS.md).
- Energy diagnostics: [experiments/rebuild_20261031/r12_energy_diagnostics/README.md](experiments/evaluation/energy_diagnostics/README.md).
- Temporal extrapolation boundary: [experiments/rebuild_20261031/r13_temporal_extrapolation/README.md](SOURCE_ARTIFACT_INDEX.md#src-1adc1a507f775959).
- Energy literature audit: [literature_audit_20261030/energy_gap_literature_audit.md](SOURCE_ARTIFACT_INDEX.md#src-0d45e877894bc0d0).

<!-- BEGIN SUPPLEMENTARY_DESIGN_CONTROLS_20261117 -->
## 2026-11-17 Supplementary Design Controls (Living Extension)

This living-matrix extension preserves dated run identities and archived values. New evidence below updates the mechanism interpretation; older conclusions must be read with these controls. Original PhaseFormer (no-PCGrad) remains the primary baseline; Exact-PCGrad remains a mechanism/attribution control.

Completion is gated on an independent 72/72 artifact audit: four models × six cities × seeds 13/17/29, 18 folds each, 2,048 source minibatches / 8,192 window exposures / 410 canonical FP32 PCGrad updates. Truth, all origins 2928–3648, station order, checkpoint tensors, parameter counts, source/file hashes, shared backbone initialization and identical sampler traces pass; Full/strict-phase initial all-state hashes match.

`full_same_run` is the **same-batch Full reference (18 folds)**. Its new-run identity remains separate from SCPRC (main experiment); these rows do not replace canonical Full or change the primary comparison.

| ID | New control | Status | Paired reference |
|---|---|---|---|
| A4 / EV11 | `phase_shared_strict` — Strict shared phase-relation operator | Complete, audited 18/18 | `full_same_run`, 18/18 |
| A5 / EV12 | `stationwise_matched` — Parameter-matched stationwise control | Complete, audited 18/18 | `full_same_run`, 18/18 |
| A6 / EV13 | `concat_project` — Concatenate-and-project fusion control | Complete, audited 18/18 | `full_same_run`, 18/18 |

### Actual Equal-Fold Metrics

All values come from the validated analysis CSV, arithmetic mean over the same 18 folds. RMSE/MAE are site-level native-unit errors; Peak/Ramp/Total/Timing use city-summed trajectories. Raw and nonnegative-clipped forecasts are separate estimands; lower error is better.

| Model | Scoring | Parameters | RMSE | MAE | Peak MAE | Ramp MAE | 24 h Total MAE | Timing MAE (h) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `full_same_run` | raw | 37263 | 32.190601 | 7.049812 | 2567.883338 | 890.203559 | 23719.130024 | 5.251657 |
| `phase_shared_strict` | raw | 37263 | 32.671691 | 7.040180 | 2705.417028 | 914.719068 | 24439.207603 | 5.269610 |
| `stationwise_matched` | raw | 37267 | 32.620369 | 7.164150 | 2333.628479 | 892.711095 | 26538.215809 | 5.226460 |
| `concat_project` | raw | 37368 | 32.421639 | 7.118222 | 2396.180391 | 888.400802 | 26079.420477 | 5.256357 |
| `full_same_run` | nonnegative_clip | 37263 | 32.175896 | 7.017762 | 2542.526078 | 889.117654 | 22919.830738 | 5.281322 |
| `phase_shared_strict` | nonnegative_clip | 37263 | 32.657748 | 7.008986 | 2687.423896 | 913.949748 | 23846.923255 | 5.288103 |
| `stationwise_matched` | nonnegative_clip | 37267 | 32.608159 | 7.130228 | 2299.719761 | 891.101628 | 25431.441409 | 5.253275 |
| `concat_project` | nonnegative_clip | 37368 | 32.406877 | 7.085638 | 2370.352425 | 887.160953 | 25268.130899 | 5.290183 |

### Paired Energy Trade-offs and Uncertainty

The table retains every available paired interval for all three new controls, both scoring rules and all four energy endpoints, without selecting favorable outcomes. Differences and their directions are copied from the analysis and checked against the equal-fold means; interval methods and definitions are retained verbatim.

| Left model | Right model | Scoring | Metric | Difference | 95% interval | Method | Difference definition | Bootstrap replicates |
|---|---|---|---|---:|---|---|---|---:|
| `concat_project` | `full_same_run` | nonnegative_clip | mae | +0.067875 | [+0.048341, +0.080885] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | nonnegative_clip | mae | +0.067875 | [-0.011653, +0.179075] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | nonnegative_clip | peak_magnitude_mae | -172.173653 | [-184.994237, -167.568090] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | nonnegative_clip | peak_magnitude_mae | -172.173653 | [-496.755048, -2.231928] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | nonnegative_clip | peak_timing_mae_hours | +0.008861 | [-0.042919, +0.045693] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | nonnegative_clip | peak_timing_mae_hours | +0.008861 | [-0.009709, +0.023039] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | nonnegative_clip | ramp_mae | -1.956701 | [-3.146156, -1.501032] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | nonnegative_clip | ramp_mae | -1.956701 | [-8.245416, +2.628257] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | nonnegative_clip | rmse | +0.230981 | [+0.149901, +0.269319] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | nonnegative_clip | rmse | +0.230981 | [-0.000259, +0.500287] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | nonnegative_clip | total_24h_mae | +2348.300161 | [+1917.390675, +2608.730088] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | nonnegative_clip | total_24h_mae | +2348.300161 | [-438.552429, +7199.284314] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | mae | -0.008776 | [-0.046384, +0.013211] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | mae | -0.008776 | [-0.177953, +0.133005] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | peak_magnitude_mae | +144.897818 | [+141.453182, +156.621275] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | peak_magnitude_mae | +144.897818 | [+3.611253, +390.358046] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | peak_timing_mae_hours | +0.006781 | [-0.033520, +0.056482] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | peak_timing_mae_hours | +0.006781 | [-0.019417, +0.033981] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | ramp_mae | +24.832094 | [+23.433459, +26.133780] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | ramp_mae | +24.832094 | [-0.959645, +75.723993] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | rmse | +0.481853 | [+0.430375, +0.519865] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | rmse | +0.481853 | [+0.007374, +1.372644] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | total_24h_mae | +927.092517 | [+732.647367, +1051.904198] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | nonnegative_clip | total_24h_mae | +927.092517 | [-39.819584, +2159.613489] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | mae | +0.112466 | [+0.086544, +0.128516] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | mae | +0.112466 | [+0.025595, +0.199337] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | peak_magnitude_mae | -242.806316 | [-269.024270, -230.173145] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | peak_magnitude_mae | -242.806316 | [-625.701811, -14.741076] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | peak_timing_mae_hours | -0.028047 | [-0.115426, +0.063956] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | peak_timing_mae_hours | -0.028047 | [-0.088380, +0.049622] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | ramp_mae | +1.983974 | [-0.664709, +2.882200] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | ramp_mae | +1.983974 | [-0.109781, +4.132626] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | rmse | +0.432263 | [+0.363133, +0.504116] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | rmse | +0.432263 | [-0.007541, +1.027777] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | total_24h_mae | +2511.610671 | [+2010.258084, +2789.716110] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | nonnegative_clip | total_24h_mae | +2511.610671 | [-1734.725949, +9050.010089] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | mae | +0.068410 | [+0.048779, +0.081160] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | mae | +0.068410 | [-0.010626, +0.178931] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | peak_magnitude_mae | -171.702947 | [-184.311610, -166.961637] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | peak_magnitude_mae | -171.702947 | [-496.510994, -2.188332] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | peak_timing_mae_hours | +0.004700 | [-0.037065, +0.051472] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | peak_timing_mae_hours | +0.004700 | [-0.034366, +0.039683] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | ramp_mae | -1.802757 | [-3.027270, -1.350294] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | ramp_mae | -1.802757 | [-8.250144, +3.089855] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | rmse | +0.231038 | [+0.150000, +0.269271] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | rmse | +0.231038 | [-0.000032, +0.500306] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | total_24h_mae | +2360.290453 | [+1930.469555, +2622.697314] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `concat_project` | `full_same_run` | raw | total_24h_mae | +2360.290453 | [-414.420320, +7207.795003] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | mae | -0.009632 | [-0.046745, +0.012932] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | mae | -0.009632 | [-0.170130, +0.130073] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | peak_magnitude_mae | +137.533690 | [+133.023181, +149.598326] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | peak_magnitude_mae | +137.533690 | [+3.545916, +383.118756] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | peak_timing_mae_hours | +0.017953 | [-0.014178, +0.074588] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | peak_timing_mae_hours | +0.017953 | [-0.016723, +0.052242] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | ramp_mae | +24.515510 | [+23.188289, +25.968888] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | ramp_mae | +24.515510 | [-1.607076, +75.687817] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | rmse | +0.481090 | [+0.430318, +0.518891] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | rmse | +0.481090 | [+0.005325, +1.373865] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | total_24h_mae | +720.077580 | [+521.104483, +841.911921] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `phase_shared_strict` | `full_same_run` | raw | total_24h_mae | +720.077580 | [-40.024134, +1940.504751] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | mae | +0.114339 | [+0.089244, +0.130616] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | mae | +0.114339 | [+0.028324, +0.202566] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | peak_magnitude_mae | -234.254859 | [-263.618919, -217.487823] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | peak_magnitude_mae | -234.254859 | [-617.419722, -14.855517] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | peak_timing_mae_hours | -0.025196 | [-0.103177, +0.069658] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | peak_timing_mae_hours | -0.025196 | [-0.079211, +0.037294] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | ramp_mae | +2.507536 | [-1.276238, +3.522119] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | ramp_mae | +2.507536 | [+0.545751, +4.648274] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | rmse | +0.429768 | [+0.361603, +0.501785] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | rmse | +0.429768 | [-0.013425, +1.028678] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | total_24h_mae | +2819.085785 | [+2360.079995, +3119.638521] | paired_168_origin_noncircular_moving_block_fixed_cities_seeds | control minus Full; positive favors Full | 10000 |
| `stationwise_matched` | `full_same_run` | raw | total_24h_mae | +2819.085785 | [-772.863707, +9058.358878] | six_city_cluster_bootstrap_all_three_seeds | control minus Full; positive favors Full | 10000 |

Each interval uses 18 paired city×seed folds conditional on six cities, three fixed seeds, shared source data and retrospective August 2023 outcomes. Folds/origins are not 18 independent cities or independent retraining populations. Report the supplied city-cluster and fixed-model time-block sensitivity methods with their stated scopes. An interval spanning zero is inconclusive for direction under that method. These exploratory endpoint-wise analyses do not establish joint superiority, mechanism necessity, unseen-city generalization, future-period performance or retraining uncertainty. Native-volume energy errors do not establish tariff savings or dispatch/reliability benefits.

Analysis multiplicity statements: exploratory pointwise conditional sensitivity; no causal or familywise claim.

### Evidence Lineage

Audit timestamp: `2026-11-16T18:53:08.606117+00:00`.

- independent audit: [experiments/supplementary_design_controls_20261117/audit/completed_runs_audit.json](../results/experiments/design_controls/audit/completed_runs_audit.json); SHA256 `be3ca3393abc2fb13cbdca7473f3d530147450cf5bbefe3b7f914e7474d95489`.
- summary CSV: [experiments/supplementary_design_controls_20261117/analysis/summaries.csv](../results/experiments/design_controls/analysis/summaries.csv); SHA256 `aca56e0d38352f580ce0a614f1ef601d5b1df04d38850979fffe44d1de542330`.
- fold CSV: [experiments/supplementary_design_controls_20261117/analysis/per_fold.csv](../results/experiments/design_controls/analysis/per_fold.csv); SHA256 `94657fb01061957fa76ad9415acf3291a66f3979cf0e4ec9859998dae7193bf8`.
- paired CSV: [experiments/supplementary_design_controls_20261117/analysis/uncertainty.csv](../results/experiments/design_controls/analysis/uncertainty.csv); SHA256 `787e15762f84e64e201b2e93d94d9cd955e19fa56eed6a6e3d3335b8e9b8f631`.
- matrix integration script: [experiments/supplementary_design_controls_20261117/update_matrices.py](../experiments/design_controls/update_matrices.py); SHA256 `0df5797aff5ccf312dfaec14f71c847833d8c4ef86a7bc5d35ac6944b388a1b0`.
- analysis manifest: [experiments/supplementary_design_controls_20261117/analysis/manifest.json](../results/experiments/design_controls/analysis/manifest.json); SHA256 `83891ac8be219eaa6a4ba18390fb81ce88360de4481573367c735eebe1fa20d4`.

<!-- END SUPPLEMENTARY_DESIGN_CONTROLS_20261117 -->

## 2026-11-17 supplementary frozen-inference completion

All 72 design-control checkpoints passed independent audit; frozen FP32 inference completed 216/216 cases (batch 1/4/16), with zero prediction-parity violations. [Efficiency table and scope](experiments/design_controls/RESULTS_AND_CONCLUSIONS_ZH.md), [per-fold measurements](../results/experiments/design_controls/efficiency/per_fold_efficiency.csv), [completion status](../results/experiments/design_controls/COMPLETION_STATUS.csv). This is pure-forward measurement and is separate from the historical Original/Exact/Full efficiency experiment.
