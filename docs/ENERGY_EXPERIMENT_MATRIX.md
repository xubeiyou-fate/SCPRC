# SCPRC Energy-Value Extension Matrix

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](MIGRATION_SCOPE_ZH.md)。


**2026-11-17 matrix update:** Three supplementary design controls are complete and integrated into the overview table and audited results below. The same-batch Full reference has its own run identity; historical canonical scores are not silently relabeled. Read the [results and conclusions](experiments/design_controls/RESULTS_AND_CONCLUSIONS_ZH.md) and [72-run independent audit](../results/experiments/design_controls/audit/completed_runs_audit.json).

Freeze ID: `SCPRC-ENERGY-VALUE-20261104`
Parent matrix: [MAIN_EXPERIMENT_MATRIX.md](MAIN_EXPERIMENT_MATRIX.md) (`SCPRC-PAPER-20261031-v2`)
Scope: post-hoc energy-facing validation of the frozen six-city CHARGED matrix.

Completed extension: 2026-11-05. Four strict arms were independently trained
for all 18 folds (72 trainings). The full analysis includes 16 model identities,
288 aligned model-fold inputs, and 576 raw/clipped fold metric rows.
Read [experiments/energy_value_20261104/RESULTS_AND_CONCLUSIONS.md](experiments/energy_value/RESULTS_AND_CONCLUSIONS.md).

This file is an extension of the paper matrix. It does not replace the main
accuracy matrix. The primary paper comparison remains Original PhaseFormer
(no-PCGrad) versus SCPRC. Exact-PCGrad is a mechanism control, and the
independent-switch component runs are descriptive controls rather than strict
Full-drop ablations.

## Frozen protocol

| Item | Setting |
|---|---|
| Cities | AMS, JHB, LOA, MEL, SPO, SZH |
| Seeds | 13, 17, 29; 18 city x seed folds |
| History / horizon | 168 h -> 24 h |
| Evaluation | Held-out August 2023 target city, retrospective |
| Primary estimand | Equal mean of fold-level metrics |
| Main pair | Original PhaseFormer vs SCPRC |
| Mechanism pair | Exact-PCGrad vs SCPRC |
| Energy units | Original native volume units; no kW or currency conversion |
| Prediction rule | Preserve raw predictions for the primary metric; any nonnegative clipping is a separately named sensitivity analysis |
| Pairing | Same fold, origin, horizon and station order; incompatible lineage is excluded from paired claims |

## Energy-value matrix

| ID | Question | Models / data | Status | Evidence required | Paper role |
|---|---|---|---|---|---|
| EV0 | Are predictions and truths strictly aligned? | All eligible 18-fold archives | Complete, 288/288 | Hashes, shapes, labels, site IDs, lineage table | Methods audit |
| EV1 | Does the proposed method improve energy-facing forecasts over the main baseline? | Original, Full | Complete | Peak, ramp, total, timing, high-load and site-weighted metrics | Main energy table |
| EV2 | Does the relation adapter add value beyond PCGrad? | Exact-PCGrad, Full | Complete; peak benefit and total trade-off | Paired fold differences; fold/city/time-block sensitivity intervals | Core attribution table |
| EV3 | Does phase conditioning matter? | Daily relation, Phase relation, Full, Daily+full attenuation | Complete | Same energy metrics and fold pairing | Structural ablation |
| EV4 | Which gates are active and where do they help? | Scale-only, Peer-only, Source-disagreement-only, Full | Complete; independent-switch identity preserved | Activation rates, phase/scale strata, energy gains | Mechanism diagnosis |
| EV5 | Are operator controls competitive in energy metrics? | Full, Attention, DeepSets | Complete; alternatives retained when better | Same fold protocol and explicit trade-offs | Operator control |
| EV6 | Is the effect robust at the factorial level? | Original, Exact, Full-no-PCGrad, Full | Complete | PCGrad x relation interaction for energy metrics | Strict attribution |
| EV7 | Does the relation adapter improve normalized reserve diagnostics? | 16 models plus existing seasonal/oracle references | Complete; old protocol parity below 1e-5 | Reserve, overload, over-reservation and proxy decomposition | Bounded operation table |
| EV8 | Do activation and phase features explain gains? | Full diagnostics + EV1/EV2 origin metrics | Complete, all 12,978 origins retained per setting | Exploratory fixed-rule correlations/strata; no outcome filtering | Mechanism evidence |
| EV9 | Are the conclusions visually auditable? | EV1-EV8 outputs | Complete, 7 main figures and 4 mechanism figures; QA passed | Paired city plots and full-model trade-offs | Figures / supplement |
| EV10 | Are strict component-removal claims identifiable? | Full-drop-scale, Full-drop-peer, Full-drop-source-envelope, Daily+full attenuation | Complete, 18 folds x 4 arms = 72 trainings | Same canonical budget, PCGrad and data protocol | Strict mechanism ablation |
| EV11 | Strict shared phase-relation operator | Same-batch Full vs `phase_shared_strict` | Complete, 18/18 | See audited 2026-11-17 design-control results below | Mechanism and energy attribution |
| EV12 | Parameter-matched stationwise control | Same-batch Full vs `stationwise_matched` | Complete, 18/18 | See audited 2026-11-17 design-control results below | Mechanism and energy attribution |
| EV13 | Concatenate-and-project fusion control | Same-batch Full vs `concat_project` | Complete, 18/18 | See audited 2026-11-17 design-control results below | Mechanism and energy attribution |

## Model roles

| Model | Role in this extension | Lineage |
|---|---|---|
| Original PhaseFormer | Primary baseline without PCGrad or relation adapter | `rebuild_20261030/p2_no_pcgrad_phaseformer` |
| Exact-PCGrad | PCGrad-only control; relation adapter off | canonical four-arm and component archives |
| Full-no-PCGrad | Relation adapter with mean-gradient training | `rebuild_20261031/r8_full_no_pcgrad` |
| SCPRC | Proposed relation adapter plus PCGrad | canonical four-arm archive |
| Daily relation | Daily relation control | canonical h24 anchor |
| Phase relation | Phase relation control | canonical h24 anchor |
| Scale-only / Peer-only / Source-disagreement-only | Independent-switch diagnostics | `rebuild_20261030/p2_components` |
| Attention / DeepSets | Operator controls | `mandatory_supplement/canonical_operator_controls_20261027` |
| Seasonal 24/168 | Periodic energy references | existing operation diagnostics |
| Full-drop-scale / Full-drop-peer / Full-drop-source-envelope | Strict gate removals, freshly trained | `energy_value_20261104/agent_strict_gap/runs` |
| Daily+full attenuation | Daily representation/relation replacement with all gates; EV11 isolates weight sharing | `energy_value_20261104/agent_strict_gap/runs` |

## Metric definitions

- `Peak`: for each origin, maximum city total over the 24 forecast hours; report magnitude error and peak-time absolute error in hours.
- `Ramp`: first difference of the city-total trajectory across adjacent forecast leads; report MAE and RMSE.
- `Total`: absolute error of the 24-hour city-total sum, with both native-unit MAE and normalized relative error.
- `Timing`: absolute difference between predicted and observed peak lead.
- `High-load`: realized city-total top-decile subgroup, reported descriptively because it uses test labels.
- `Activation`: fraction of diagnostic entries with each attenuation factor below 1, 0.9 and 0.5; report by city and seed.
- `Capacity`: normalized reserve proxy only. It is not a tariff, dispatch, transformer, queue or reliability result.

## Decision rules

1. The primary claim is supported only when Original vs Full is paired on the same fold and the metric direction is stated explicitly.
2. The module-value claim is supported by Exact vs Full only when peak/ramp/total/timing trade-offs are reported together. A gain in one energy metric does not imply universal superiority.
3. Independent-switch runs identify active mechanisms but cannot establish the contribution of removing that mechanism from the full model. EV10 is the strict follow-up.
4. No data filtering, test-protocol change or favorable-case selection is allowed. All 18 folds are retained.
5. Energy metrics support forecasting and normalized scenario evidence. They do not prove monetary savings, grid safety or dispatch benefit without the missing real-world inputs.

## Current interpretation boundary

The existing Daily replacement contrast changes both Q/K representation and relation sharing; EV11 below isolates normalized relation-weight sharing. Existing diagnostics show that Full lowers
the existing clipped peak MAE by 9.43% versus Exact and 4.42% versus
Daily+full attenuation. Ramp/Total/Timing benefits are not uniformly supported.
Full-drop-peer is better on peak/timing, while Full has a better fixed capacity
proxy. Source-envelope changes are very small. Keep these outcomes together;
the evidence does not establish that all gates are necessary.

Statistics are exploratory. The 18 folds share cities, outcomes and source data;
report both six-city cluster sensitivity and fixed-model time-block sensitivity.
No interval establishes an unseen-city, future-period or retraining guarantee.

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
