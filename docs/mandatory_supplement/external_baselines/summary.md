# External strong-baseline results (2026-10-28)

Metrics were recomputed from saved truth/prediction arrays. No cross-dataset RMSE/MAE pooling is performed.
Seed means and SDs summarize the three fixed training seeds within each dataset; they are not city-level population uncertainty.
The Exact/SCPRC comparison rows are from the archived 2026-10-24 external matrix (SCPRC-v2.2), not the newer homogeneous CHARGED 2026-10-27 matrix.
Jiaxing uses a session-derived hourly-energy proxy (kWh); Shanghai is published station power (kW); UrbanEV retains its native volume unit.
All experiments are retrospective. No external test score is used for training, scaling, or model selection.

## jiaxing

| Model | RMSE mean ± SD | MAE mean ± SD | n seeds |
|---|---:|---:|---:|
| DLINEAR | 26.0739 ± 0.0648 | 19.2442 ± 0.1002 | 3 |
| Daily-mean rule (legacy external matrix) | 33.9683 ± 0.0000 | 26.0119 ± 0.0000 | 3 |
| NBEATS_GRAPH | 26.9033 ± 0.2207 | 19.7756 ± 0.3123 | 3 |
| PATCHTST | 27.2789 ± 0.1779 | 20.1866 ± 0.1495 | 3 |
| Persistence / last-value rule (legacy external matrix) | 46.9571 ± 0.0000 | 33.7134 ± 0.0000 | 3 |
| PhaseFormer Exact (legacy external matrix) | 26.4107 ± 2.2231 | 19.2980 ± 1.9823 | 3 |
| SCPRC candidate (v2.2 legacy external matrix) | 25.0502 ± 0.3605 | 18.3736 ± 0.3827 | 3 |
| TIMEMIXER | 26.6191 ± 0.8067 | 19.6504 ± 0.6368 | 3 |
| Weekly-naive rule (legacy external matrix) | 32.4004 ± 0.0000 | 22.7603 ± 0.0000 | 3 |

## shanghai

| Model | RMSE mean ± SD | MAE mean ± SD | n seeds |
|---|---:|---:|---:|
| DLINEAR | 288.5929 ± 19.6069 | 158.1101 ± 11.9861 | 3 |
| Daily-mean rule (legacy external matrix) | 175.4674 ± 0.0000 | 82.8385 ± 0.0000 | 3 |
| NBEATS_GRAPH | 593.7645 ± 18.0275 | 299.5258 ± 7.5222 | 3 |
| PATCHTST | 317.6650 ± 9.2639 | 176.6518 ± 4.8098 | 3 |
| Persistence / last-value rule (legacy external matrix) | 1328.5340 ± 0.0000 | 501.5145 ± 0.0000 | 3 |
| PhaseFormer Exact (legacy external matrix) | 394.4765 ± 87.3778 | 196.5632 ± 86.2367 | 3 |
| SCPRC candidate (v2.2 legacy external matrix) | 333.0230 ± 18.7666 | 188.4495 ± 17.6785 | 3 |
| TIMEMIXER | 360.8646 ± 20.0615 | 182.0132 ± 2.2282 | 3 |
| Weekly-naive rule (legacy external matrix) | 225.3904 ± 0.0000 | 105.0237 ± 0.0000 | 3 |

## urbanev

| Model | RMSE mean ± SD | MAE mean ± SD | n seeds |
|---|---:|---:|---:|
| DLINEAR | 208.4042 ± 3.9763 | 55.8891 ± 0.8559 | 3 |
| Daily-mean rule (legacy external matrix) | 426.9106 ± 0.0000 | 106.7412 ± 0.0000 | 3 |
| NBEATS_GRAPH | 255.3611 ± 13.6485 | 67.5440 ± 4.0701 | 3 |
| PATCHTST | 218.0781 ± 3.0968 | 58.0634 ± 1.3013 | 3 |
| Persistence / last-value rule (legacy external matrix) | 591.3484 ± 0.0000 | 133.1669 ± 0.0000 | 3 |
| PhaseFormer Exact (legacy external matrix) | 230.8918 ± 24.4139 | 60.2168 ± 6.8805 | 3 |
| SCPRC candidate (v2.2 legacy external matrix) | 202.1759 ± 3.1450 | 53.3197 ± 0.5833 | 3 |
| TIMEMIXER | 231.1178 ± 10.2858 | 62.0088 ± 3.3806 | 3 |
| Weekly-naive rule (legacy external matrix) | 322.2413 ± 0.0000 | 77.1326 ± 0.0000 | 3 |

## Protocol interpretation

- PatchTST uses the pinned local official adapter, one volume channel, source-only six-city CHARGED training, 2048 source minibatches × 4 windows, and PCGrad over source-city tasks (342 parameter updates).
- TimeMixer uses the vendored official CD route, volume-only input, source-only six-city CHARGED training, 2048 minibatches × 4 windows (2048 optimizer updates).
- N-BEATS graph control uses the local shared N-BEATS-style backbone plus history-conditioned behavior-graph residual, source-only six-city CHARGED training, and 2048 minibatches × 4 windows. It is not a reproduction of the cited EV paper; its objective and graph structure differ from SCPRC.
- DLinear uses the local shared univariate LTSF-Linear implementation, source-only six-city CHARGED training, 2048 updates × 4 windows. It is an additional external control, not a neural strong-baseline family equivalent to PatchTST/TimeMixer.
- Exposure is matched as 8192 sampled source windows, but optimizer steps differ across PCGrad PatchTST and single-task controls; do not claim optimizer-update identity.
- Report each dataset separately. Historical candidate/release rows are lineage context only and are not treated as the new SCPRC-Online protocol.
- [paired_block_bootstrap_168origin.csv](../../../results/mandatory_supplement/external_baselines/paired_block_bootstrap_168origin.csv) gives within-dataset paired moving-block intervals (168 consecutive forecast origins), conditional on the three fixed trained seeds; this is not population uncertainty over cities.
- The graph control is a local N-BEATS-style behavior-graph baseline. It does not establish superiority over adaptive Graph WaveNet/MTGNN, and it is not a faithful reproduction of a named N-BEATS EV paper.
