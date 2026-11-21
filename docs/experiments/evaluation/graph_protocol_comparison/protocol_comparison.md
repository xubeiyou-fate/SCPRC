# R11 protocol comparison

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](../../../MIGRATION_SCOPE_ZH.md)。


Status: **local diagnostic complete; faithful named-paper reproduction unavailable**.

The local N-BEATS-style run is useful as a transparent near-neighbor diagnostic.
The following table prevents it from being relabeled as the Energy method.

| Protocol item | Energy dynamic behavior-graph residual paper | Local diagnostic in this checkout | Match | Manuscript boundary |
|---|---|---|---|---|
| Main temporal backbone | N-BEATS; paper page describes strong temporal backbone and base quantile forecasts | Generic shared station-wise N-BEATS-style fully connected blocks | Partial | Say “N-BEATS-style local diagnostic,” not N-BEATS reproduction |
| Graph purpose | Dynamic behavior graph supplies a gated residual correction | History-conditioned query/key graph supplies bounded centered peer-forecast correction | Partial | Mechanism is conceptually related, implementation differs |
| Static graph prior | Paper page says static priors are fused with sample-specific graph | No verified paper-matched static prior | No | Cannot claim static-prior reproduction |
| Dynamic graph construction | Recent load profiles; exact paper construction unavailable locally | 168-hour descriptor, shared query/key projections, dense directed softmax graph | Partial | Report local construction explicitly |
| Residual gate | Gated residual path in paper | Bounded tanh correction, centered across sites; no paper-matched gate | No/partial | Do not claim gate equivalence |
| Point output | Paper produces point median as part of quantile output | Point forecast only | No | Local RMSE/MAE are not the paper's probabilistic result |
| Quantiles | Base quantile forecasts | No quantile head | No | No Pinball, WIS80, calibration, or interval claim for local run |
| Peak-risk task | Companion peak-risk head and loss-weight experiments | No peak labels or risk head | No | No peak-risk reproduction or risk trade-off claim |
| Dataset | UrbanEV primary; CHARGED external transfer | CHARGED v3 six-city LOCO only | No | Local run is a CHARGED diagnostic |
| UrbanEV split | Publisher page describes chronological 60%/20%/20% split over 275 stations | Not implemented in this runner | No | No UrbanEV faithful comparison |
| CHARGED role | External validation across six cities | Six-city CHARGED is the complete local evaluation | Partial | Comparable dataset name only; not paper protocol match |
| Horizons | Four horizons reported by paper page | One canonical 24-hour horizon | No | No multi-horizon reproduction claim |
| Seeds | Five random seeds reported by paper page | Three seeds: 13, 17, 29 | No | Local uncertainty is three-seed only |
| Metrics | MAE, RMSE, Pinball, WIS80, risk metrics | Raw-volume RMSE and MAE | Partial | Compare point metrics only as local diagnostics |
| Training split | Paper-specific chronological protocol | April–July source-only five-city LOCO | No | Do not call protocol matched |
| Optimizer/loss | Exact paper details not available in local protocol | AdamW, source-city PCGrad, standardized MSE plus multiresolution consistency | No | Training protocol differs |
| Blindness | Paper page reports chronological evaluation; exact release audit unavailable | August was exposed to previous development | No | Mark retrospective; no blind confirmation |
| Code | No exact official code found in checkout | Independent local implementation and recorded checkpoints | No | Faithful reproduction unavailable |

## Existing local result

The completed local matrix has 18 city×seed rows. Equal-fold means:

| Local model | RMSE | MAE |
|---|---:|---:|
| N-BEATS-style history behavior graph | 64.2645 | 12.4309 |
| PhaseFormer Exact reference | 35.8956 | 8.4028 |

The local diagnostic is also approximately ten times the parameter count of
PhaseFormer Exact according to the existing audit. It is therefore not a
capacity-matched comparison. City-scale errors are highly heterogeneous,
especially for MEL and SZH, so the equal-fold mean is descriptive and should
not be interpreted as a general claim about the named Energy paper.

## Reproduction boundary

### Reportable

> We additionally evaluated an independent N-BEATS-style history-conditioned
> behavior-graph residual under the CHARGED 168-to-24 diagnostic protocol. This
> local implementation achieved RMSE 64.2645 and MAE 12.4309 averaged over the
> 18 city–seed folds, versus 35.8956 and 8.4028 for the PhaseFormer Exact
> reference. Because the local implementation does not include the cited
> Energy method's probabilistic heads, peak-risk task, paper-specific
> preprocessing, or official code, this result is reported only as a local
> diagnostic and not as a faithful reproduction or superiority test.

### Not reportable

- “The Energy dynamic behavior-graph method was reproduced.”
- “SCPRC outperforms the Energy paper's method.”
- “The local point RMSE/MAE establish worse WIS or peak-risk performance.”
- “The local result verifies the paper's five-seed UrbanEV protocol.”
- “The local result is blind confirmation.”

## Required assets for a future faithful reproduction

The claim can only be upgraded after obtaining and hashing the named paper's
official code or a complete author protocol package, including: UrbanEV data
release and preprocessing, chronological split, exact horizons, all five
seeds, quantile definitions and pinball loss, WIS80 calculation, static graph
priors, dynamic graph construction, residual gate, peak labels, risk target,
risk loss weights, and the CHARGED transfer release policy.
