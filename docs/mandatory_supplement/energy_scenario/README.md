# E1 Energy-operation scenario replay (CHARGED 18-fold)

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](../../MIGRATION_SCOPE_ZH.md)。


Status: completed read-only replay; no model retraining. All August labels were already exposed during development, so this is retrospective evidence rather than blind confirmation.

## Protocol

- Six cities × three seeds = 18 city×seed folds; 721 origins/fold; 24-hour horizon; 168-hour context.
- Predictions come from the canonical homogeneous 2026-10-27 matrix; Exact and SCPRC use `pred_exact_pcgrad_raw` and `pred_scpcr_candidate_raw`.
- Seasonal naive rules are reconstructed causally from the preceding 168 observed hours in the pinned CHARGED archive.
- Primary scenario: 10% a-priori safety margin and overload penalty λ=10; margins 0%, 5%, 10%, 20% and λ values 1.0, 5.0, 10.0, 20.0 are sensitivity analyses.
- City capacity is `max_24h(predicted aggregate volume) × (1 + margin)`. Overload volume is the positive hourly excess over that fixed capacity.
- Relative cost is `reserved_capacity + λ × overload_volume`; normalized relative cost divides each origin's cost by its actual aggregate peak. No monetary prices or physical kW conversion are assumed.
- Oracle boundary: `Oracle actual peak (unavailable)` uses the realized peak itself, hence zero overload and zero idle capacity. It is a diagnostic lower bound and cannot be used in deployment or tuning.

## Primary capacity/risk results (equal mean over 18 folds; margin=10%, λ=10)

| Model | Reserved capacity | Actual peak | Capacity/peak (%) | Overload-window rate (%) | Overload hours | Overload volume | Idle capacity | Normalized cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact PhaseFormer | 57742.2595 | 55002.1180 | 103.93 | 34.20 | 1.051 | 255.8489 | 2884.8240 | 1.9117 |
| SCPRC | 57990.1198 | 55002.1180 | 103.51 | 35.27 | 1.103 | 234.4433 | 3120.3645 | 1.9580 |
| Oracle actual peak (unavailable) | 55002.1180 | 55002.1180 | 100.00 | 0.00 | 0.000 | 0.0000 | 0.0000 | 1.0000 |
| Seasonal naive 168 h | 60722.1277 | 55002.1180 | 111.90 | 18.84 | 1.028 | 150.2803 | 5781.4025 | 2.1853 |
| Seasonal naive 24 h | 60566.5432 | 55002.1180 | 110.91 | 16.67 | 0.601 | 129.1049 | 5624.6125 | 1.8084 |

## Outputs

- [capacity_scenario_per_origin.csv](../../SOURCE_ARTIFACT_INDEX.md#src-0cb9a4ca61b986b8): origin-level capacity, overload, idle-capacity, and relative-cost values.
- [capacity_scenario_per_fold.csv](../../SOURCE_ARTIFACT_INDEX.md#src-0f91817d2284ffc6): fold-level summaries for all margin/penalty settings.
- [capacity_scenario_equal_fold_summary.csv](../../../results/mandatory_supplement/energy_scenario/capacity_scenario_equal_fold_summary.csv): equal-fold means across the 18 city×seed folds.
- [horizon_error_curve.csv](../../SOURCE_ARTIFACT_INDEX.md#src-744f60e27ed29c09): 1–24 h RMSE/MAE curves.
- [peak_error_decomposition.csv](../../SOURCE_ARTIFACT_INDEX.md#src-f666b67da41ae62e): within-origin 75th-percentile peak-hour versus off-peak errors.
- [capacity_risk_tradeoff.png](../../SOURCE_ARTIFACT_INDEX.md#src-e3c583d00e80f662), [relative_cost_sensitivity.png](../../SOURCE_ARTIFACT_INDEX.md#src-6244ddb8c8890d0b), [horizon_error_curve.png](../../SOURCE_ARTIFACT_INDEX.md#src-b3a39b52d40649ac), [peak_error_decomposition.png](../../SOURCE_ARTIFACT_INDEX.md#src-84c1193834dc5d8b): publication-draft diagnostics.

## Interpretation boundary

These metrics quantify a deployment-style capacity reservation proxy in the dataset's native volume units. Without site capacity, electricity price, or overload damage data, the relative-cost curves should not be reported as monetary savings or grid reliability guarantees. The model can reduce or increase capacity risk depending on city, seed, safety margin, and penalty; report the city-level table and sensitivity rather than a single universal dominance claim.
