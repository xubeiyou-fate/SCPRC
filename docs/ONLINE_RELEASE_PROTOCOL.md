# SCPRC Optional Online Release Protocol

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](MIGRATION_SCOPE_ZH.md)。


## Algorithm and release-policy scope

The SCPRC is the PhaseFormer backbone plus the phase-conditioned relation adapter, trained on source-city data as documented in [`MODEL_ARCHITECTURE.md`](MODEL_ARCHITECTURE.md). Source-support screening and `SCPRC-Online(84,.25)` are optional deployment/release layers applied after Exact and candidate forecasts are produced; neither defines the offline candidate model. The parameters below freeze only that optional Online controller.

## Frozen optional deployment policy

The deployment policy frozen for separate release analysis is `SCPRC-Online(84,.25)`.

| Parameter | Frozen value |
|---|---:|
| Operational update block | 84 forecast origins |
| Forecast horizon | 24 hours |
| Exact minimum weight | 0.25 |
| Learning rate `eta` | 0.10 |
| Maximum candidate residual weight | 0.25 |
| Tail power | 8 |
| Experts | Exact, fixed-cap `p=1`, fixed-cap `p=8` |
| Safety deletion | Remove an alternative if either block RMSE or block MAE is worse than Exact |
| Fallback | Two consecutive blocks whose released mixture is worse on both RMSE and MAE |
| Fallback action | One Exact block followed by weight reset |
| Warm-up | Exact until the first complete labelled block is available |
| Label timing | Labels are used only after the complete forecast horizon is observable |

The policy combines the candidate and Exact forecasts continuously. For support ratio `r`, source support radius `R`, and cap `c=0.25`:

```text
alpha_p = c * min(1, R / max(r, epsilon)) ** p
expert_p = Exact + alpha_p * (candidate - Exact)
```

The online mixture updates expert weights only at completed 84-origin blocks using normalized RMSE/MAE loss. The Exact weight is projected back to at least 0.25 after each update.

## Online policy estimands

- Primary endpoint: city-macro relative RMSE versus PhaseFormer exact.
- Joint safety endpoint: relative MAE versus PhaseFormer exact.
- Unit of comparison: city × seed, with rolling origins retained as a temporal series.
- Secondary reporting: global RMSE/MAE, macro-site RMSE/MAE, coverage, fallback frequency, worst case, p95 regret, and mean expert weights.
- Uncertainty: paired non-circular moving-block bootstrap; rolling origins are not treated as independent observations.

## Required comparisons

1. PhaseFormer exact.
2. Always-candidate SCPRC.
3. Frozen `SCPRC-Online(84,.25)`.

The frozen online policy is the only deployment policy reported in `paper2`. Historical deployment variants remain outside the paper-facing evidence package.

## Claim boundary

All currently available evaluation windows have been inspected. Results based on them are labelled retrospective and cannot be described as blind confirmation or a formal risk guarantee.
