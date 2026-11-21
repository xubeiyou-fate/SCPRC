# Offline SCPRC candidate accuracy: Original PhaseFormer main comparison

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](../MIGRATION_SCOPE_ZH.md)。


This is the main offline-accuracy table. The primary comparison is **Original PhaseFormer (no-PCGrad) versus SCPRC**. `Exact-PCGrad` is a mechanism-ablation control and is reported separately. The old Conventional Exact row remains historical provenance and is not used as the main baseline.

All CHARGED scores are retrospective raw-volume RMSE/MAE with lower values better. The canonical protocol is six cities, three seeds, 18 city×seed folds, 168 h input, 24 h output and 721 August origins per fold.

## Main comparison

| Model | RMSE | MAE | Parameters | Evidence status |
|---|---:|---:|---:|---|
| **Original PhaseFormer (no-PCGrad)** | 35.0675 | 8.2817 | 36,360 | canonical 18-fold strict control; main baseline |
| **SCPRC** | **32.1907** | **7.0498** | 37,263 | canonical 18-fold candidate |
| PatchTST, fair volume-only | 42.8268 | 8.1441 | 25,913 | 18/18 audited folds |
| TimeMixer, fair volume-only | 44.4249 | 8.3487 | 293,263 | 18/18 audited folds |
| Seasonal naive (24 h) | 35.8339 | 7.5424 | — | retrospective rule baseline |

SCPRC relative to Original PhaseFormer improves equal-fold mean RMSE by approximately 8.20% and MAE by approximately 14.87%. This is the current main effect. It is separate from the historical Conventional Exact comparison.

## Mechanism-ablation controls

| Model / variant | RMSE | MAE | Parameters | Role |
|---|---:|---:|---:|---|
| Exact-PCGrad | 32.5417 | 7.0227 | 36,360 | PCGrad-only control; no SCPRC gates |
| Daily relation-PCGrad | 32.2066 | **6.9798** | 37,263 | daily-relation control |
| Phase relation-PCGrad | 32.6624 | 7.0384 | 37,263 | phase-relation control |
| **SCPRC** | **32.1907** | 7.0498 | 37,263 | full combination |

This panel estimates the mechanism chain. SCPRC versus Original PhaseFormer is the full-method comparison; Full versus Exact-PCGrad estimates the added relation components after PCGrad. Full does not win every metric: Daily relation has lower MAE.

## Strict independent-component controls

| Model / variant | RMSE | MAE | Role |
|---|---:|---:|---|
| Original PhaseFormer (no-PCGrad) | 35.0675 | 8.2817 | main baseline |
| Exact-PCGrad | 32.5417 | 7.0227 | PCGrad contribution control |
| Scale-only | 32.1563 | 7.0385 | independent scale gate |
| Peer-only | 32.7614 | 7.1095 | independent peer gate |
| Source-disagreement-only | 32.6720 | 7.0840 | independent disagreement gate |
| SCPRC | 32.1893 | 7.0497 | full component combination |

The independent-component table is the 18-fold P2 matrix. Its Full value is kept distinct from the R1 canonical four-arm value because the output artifacts have separate run identities.

## Historical Conventional Exact reference

| Model / variant | RMSE | MAE | Evidence status |
|---|---:|---:|---|
| PhaseFormer Exact | 35.8956 | 8.4028 | legacy Conventional Exact matrix; provenance only |

This row must not be renamed `Original PhaseFormer`, and it must not be pooled with `Exact-PCGrad`. The historical 10.32% RMSE and 16.10% MAE reductions are excluded from the current headline.

## Other offline baselines

| Model | RMSE | MAE | Evidence status |
|---|---:|---:|---|
| Seasonal naive (168 h) | 36.0085 | 8.2180 | retrospective rule baseline |
| Persistence (last value) | 78.6753 | 11.6569 | retrospective rule baseline |
| Previous-24 h mean | 57.2497 | 9.9171 | retrospective rule baseline |
| DLinear (two-variable) | 70.9589 | 13.4285 | retrospective trained baseline; qualified optimization match |
| N-BEATS + behavior-graph residual | 64.2645 | 12.4309 | local diagnostic; not faithful paper reproduction |

PatchTST and TimeMixer use the matched exposure budget of 8,192 source windows and effective batch 4. N-BEATS remains a local diagnostic and is not a capacity-matched SOTA claim. Do not claim blanket superiority over every baseline.

The optional `SCPRC-Online(84,.25)` release result is excluded and reported separately in [table_primary_results.md](../SOURCE_ARTIFACT_INDEX.md#src-6d300dc7d5e314a7).
