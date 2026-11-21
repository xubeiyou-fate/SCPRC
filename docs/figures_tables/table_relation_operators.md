# Relation and component attribution on CHARGED

This table is the mechanism-attribution matrix. The main paper baseline is Original PhaseFormer; Exact-PCGrad is the intermediate control that isolates the PCGrad step before the SCPRC relation components are added.

All canonical rows use the same six-city leave-one-city-out protocol, three seeds, source-only normalization, 18 city×seed folds and 721 August origins per fold. Results are retrospective raw offline predictions.

## Main baseline and PCGrad control

| Model variant | Total parameters | RMSE | MAE | Role |
|---|---:|---:|---:|---|
| **Original PhaseFormer (no-PCGrad)** | 36,360 | 35.0675 | 8.2817 | **main baseline** |
| Exact-PCGrad | 36,360 | 32.5417 | 7.0227 | PCGrad-only mechanism control |
| Daily relation-PCGrad | 37,263 | 32.2066 | **6.9798** | daily relation control |
| Phase relation-PCGrad | 37,263 | 32.6624 | 7.0384 | phase relation control |
| **SCPRC** | **37,263** | **32.1907** | 7.0498 | **main method** |

Interpretation: Original → Exact-PCGrad estimates the PCGrad contribution; Exact-PCGrad → Full estimates the added relation contribution. Full versus Original is the method-level result. Daily relation has lower MAE than Full, so the table supports a metric trade-off rather than universal superiority.

## Independent component ablation

| Variant | RMSE | MAE | Interpretation |
|---|---:|---:|---|
| Scale-only | 32.1563 | 7.0385 | independent scale gate |
| Peer-only | 32.7614 | 7.1095 | independent peer gate |
| Source-disagreement-only | 32.6720 | 7.0840 | independent disagreement gate |
| SCPRC | 32.1893 | 7.0497 | all three components |

These rows are the separate P2 independent-switch matrix. They are not silently merged with the R1 four-arm output file.

## Matched generic relation-operator controls

| Model | Total parameters | RMSE | MAE | Role |
|---|---:|---:|---:|---|
| SCPRC | 37,263 | 32.1907 | 7.0498 | reference |
| Phase-wise generic attention | 37,257 | 32.6658 | 7.0180 | parameter-matched operator control |
| PhaseFormer + DeepSets adapter | 37,255 | 32.5458 | 7.0029 | parameter-matched set control |

SCPRC has lower RMSE than both tested operators, while both controls have lower MAE. Report this as a conditional operator trade-off. The paired temporal-block bootstrap is conditional on the six observed cities and three seeds; it does not quantify new-city or prospective uncertainty.

Historical generic-attention and DeepSets outputs are indexed in [`../archive/legacy_20261027/INDEX.md`](../SOURCE_ARTIFACT_INDEX.md#src-df3694d61e1fdde2). The optional Online release layer is excluded from this attribution table.
