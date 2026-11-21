# External versus canonical protocol difference (E2)

Audit date: 2026-10-30 (local project date). Existing external files were read only.

| Dimension | Canonical primary CHARGED matrix | Jiaxing external | Shanghai MP-EVData external | UrbanEV external |
|---|---|---|---|---|
| Protocol identity | `scpcr-matched-baseline-20261023`, SHA-256 `dc0cb75e...` | `jiaxing_external_freeze_20261021`, SHA-256 `21f10145...` | `mp_evdata_external_freeze_20261021`, SHA-256 `870ee3f9...` | embedded external freeze in each [result.json](../../SOURCE_ARTIFACT_INDEX.md#src-bd4cf04bb281f795) |
| Source cities | LOCO: five source cities per target | all six CHARGED cities | all six CHARGED cities | all six CHARGED cities |
| Input / horizon | 168 → 24 h | 168 → 24 h | 168 → 24 h | 168 → 24 h |
| Evaluation origins | 721 per CHARGED city (August 2023) | 2,185 (July–September 2021) | 2,185 (Q4 2024) | 649 (February 2023) |
| Target series | CHARGED hourly volume | 13 transaction-derived kWh/hour proxy series | seven station mean-kW series | 275 Shenzhen zone-volume series |
| Scaler | source-only equal-city global scaler | fixed CHARGED source-only scaler | fixed CHARGED source-only scaler | source-global strict track; target-prefix scale only as sensitivity |
| Release layer | canonical primary table is offline candidate; no support fallback | fixed source-support certificate and strict release | fixed source-support certificate and strict release | fixed source-support certificate and strict release |
| Blindness | retrospective August audit | retrospective external case | retrospective external case | retrospective external case |

The external artifacts are therefore valid **separate external case studies**. Their hash mismatch against the canonical protocol is expected and does not indicate stale checkpoints. They must not be merged with the 18-fold canonical CHARGED table or described as if they used the canonical LOCO protocol.

The saved external Exact/SCPRC artifacts are internally consistent with the current implementation: all recorded core/wrapper hashes, prediction/checkpoint hashes, array shapes and finite checks, and stored metrics recompute successfully. No external retraining was launched because doing so without choosing a new source-scope freeze would create another incomparable result.
