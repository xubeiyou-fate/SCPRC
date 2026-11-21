# E2 External Matrix Version Audit (2026-10-30)

- Audit timestamp (UTC): `2026-10-29T18:41:02.182371+00:00`
- Scope: Jiaxing, Shanghai MP-EVData, and UrbanEV Shenzhen; seeds `13/17/29`.

## Result

The existing external Exact/SCPRC artifacts are internally consistent with the current implementation: the recorded core hashes, external runner hashes, prediction archive hashes, checkpoint hashes, prediction shapes, finite-value checks, and selected metric recomputations all pass.

The external artifacts do **not** use the canonical primary offline protocol hash `dc0cb75e...`. This is expected from the design: the primary CHARGED matrix is LOCO (five source cities per target, 721 August origins, CHARGED volume units), while each external case trains on all six CHARGED source cities and uses its own frozen target window and native target unit.

Therefore the canonical hash mismatch is a protocol identity difference, not evidence that the saved external checkpoints are stale or corrupted. The external results should remain separate case studies and must not be merged into the 18-fold canonical table.

## Dataset-specific boundaries

- **Jiaxing:** 2,185 rolling 168→24 origins, 13 location series, transaction-allocated kWh/hour proxy, July–September 2021 test.
- **Shanghai MP-EVData:** 2,185 rolling 168→24 origins, seven continuous-load stations, hourly mean kW, Q4 2024 test.
- **UrbanEV Shenzhen:** 649 February 2023 origins, 275 zones, rated-power-derived zone-volume target; same city as CHARGED-SZH, so this is cross-dataset evidence rather than a seventh independent city.

## Rerun decision

No GPU rerun was launched. Replacing these runs would require a new external protocol freeze that explicitly chooses the source scope (all six source cities versus LOCO five-city training), target scaling, and release policy. Running the external scripts under the canonical LOCO protocol without that decision would produce an incomparable experiment.

See [external_protocol_audit.csv](../../../results/mandatory_supplement/external_version_audit/external_protocol_audit.csv) for one row per dataset×seed, [external_metric_recompute_check.csv](../../SOURCE_ARTIFACT_INDEX.md#src-c265395d1a19c4d8) for stored-versus-recomputed metrics, [external_exact_full_final.csv](../../../results/mandatory_supplement/external_version_audit/external_exact_full_final.csv) for the compact external Exact/SCPRC table, and [external_version_manifest.json](../../../results/mandatory_supplement/external_version_audit/external_version_manifest.json) for full provenance and hashes.

