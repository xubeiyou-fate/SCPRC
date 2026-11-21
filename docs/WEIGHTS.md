# Checkpoints

The [paper-v1 release](https://github.com/xubeiyou-fate/SCPRC/releases/tag/paper-v1) contains 14 ZIP files and a `SHA256SUMS` file. The ZIPs restore 534 checkpoint paths from 528 unique payloads and six aliases. Data and prediction arrays are not included.

## Download and restore

Download the ZIP files and `SHA256SUMS` into `release_assets/` in the repository. With GitHub CLI:

```bash
mkdir -p release_assets
gh release download paper-v1 --repo xubeiyou-fate/SCPRC --pattern '*.zip' --pattern SHA256SUMS --dir release_assets
(cd release_assets && sha256sum -c SHA256SUMS)
python scripts/restore_checkpoints.py
python scripts/verify_checkpoints.py
```

GitHub's source-code download does not include these checkpoint attachments.

The restoration script checks archive and file hashes and creates ordinary files under `checkpoints/`. Existing identical files are retained; a different file at the same path produces an error. The six aliases are materialized by the script, so manual extraction alone does not restore the complete set.

## Experiment groups

| ZIP group | Checkpoint paths |
|---|---:|
| `charged_dlinear.zip` | 18 |
| `charged_nbeats_graph.zip` | 18 |
| `charged_patchtst.zip` | 18 |
| `charged_timemixer.zip` | 18 |
| `design_controls.zip` | 72 |
| `external_baselines.zip` | 30 |
| `gate_removal.zip` | 72 |
| `independent_gates.zip` | 90 |
| `phaseformer_h24.zip` | 18 |
| `phaseformer_h48.zip` | 18 |
| `relation_operators.zip` | 36 |
| `scprc_h24_and_controls.zip` | 72 |
| `scprc_h48_and_control.zip` | 36 |
| `scprc_mean_gradient.zip` | 18 |

To restore only the 24-hour SCPRC and relation controls:

```bash
python scripts/restore_checkpoints.py --group scprc_h24_and_controls
```

The full verification command expects all groups. Check `--help` for the available verification modes when working with a subset.

## File identities

[CHECKPOINTS.csv](../manifests/CHECKPOINTS.csv) records each model, variant, city, seed, dataset, horizon, source identity, and SHA-256. Main, independent-gate, gate-removal, paired-design, and 48-hour references are separate runs.

DLinear assets contain the DLinear parameters and required inference metadata. Parameter and fixed-input consistency checks are recorded in `manifests/CHECKPOINT_MIGRATION_AUDIT.json`.
