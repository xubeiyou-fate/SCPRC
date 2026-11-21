# R10 live run status

Updated: 2026-11-12T20:01:51+0800.

- Primary queue: **18/18 completed**; PID **4049557** is preserved.
- Next unmaterialized fold in primary order: **none**.
- Manual reserved workers are observed only; no replacement worker is started.
- Baseline identity: **fixed-distance Graph WaveNet**. It is not adaptive Graph WaveNet, MTGNN or DCRNN.

## Reservation ledger

| Fold | Worker | PID | State |
|---|---|---:|---|
| JHB_seed17 | `parallel_jhb17` | 4071008 | destination_hash_mismatch |
| LOA_seed17 | `parallel_loa17` | 4071009 | consolidated |

Each copied fold must contain [result.json](../../../SOURCE_ARTIFACT_INDEX.md#src-f5620e9e04403f39), [predictions.npz](../../../SOURCE_ARTIFACT_INDEX.md#src-a745a6471f159fc1), and [state.pt](../../../SOURCE_ARTIFACT_INDEX.md#src-26c8759dfe4cfe24); source and destination SHA-256 values are recorded in [consolidation_status.json](../../../../results/experiments/evaluation/spatial_baseline_review/consolidation_status.json).
