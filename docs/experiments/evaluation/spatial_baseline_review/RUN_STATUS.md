# R10 live run status

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](../../../MIGRATION_SCOPE_ZH.md)。


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
