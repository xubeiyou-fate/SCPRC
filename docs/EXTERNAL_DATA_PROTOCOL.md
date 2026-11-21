# External Data and Result Boundaries

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](MIGRATION_SCOPE_ZH.md)。


The external matrix is a collection of separate retrospective case studies, not one pooled benchmark. Its current reported Exact/SCPRC rows come from the 2026-10-24 v2.2 lineage; they are not a new Original PhaseFormer (no-PCGrad) vs SCPRC matched matrix.

| Dataset | Target and unit | Interpretation boundary |
|---|---|---|
| Jiaxing | Session energy allocated uniformly into hourly bins; kWh proxy | 13 location aggregates, not 13 individual chargers; some seeds regress vs Exact |
| Shanghai MP-EVData | Hourly mean station power, kW | Daily-mean rule is substantially stronger than the learned models; retain as a negative/boundary case |
| UrbanEV | Rated-power-derived zone volume | Shenzhen, same city as CHARGED-SZH; cross-dataset/spatial-scale test, not an independent city |

Do not pool RMSE or MAE across these units. The complete three-seed summary table is in [mandatory_supplement/external_strong_baselines_20261028/summary.md](mandatory_supplement/external_baselines/summary.md). The local source files and model arrays remain on the server and are not part of the writing bundle.

