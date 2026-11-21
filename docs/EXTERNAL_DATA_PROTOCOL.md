# External datasets

Jiaxing, Shanghai MP-EVData, and UrbanEV are evaluated separately. The Exact/SCPRC rows use protocol v2.2, recorded on 2026-10-24. They are distinct from the main CHARGED Original PhaseFormer versus SCPRC comparison.

| Dataset | Target and unit | Interpretation |
|---|---|---|
| Jiaxing | Session energy allocated to hourly bins; kWh proxy | 13 location aggregates; some seeds have higher errors than Exact |
| Shanghai MP-EVData | Hourly mean station power, kW | The daily-mean rule performs better than the reported learned models |
| UrbanEV | Rated-power-derived zone volume | Shenzhen cross-dataset evaluation; it shares a city with CHARGED-SZH |

RMSE and MAE are reported in each dataset's own units. The three-seed results are in [external baseline results](mandatory_supplement/external_baselines/summary.md).

External baseline checkpoints are listed in [CHECKPOINTS.csv](../manifests/CHECKPOINTS.csv). The 18 v2.2 Exact/SCPRC checkpoints are not included. Available summaries, configuration records, and other source references are described in [release contents](MIGRATION_SCOPE_ZH.md).
