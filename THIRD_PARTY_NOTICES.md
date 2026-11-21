# Third-party provenance and license status

The repository contains the SCPRC experiment integration and selected upstream forecasting implementations. Upstream models and PCGrad are attributed methods; the repository does not claim to have invented them. Licenses of upstream code and datasets are separate from the project's own code.

| Component | Upstream | Pinned local commit | License evidence |
|---|---|---|---|
| PhaseFormer | https://github.com/neumyor/PhaseFormer | `ff14c16fab136aef60048b20f2ef0d3bc6d86fc7` | No LICENSE found in the pinned local snapshot; redistribution terms require clarification before public release |
| PatchTST | https://github.com/yuqinie98/PatchTST | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | Apache-2.0; preserve upstream LICENSE |
| TimeMixer | https://github.com/kwuking/TimeMixer | `e24610583b36fdd8c76cc17a8df4e65759a5f460` | Apache-2.0; preserve upstream LICENSE and record local changes |
| Graph WaveNet | https://github.com/nnzhan/Graph-WaveNet | `6b162e80c59a1d494809252eca055cff93dc66b1` | MIT; preserve upstream LICENSE |
| DLinear reference | https://github.com/cure-lab/LTSF-Linear | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | The local adapter records this origin; its corresponding license text still requires verification |
| PCGrad | https://github.com/tianheyu927/PCGrad | See local experiment implementation and paper citation | Cite Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020 |

TimeMixer contains local changes. Model adapters, training budgets, channel choices and objectives also differ across baselines; consult the experiment matrix. The local N-BEATS graph residual is a diagnostic implementation. Fixed-distance Graph WaveNet must not be described as adaptive Graph WaveNet or MTGNN.

The local migration does not assign a blanket MIT/Apache license to all these files. This file records actual provenance and unresolved terms. Attempts to retrieve missing upstream licenses during migration failed because of network TLS errors; absence of a retrieved license is not proof of permission. Local preparation can be reviewed now; public redistribution should resolve the PhaseFormer and DLinear items.

Dataset downloads and their licenses are documented in `DATA_SOURCES.md`; dataset files are not redistributed here.
