# Third-party software and licenses

SCPRC uses PhaseFormer as its forecasting backbone and PCGrad for source-gradient projection. The repository also includes selected forecasting baselines and their dependencies. Each component retains its own attribution and license.

| Component | Upstream | Pinned local commit | License evidence |
|---|---|---|---|
| PhaseFormer | https://github.com/neumyor/PhaseFormer | `ff14c16fab136aef60048b20f2ef0d3bc6d86fc7` | No LICENSE in the pinned snapshot; reuse and redistribution terms remain unverified |
| PatchTST | https://github.com/yuqinie98/PatchTST | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | Apache-2.0; preserve upstream LICENSE |
| TimeMixer | https://github.com/kwuking/TimeMixer | `e24610583b36fdd8c76cc17a8df4e65759a5f460` | Apache-2.0; preserve upstream LICENSE and record local changes |
| Graph WaveNet | https://github.com/nnzhan/Graph-WaveNet | `6b162e80c59a1d494809252eca055cff93dc66b1` | MIT; preserve upstream LICENSE |
| DLinear reference | https://github.com/cure-lab/LTSF-Linear | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | Apache-2.0 verified at the pinned commit; upstream text preserved in `scprc/baselines/DLINEAR_LICENSE` |
| PCGrad | https://github.com/tianheyu927/PCGrad | See local experiment implementation and paper citation | Cite Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020 |

TimeMixer includes local modifications. Model adapters, training budgets, channel choices, and objectives are described in the experiment matrix. The N-BEATS graph residual and fixed-distance Graph WaveNet are diagnostic implementations; the latter is distinct from adaptive Graph WaveNet and MTGNN.

DLinear license text is included at the pinned upstream commit. The pinned PhaseFormer snapshot has no retrieved LICENSE, so its reuse and redistribution terms require clarification with the upstream rights holder. This repository does not assign a blanket MIT or Apache license to third-party components.

Dataset downloads and their licenses are documented in `DATA_SOURCES.md`; dataset files are not redistributed here.
