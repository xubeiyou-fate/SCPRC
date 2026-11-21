# SCPRC Framework and Optional Release Layer

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](MIGRATION_SCOPE_ZH.md)。


The primary model evaluated in this paper is the **SCPRC forecaster**. The main comparison asks whether its phase-conditioned relation adapter improves forecasting over **Original PhaseFormer (no-PCGrad)** and the eligible matched baselines. Original and Full use the same PhaseFormer architecture; the historical run named `PhaseFormer Exact` is a distinct legacy result and is not the current main baseline. `Exact-PCGrad` is an attribution control. `SCPRC-Online(84,.25)` is an optional deployment/release policy that post-processes candidate and Exact forecasts; it is not part of the offline candidate score and is not required to use the candidate model. The primary implementation lineage is the end-to-end internal PhaseFormer adapter used by the CHARGED LOCO runs. A separate frozen-backbone sidecar residual calibrator exists in [reproduction/models/phase_relation_aware_bounded_residual.py](SOURCE_ARTIFACT_INDEX.md#src-0a2f3ecd71446fed); do not silently merge its formulation or results with the primary implementation.

## 1. Problem and notation

At forecast origin `t`, the model receives histories for a variable-size set of `N` charging stations, `X_t ∈ R^(B×168×N)`, and predicts the next 24 hours. The main CHARGED setting trains on several source cities and evaluates a held-out city. The target forecast-period labels do not train the relation adapter. Station identifiers, coordinates, and a fixed target-city adjacency matrix are not inputs to the relation operator.

The **offline candidate model** consists of a pinned PhaseFormer backbone and a phase-conditioned inter-station relation adapter trained on source-city data. The backbone is inherited from PhaseFormer; the paper's algorithmic claim must be attached to the adapter formulation and supported by its controls, not to the backbone or optimizer alone. Deployment-time selection and release are separate optional stages.

## 2. End-to-end SCPRC framework

### 2.1 PhaseFormer representation

The 168-hour history is normalized by the pinned RevIN implementation and reorganized into seven daily cycles and 24 within-day phases. The backbone applies its phase embedding, routing layers, prediction head, and inverse normalization to produce the 24-hour forecast. The relation adapter modifies the history-phase representation before the backbone phase embedding; it does not replace the PhaseFormer forecaster.

### 2.2 Phase-conditioned station relation

Let `X̄ ∈ R^(B×N×24×7)` be the normalized phase representation. For each station and phase, the seven observed daily values are encoded by shared profile layers. Shared query/key projections with `ELU + 1` produce positive features `q_i,k_i ∈ R^r`; values are centered across the station set. The phase-wise relation descriptor is

```text
R_i = q_iᵀ [Σ_j k_j v_j] / [q_iᵀ Σ_j k_j + ε].
```

The aggregation is evaluated through low-rank sufficient statistics, without explicitly materializing an `N×N` adjacency/relation matrix. The current implementation uses rank 8 and accumulates relation reductions in FP32. Shared weights make the operator usable with varying station counts and permutation-equivariant to station ordering.

A small correction network maps the relation descriptor to a residual phase correction. Its final layer is zero-initialized, so the adapter begins as an identity perturbation of the PhaseFormer input. The correction is centered over stations to preserve the city-level mean component. The source-certified implementation additionally attenuates corrections using station-scale, peer-reliability, and source-derived relation-disagreement constraints. These are constraints on the adapter correction; they are distinct from the later release-time support score.

The core implementation is [`phase_relations.py`](../scprc/training/phase_relations.py), with the source-certified end-to-end wrapper in [`scprc.py`](../scprc/models/scprc.py). The exact runner/configuration used for any cited result must be taken from that result's manifest, not inferred from a filename.

### 2.3 Multi-source training

The principal CHARGED v2.2 training lineage forms one loss/gradient task per source city. It computes source-city gradients, applies the established PCGrad projection to conflicting gradients, averages the projected gradients, clips the resulting norm, and updates model parameters with AdamW. The recorded objective is standardized MSE plus a `0.15` multi-resolution consistency term at resolutions `(2, 3, 4, 6, 12)`.

The configured budget is 2,048 source minibatches per fold, batch size 4 (8,192 source windows); because each optimizer update aggregates gradients from the source-city tasks, the run records about 410 AdamW updates rather than 2,048 ordinary single-task updates. PCGrad is an existing optimizer component, not a novel SCPRC contribution. CAGrad, mean-gradient, and other optimizer results must remain identified by their actual run/configuration; they are not interchangeable with the PCGrad main lineage.

## 3. Optional transfer selection and deployment layers

Sections 3.1–3.2 describe optional post-forecast release mechanisms. They do not alter the definition of the SCPRC or its direct-model metrics. The candidate is evaluated unconditionally against Original PhaseFormer and eligible matched baselines; report selective/released forecasts in separate deployment tables. Any Exact fallback named by the release protocol is a deployment-layer reference and must not be substituted for the Original PhaseFormer main baseline.

### 3.1 Optional source-support screening and Exact fallback

The transfer protocol estimates relation/reliability thresholds and support ranges from source-domain data or source pseudo-target folds. A target's allowed history prefix supplies a diagnostic profile; the forecast-period target labels are reserved for scoring. A candidate can be rejected when the target profile falls outside the frozen source support, in which case the protocol falls back to the PhaseFormer Exact forecast defined by that release protocol. This fallback is separate from the Original PhaseFormer (no-PCGrad) main baseline.

This is an empirical selective-transfer rule. It is not a conformal risk bound, coverage guarantee, theorem of no negative transfer, or guarantee of per-city non-degradation. Always report `Exact`, `candidate`, the selection decision, and the released forecast separately.

### 3.2 Optional online controller: `SCPRC-Online(84,.25)`

The current selected release policy is causal online mixing over Exact and two support-shrunk SCPRC candidate experts (`p=1` and `p=8`). For support ratio `r`, source support radius `R`, and cap `c=0.25`,

```text
α_p = c · min(1, R / max(r, ε))^p
expert_p = Exact + α_p · (candidate − Exact),   p ∈ {1, 8}.
```

It warms up on Exact, updates weights only after a complete 84-origin block's 24-hour labels become observable, uses normalized RMSE/MAE feedback, maintains Exact weight ≥0.25, removes an alternative expert if either block metric is worse than Exact, and invokes the frozen two-block fallback/reset rule. The complete configuration is in [`ONLINE_RELEASE_PROTOCOL.md`](ONLINE_RELEASE_PROTOCOL.md) and its freeze manifest.

This controller is one optional deployment realization. It can be omitted when evaluating or deploying the offline candidate directly. Its performance must not replace candidate-model scores when attributing the relation algorithm, and a good Online aggregate does not by itself prove the adapter's operator is superior to generic relation attention.

## 4. Algorithm flow

```mermaid
flowchart TD
  A[168 h station histories] --> B[RevIN and 7 × 24 phase representation]
  B --> C[Phase-conditioned positive-kernel station relation]
  C --> D[Centered, reliability/source-constrained residual correction]
  D --> E[PhaseFormer embedding, routing, and 24 h forecast]
  E --> F[SCPRC candidate]
  G[Multi-source city losses] --> H[PCGrad aggregation + AdamW]
  H --> C
  F --> I{Optional frozen source-support / release layer}
  J[PhaseFormer Exact forecast] --> I
  I --> K[Selected released forecast]
  K --> L[Optional Online(84,.25) delayed-feedback update]
```

## 5. Evidence-to-claim mapping

- **Primary offline accuracy:** lead with Original PhaseFormer (no-PCGrad) versus SCPRC on the frozen CHARGED six-city LOCO matrix. Use Exact-PCGrad only as a mechanism control; report other eligible matched baselines separately and each external dataset on its own scale.
- **Relation adapter helps the candidate in the present CHARGED study:** use the high-budget matched structure ablation, with the aggregation statistic clearly specified.
- **Phase conditioning or the chosen relation operator is better than alternatives:** use the daily/phase structural ablation and parameter-matched phase-wise attention/DeepSets controls. On the aligned retrospective matrix SCPRC is modestly lower-error than these tested controls, but this does not establish universal operator superiority.
- **Source-support selection affects release/coverage:** use nested source-only release and risk-coverage/fallback analyses, not candidate-only accuracy.
- **Optional release behavior:** use the separate current 27-case `SCPRC-Online(84,.25)` replay table, label it retrospective, and report its grouped estimands. The older 26-case replay is retained only for provenance. Do not use either release result as the primary offline accuracy claim.
- **Optimizer novelty:** do not claim it. The available PCGrad/CAGrad/mean controls characterize training choices and attribution only.

See [`05_evidence_inventory.md`](SOURCE_ARTIFACT_INDEX.md#src-ed0b66a290e238cd) for exact source paths, metrics, and protocol incompatibilities.

## 6. Claim boundaries

- The local evidence supports an application-specific phase-conditioned relation adapter as the most plausible algorithmic contribution; it does not establish that every part of the operator is novel in the literature.
- An earlier attention summary appeared better because it used a different SCPRC candidate matrix. On the aligned structure matrix SCPRC has modestly lower equal-fold RMSE/MAE than the tested attention and DeepSets adapters. Report both the matrix lineage and narrow comparison; do not claim general superiority. The same-device profile also shows SCPRC forward latency is higher than these controls.
- Certificate variants have very similar candidate accuracy; their role is selection/release, not a demonstrated source of the candidate's accuracy gain.
- Station-input missingness experiments show material brittleness. The support score is a distribution-support diagnostic, not a missing-sensor detector.
- All currently available results are retrospective. None is a blind or prospective confirmation.
