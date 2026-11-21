# SCPRC architecture

SCPRC combines a PhaseFormer forecasting backbone, PCGrad-based multi-source training, and a phase-conditioned station-relation adapter. The main comparison is Original PhaseFormer versus SCPRC. Exact-PCGrad provides a training control without the adapter. The separately recorded PhaseFormer Exact run is a different reference.

The CHARGED LOCO experiments train the adapter within the forecasting model. A [sidecar residual calibrator](SOURCE_ARTIFACT_INDEX.md#src-0a2f3ecd71446fed) is a separate implementation with separate results. The optional `SCPRC-Online(84,.25)` policy selects among forecasts after model inference and is evaluated separately from the core forecasting model.

## 1. Problem and notation

At forecast origin `t`, the model receives histories for a variable-size set of `N` charging stations, `X_t ∈ R^(B×168×N)`, and predicts the next 24 hours. The main CHARGED setting trains on several source cities and evaluates a held-out city. The target forecast-period labels do not train the relation adapter. Station identifiers, coordinates, and a fixed target-city adjacency matrix are not inputs to the relation operator.

The **SCPRC forecasting model** uses a PhaseFormer backbone and a phase-conditioned inter-station relation adapter trained on source-city data. PCGrad coordinates source-city gradients. The experiments evaluate the combined framework and its components. Forecast selection is a separate optional stage.

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

## 5. Experimental comparisons

The main forecasting comparison uses Original PhaseFormer and SCPRC on the CHARGED six-city LOCO protocol. Exact-PCGrad is the mechanism control. External datasets are evaluated on their own targets and scales.

The relation studies compare daily and phase-conditioned relations, phase-wise attention, DeepSets, independent component switches, and component removals. Results identify metric-specific differences; they do not establish that one operator is uniformly best. The adapter is also evaluated against phase-sharing, station-wise, and concatenation controls using a separately recorded paired reference.

Source-support screening is assessed with risk-coverage and fallback analyses. The optional online policy uses the 27-case `SCPRC-Online(84,.25)` replay and its own grouped metrics. These are forecast-selection evaluations, separate from the core model's point accuracy. The 26-case replay remains a separate source record. PCGrad/CAGrad/mean-gradient controls evaluate training choices; PCGrad itself is an existing method.

Detailed source identities are listed in the [experiment inventory](SOURCE_ARTIFACT_INDEX.md#src-ed0b66a290e238cd).

## 6. Interpretation

The experiments assess the phase-conditioned relation adapter within the SCPRC framework. Their scope is the recorded tasks and controls, rather than a claim that every operator component is new.

On the aligned structure matrix, SCPRC has modestly lower equal-fold RMSE/MAE than the tested attention and DeepSets adapters. Its same-device forward latency is higher. A summary using a different SCPRC reference is kept separate from that comparison.

Source-support variants have similar model accuracy and are interpreted through selection and coverage. Input-missingness tests show sensitivity to missing station histories; the support score measures distribution support rather than sensor failure.

All reported evaluations are retrospective.
