# Supplementary design controls: methods and limits

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](../../MIGRATION_SCOPE_ZH.md)。


This is the 2026-11-17 supplementary experiment, not a new primary baseline
definition. The main comparison remains **Original PhaseFormer versus Full
SCPRC**. **Exact-PCGrad is a mechanism/attribution control** for the matched
training procedure. The new `full_same_run` reference has 18 same-batch folds;
it does not automatically replace the historical canonical Full results.
Any canonical replacement requires a separate documented reconciliation.

## Architecture and what each intervention identifies

All four arms use the canonical Full constructor through
`model_variants.build_model`. History length is 168 hours and forecast length
24 hours. The history is rearranged into 24 **relative-window phases**, each
with seven cycles: `[batch, stations, 24, 7]`. These are positions within the
observed window, not supplied absolute calendar or geographic coordinates.
No station coordinates or future target labels enter these adapters. Historical
target values provide the input; normalization, centering and reliability
statistics can depend on that observed history. Source scaler and source q90
threshold are fixed from the five source cities, not refitted to target labels.

- `full_same_run` preserves the exact canonical Full mechanism, including its
  phase-specific normalized Q/K relation, correction, deterministic gates,
  raw-space recentering, additive fusion and PhaseFormer backbone. Its purpose
  is a contemporaneous reference for the three new design controls.
- `phase_shared_strict` retains the same Q/K representation and registered
  parameters. For source phase s, define
  `A_s(i,j) = q_is^T k_js / max(sum_j q_is^T k_js, 1e-6)`. The shared operator
  is `A_bar = (1/24) sum_s A_s`, applied to each value phase. The implementation
  averages **normalized A**, not pooled unnormalized kernels or averaged Q/K.
  It removes variation of the applied station operator across value phases,
  while retaining phase-specific features in Q/K. A sum of rank-r operators
  can have higher effective operator rank than a single rank-r operator;
  operator-rank differences may remain. This is not a claim of matching every
  effective rank or representational property.
- `stationwise_matched` replaces the learned cross-station relation with a
  shared stationwise `7 -> 60 -> 7` MLP: **907 registered trainable relation
  parameters versus 903** in Full. It has no learned cross-station aggregation.
  Deterministic cross-station centering, common statistics, source-envelope
  attenuation and other outer processing remain. It is not an independently
  processed single-station model. Here `R = V = Z - mean_stations(Z)`; therefore
  `Z - R` is identical for every station. The peer disagreement is common and
  the peer attenuation is mathematically **one** (the implementation stabilizes
  the calculation). Biases canceled by centering still count as registered
  parameters; registration does not imply nonzero effective gradients.
- `concat_project` adds a trainable `Linear(14,7)` to the seven-cycle axis of
  `[X, Delta]`, initialized with weight **[I, I]** and zero bias. This contributes
  **105 additional registered parameters** (98 weights and 7 biases). Both
  `W_X` and `W_Delta` remain trainable, so it is **not strictly the same capacity**
  as addition. Although the relation correction Delta is gated and recentered,
  the learned direct path `(W_X - I)X + b` can change independently. The gate
  cannot bound the whole direct path or the entire fused update.

Centering constrains the station sum of the **input correction** at its specific
location in the computation, subject to floating-point error. The downstream
embedding, nonlinear routing, predictor and inverse normalization do not imply
forecast energy conservation. Neither centering nor reliability gates guarantee
conservation, calibrated uncertainty, physical feasibility or dispatch safety.

### Existing diagnostics: sample scope and units

The runner's existing [diagnostics.json](../../SOURCE_ARTIFACT_INDEX.md#src-f2ce1694116793e5) uses only the seven fixed evaluation
origin indices **[0, 24, 168, 336, 504, 672, 720]**. Those summaries describe
that diagnostic sample, not all 721 evaluation origins, the first 16 efficiency
histories, or a population-wide correction distribution. No diagnostic outcome
selects the efficiency origins or changes which folds are included.

Model input is source-standardized: `x = (history_native - fold_mean) / fold_std`.
In diagnostics, names such as `raw_correction_before_attenuation`,
`raw_correction_after_attenuation` and `raw_correction_after_last_centering`
refer to corrections **after undoing the within-window RevIN scale, but still
in source-standardized units**. They are not native kWh solely because their
names say `raw`. Multiply a correction (or its absolute magnitude) by the saved
positive **fold_std** to express it in native units; do not add fold_mean to
a correction/difference. Native forecast values instead use
`prediction_native = prediction_source_standardized * fold_std + fold_mean`.
Dimensionless attenuation/gate summaries are not rescaled.

## Population, statistical interpretation and application limits

The intended population is all four arms x six cities (AMS, JHB, LOA, MEL, SPO,
SZH) x three seeds (13, 17, 29): 72 runs, 18 city-seed folds per arm. All 18
folds must be retained; there is no favorable-city, seed, metric, clipping or
origin selection. Training uses the canonical FP32 PCGrad procedure with 2,048
source minibatches of four windows (8,192 exposures; 410 optimizer updates).
This measurement script never invokes that procedure.

Evaluation is retrospective. Seeds are nested within six cities, not 18
independent city replications. Shared source data and overlapping hourly
evaluation windows further limit independence. Error summaries equally weight
the 18 fold metrics. Raw predictions are primary; nonnegative-clipped results
are separately named sensitivities, never substituted after seeing outcomes.
The full endpoint definitions and paired resampling are in [ANALYSIS_RULES.md](ANALYSIS_RULES.md).
City-cluster intervals retaining all three seeds and paired noncircular
168-origin moving-block intervals are **exploratory pointwise** sensitivities,
not multiplicity-controlled confirmatory intervals or guarantees for fresh
cities/retraining. No independent-city interpretation follows from seed count.

City aggregate peaks, ramps and 24-hour totals are forecast proxies. There is
no real tariff, site-level dispatch optimization, battery/charger feasibility
model or field intervention here. These errors do not establish realized
electricity savings, revenue, operational dispatch value or causal effects.

## Frozen efficiency protocol

[measure_frozen_efficiency.py](../../../experiments/design_controls/measure_frozen_efficiency.py) is read-only with respect to training artifacts.
Its default `--root` is the script directory (HERE), independent of the working
directory. Only completed full-budget artifacts under
`ROOT/runs/target_<city>_seed<seed>/<variant>/` are eligible. Smoke/benchmark
directories are never searched. The default requires all 72 eligible runs
before importing torch or doing CUDA work. Incomplete snapshots still produce
the full case inventory and an incomplete manifest, returning exit code 2.
`--allow-partial` explicitly enables monitoring of available completed runs;
it cannot turn fewer than 216 successful cases into a complete result.

The fixed design is **4 models x 18 folds x batches 1/4/16 = 216 cases**. For
each fold, take the first 16 ordered evaluation origins from [fold_data.npz](../../SOURCE_ARTIFACT_INDEX.md#src-c3250c3cf66d649b)
`label_starts`, normally 2928 through 2943. Origin s uses exactly
`ROOT/data_cache/CITY_values.npy[s-168:s]`, with station order checked against
fold and cache metadata. Convert to FP32, then use `(history - mean) / std`
with the saved [fold_protocol.json](../../SOURCE_ARTIFACT_INDEX.md#src-2b917267602c0fcd) **source scaler**. No new scaler, threshold,
random input, target label, padding or duplicated history is used. Fold truth
is not read. If fewer than 16 origins exist, record `insufficient_cases`; do
not fabricate a batch of 16. Normal evaluation has 721 origins; this benchmark
uses a fixed prefix for cost measurement, not an accuracy-estimation sample.

Load the existing local [model.pt](../../SOURCE_ARTIFACT_INDEX.md#src-375c2b8a64c9a598) onto CPU, reconstruct through
`build_model(variant, n_stations, saved_threshold, seed)`, and call
`load_state_dict(..., strict=True)`. Check model identity, fold protocol,
source hashes and any supplied artifact digests. The local checkpoints contain
NumPy RNG objects, so `torch.load(weights_only=False)` is required; only trusted
local runner checkpoints are appropriate. Saved optimizer state is discarded,
never instantiated or stepped. No optimizer, backward pass, retraining, resume,
target fitting, compilation or hyperparameter search occurs.

All forwards use `eval()`, `torch.inference_mode()`, FP32 inputs/weights and
disabled autocast. TF32 is disabled. The same process/device/thread settings
serve all arms. For CUDA, recorded Python, torch, NumPy, CUDA, GPU name and
recorded numerical flags must match each training environment; mismatches are
failures, not automatic migrations or CPU fallback. Additional runtime and
backend settings are included in the manifest. CPU mode is explicitly labeled
and does not provide GPU latency or CUDA memory numbers.

Before timing, predict the first four histories as one batch of four, reproduce
the runner's native-unit inverse scaling, and compare to
`prediction_raw.npy[:4]` with **rtol=1e-5, atol=1e-3**:
`abs(new - saved) <= atol + rtol * abs(saved)`. This is raw native prediction
validation, with no clipping or rounding. The CSV records pass/fail, max
absolute error, maximum relative error over nonzero references, tolerance
violation counts, maximum absolute tolerance excess, and maximum excess rtol
after allowing atol:
`max(max((abs_error-atol),0)/abs(saved) - rtol, 0)`. Zero-reference mismatches
have their own count; undefined relative maxima are empty rather than silently
set to zero. A failed check records `validation_failed` for all three batches;
it does not start timing, loosen tolerances, repair the checkpoint or train.

For each batch size, one sweep visits all 16 histories in contiguous batches:
16 forwards for batch 1, four for batch 4, and one for batch 16. Default
**warmup=3 sweeps and reps=10 measured sweeps**, editable via CLI. This means
3/10 repetitions of the same fixed workload at every batch size. Histories
and weights are already resident on the selected device. Data/checkpoint
loading, hashing, normalization, transfers, native validation and warmup are
outside the forward clocks. No end-to-end loading or serving claim is made.
Both CSVs and the manifest label this scope
`pure_forward_data_transfer_excluded`.

The CPU `perf_counter_ns` wall clock and GPU CUDA-event elapsed times are
separate fields. On CUDA, synchronize before each timed forward, record events
around the forward on the current stream, and synchronize after the ending
event. The host wall interval includes dispatch, event-record and completion
synchronization overhead; it is not a CPU-only compute duration. CUDA-event
latency is the stream interval, not total host latency. Output checks/copies
and deletion occur outside the forward clock. Per-sweep totals are retained;
reported per-batch means/medians/sample standard deviations are statistics of
the ten sweep totals divided by forwards per sweep. Per-origin latency divides
by 16; throughput is 16 divided by the mean sweep duration. Throughput excludes
checks and between-forward setup, so it is forward throughput, not service QPS.

After warmup, record allocated/reserved CUDA memory baselines and reset peak
statistics. Record allocated/reserved peaks over the measured phase and their
increments over baseline. The baseline includes the model, all 16 resident
histories and allocator cache left by validation/warmup. Peaks also include
untimed output checks; incremental reserved memory can legitimately be zero.
These are PyTorch process allocator measurements, not whole-device usage,
training memory or an isolated activation-only footprint. Registered total,
trainable, backbone, relation and fusion parameter counts are measured from
the reconstructed model, not inferred from names or checkpoint file size.

Results are valid only for the recorded hardware/software, FP32 implementation,
station counts, input prefix and synchronized protocol. Model order is fixed,
not randomized; thermal state, other processes and operating-system scheduling
can affect timing. Run formal measurements after training and competing GPU
jobs finish. No concurrent-load adjustment or speedup extrapolation is made.
The ten repeated sweeps are technical repeats, not independent city evidence.

## Outputs and usage

Default outputs are [ROOT/efficiency/per_fold_efficiency.csv](../../SOURCE_ARTIFACT_INDEX.md#src-1a3fc016e6ee82f5), [summary.csv](../../SOURCE_ARTIFACT_INDEX.md#src-6aa8b1a14d476f34)
and [manifest.json](../../SOURCE_ARTIFACT_INDEX.md#src-070716086c420660). The per-fold file always has 216 planned rows, including
missing, invalid, validation-failed, insufficient or blocked cases. The summary
has 12 model/batch rows and reports equal-fold means, counts and completeness.
Partial means are labeled monitoring and never presented as 18-fold estimates.
The manifest records settings, environments, fixed selection, failed cases,
SHA-256 input/code/output hashes and whether all 216 cases succeeded. Inputs
changing during measurement invalidate the snapshot. CSV/JSON writes use
atomic replacement and do not alter matrices or training artifacts.

```bash
# Safe now: standard-library help; no torch import, no GPU use.
python experiments/design_controls/measure_frozen_efficiency.py --help

# Only after all training has finished and the GPU is idle:
python experiments/design_controls/measure_frozen_efficiency.py

# Same formal matrix, explicitly editable technical repeats:
python experiments/design_controls/measure_frozen_efficiency.py \
  --root experiments/design_controls \
  --device cuda:0 --warmup 3 --reps 10

# Explicit CPU monitoring; preferably use a separate output directory.
CUDA_VISIBLE_DEVICES='' python experiments/design_controls/measure_frozen_efficiency.py \
  --device cpu --allow-partial --out /tmp/scprc_cpu_efficiency
```

Exit 0 means all 216 cases were measured successfully; exit 2 denotes an
incomplete or invalid snapshot, including explicit partial monitoring. Synthetic
CPU fixtures verify mechanics only and must never enter paper result tables.
