# Experiments

The main comparison is **Original PhaseFormer (mean-gradient training) versus SCPRC**. Exact-PCGrad uses the same backbone with PCGrad and no relation adapter. Full-no-PCGrad retains the adapter and uses mean-gradient training. Together, these models provide the framework and training controls.

The CHARGED protocol uses six leave-one-city-out splits and seeds 13, 17, and 29, giving 18 city-by-seed folds. The main task predicts 24 hours from 168 hours of history.

| Experiment | Models or measurements | Documentation |
|---|---|---|
| Main forecasting results | Original PhaseFormer, SCPRC, and forecasting baselines | [Main matrix](MAIN_EXPERIMENT_MATRIX.md), [results table](figures_tables/table_strong_baselines.md) |
| Training and adapter controls | Original, Exact-PCGrad, Full-no-PCGrad, SCPRC | [Model and protocol](MODEL_AND_PROTOCOL.md) |
| Component analysis | Independent Scale/Peer/Source switches and separate Full-drop controls | [Energy experiment matrix](ENERGY_EXPERIMENT_MATRIX.md) |
| Relation operators | SCPRC, Attention, DeepSets | [Operator results](figures_tables/table_relation_operators.md) |
| Paired design controls | Phase sharing, station-wise adaptation, and concatenation versus residual integration | [Methods](experiments/design_controls/METHODS_AND_LIMITS.md), [results](experiments/design_controls/RESULTS_AND_CONCLUSIONS_ZH.md) |
| Prediction intervals | Empirical coverage, interval width, interval score, WIS | [Main matrix](MAIN_EXPERIMENT_MATRIX.md) |
| Robustness and horizon | Six input perturbations; separately trained 48-hour models | [Main matrix](MAIN_EXPERIMENT_MATRIX.md) |
| Energy-related metrics | Peak, ramp, total energy, peak timing, and capacity-reservation scenarios | [Energy matrix](ENERGY_EXPERIMENT_MATRIX.md), [results](experiments/energy_value/RESULTS_AND_CONCLUSIONS.md) |
| Efficiency | Parameter count and recorded forward-inference measurements | [Main matrix](MAIN_EXPERIMENT_MATRIX.md) |
| External datasets | Jiaxing, Shanghai MP-EVData, UrbanEV | [External protocols](EXTERNAL_DATA_PROTOCOL.md), [results](mandatory_supplement/external_baselines/summary.md) |

## Reading the results

Independent component switches and Full-drop ablations answer different questions. The former add individual mechanisms to a reference model; the latter remove a mechanism from the complete adapter. Their SCPRC reference runs are identified separately. The paired design controls use a further same-batch SCPRC reference.

The evaluations are retrospective. Some mean-gradient runs lack a recorded common initialization with PCGrad runs; the paired design controls have separate initialization and sampling records. See [comparison limits](SOURCE_LIMITATIONS_ZH.md).

External results use different targets, units, and protocols. The external v2.2 Exact/SCPRC results are available as summaries, but their 18 checkpoints are not distributed. External baseline checkpoints are listed individually in the checkpoint manifest. The fixed-distance Graph WaveNet diagnostic has unresolved run-quality questions and is excluded from the accepted main comparison.

[Result files](RESULTS_INDEX.md) · [Available materials](MIGRATION_SCOPE_ZH.md) · [Source records](SOURCE_ARTIFACT_INDEX.md)
