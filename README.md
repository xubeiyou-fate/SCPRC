# SCPRC

Cross-city EV charging demand forecasting.

[简体中文](README.zh-CN.md) · [Experiments](docs/EXPERIMENT_MATRIX.md) · [Checkpoints](docs/WEIGHTS.md) · [Data](DATA_SOURCES.md)

**Paper status:** a manuscript describing this work is in preparation for journal submission. The paper link and citation will be added when available.

SCPRC combines a PhaseFormer forecasting backbone, PCGrad-based multi-source training, and a phase-conditioned station-relation adapter. The adapter uses station scale, peer reliability, and source-derived disagreement to regulate its residual corrections.

The repository contains the model, training and evaluation code, experiment protocols, and result tables. Trained checkpoints are available in [Releases](https://github.com/xubeiyou-fate/SCPRC/releases/tag/paper-v1). Datasets are downloaded separately from their original sources.

## Installation

The inference code has been checked with Python 3.12, PyTorch 2.7.1, NumPy 1.26.4, and pandas 2.2.2.

```bash
git clone https://github.com/xubeiyou-fate/SCPRC.git
cd SCPRC
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For a CUDA installation and the recorded environment, see [Environment](environment/README.md).

## Data

The main experiments use the CHARGED hourly data for AMS, JHB, LOA, MEL, SPO, and SZH. External evaluations use Jiaxing, Shanghai MP-EVData, and UrbanEV. Download links, file hashes, units, and preprocessing details are listed in [Data sources](DATA_SOURCES.md). Store the data outside this repository.

The UrbanEV experiment uses the official GitHub hourly `volume.csv`. Its values differ from the file in the cloud archive; use the recorded hash to select the matching version.

## Checkpoints

Download the 14 ZIP files and `SHA256SUMS` from [paper-v1](https://github.com/xubeiyou-fate/SCPRC/releases/tag/paper-v1) into `release_assets/`. With GitHub CLI:

```bash
mkdir -p release_assets
gh release download paper-v1 --repo xubeiyou-fate/SCPRC \
  --pattern '*.zip' --pattern SHA256SUMS --dir release_assets
(cd release_assets && sha256sum -c SHA256SUMS)
python scripts/restore_checkpoints.py
python scripts/verify_checkpoints.py
```

The release restores 534 checkpoint paths, including six aliases of identical weights. Model, city, seed, horizon, and file hashes are listed in [CHECKPOINTS.csv](manifests/CHECKPOINTS.csv). See [Checkpoint guide](docs/WEIGHTS.md) for individual experiment groups.

## Inference

The following example uses the AMS, seed 13, 24-hour SCPRC checkpoint:

```bash
python scripts/infer.py \
  --checkpoint checkpoints/forecasting/h24/scprc/AMS/seed_13/model.pt \
  --input /path/to/history.npy \
  --output /path/to/forecast.npy \
  --device cpu
```

Input is an array in the original data units with shape `[batch, 168, stations]`. Output has shape `[batch, 24, stations]`, or `[batch, 48, stations]` for a 48-hour checkpoint. Keep input and output arrays outside the repository. The checkpoint's source-domain normalization is applied automatically. Raw predictions are returned by default; nonnegative clipping is an explicit option.

The two-variable DLinear checkpoints also require `--duration-input`. Run `python scripts/infer.py --help` for the available options.

## Experiments

The main CHARGED protocol uses leave-one-city-out evaluation, seeds 13, 17, and 29, 168-hour inputs, and 24-hour forecasts. Training uses April–July 2023 source-city data; evaluation uses August 2023 target-city data.

| Comparison | Purpose |
|---|---|
| Original PhaseFormer vs. SCPRC | Main framework comparison |
| Original, Exact-PCGrad, Full-no-PCGrad, SCPRC | Training and adapter controls |
| Daily/phase relations, component controls, Attention, DeepSets | Relation design and component analysis |
| Prediction intervals, input perturbations, 48-hour forecasts | Additional forecasting evaluations |
| Peak, ramp, total energy, timing, and capacity scenarios | Energy-related forecast evaluation |
| Jiaxing, Shanghai, and UrbanEV | Separate external-dataset evaluations |

[Experiment overview](docs/EXPERIMENT_MATRIX.md) · [Model and protocol](docs/MODEL_AND_PROTOCOL.md) · [Main matrix](docs/MAIN_EXPERIMENT_MATRIX.md) · [Energy experiments](docs/ENERGY_EXPERIMENT_MATRIX.md)

### Interpretation

These evaluations are retrospective. August results informed parts of the adapter design, and inspected September results influenced subsequent research directions. The reported periods are therefore not an untouched final test.

Performance varies by city and metric. Some simpler adapters match or improve individual metrics, and the Shanghai daily-mean baseline outperforms the reported learned models on that dataset. External results use different targets and units and are reported separately.

Some mean-gradient controls lack a recorded common initialization with their PCGrad counterparts, which limits optimizer attribution. [Initialization and comparison limits](docs/SOURCE_LIMITATIONS_ZH.md) describes this distinction and the separately recorded paired design controls. Capacity results are reservation-risk scenarios; they do not measure electricity-bill savings or deployed grid operation.

## Repository structure

```text
scprc/          Models, training components, data adapters, and baseline dependencies
experiments/    Training and evaluation entry points
results/        Result tables, protocols, and run metadata
docs/           Methods and experiment documentation
scripts/        Inference, checkpoint restoration, and verification
manifests/      Checkpoint identities, file hashes, and provenance
environment/    Recorded environment and installation notes
```

## Citation and licenses

Until the paper is available, cite this repository and the release or commit used. PhaseFormer, PCGrad, and the other baselines retain their original attribution. Component licenses are listed in [Third-party notices](THIRD_PARTY_NOTICES.md); the reuse terms of the pinned PhaseFormer snapshot remain unverified.
