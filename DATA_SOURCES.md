# Data sources

Download the datasets from the sources below and keep them outside the repository. The tables record the files used in the experiments, their versions, and their expected hashes. Dataset files, processed panels, labels, and raw predictions are not distributed here.

## Required sources

| Dataset | Official source | Required input | Version / verification |
|---|---|---|---|
| CHARGED | [Zenodo v3](https://doi.org/10.5281/zenodo.15814263) | `Hourly.zip`, six standard city folders | v3; MD5 `099a592653ca74fa1f1051e9c23d6456`; `remove_zero=False` |
| UrbanEV | [Official GitHub data](https://github.com/IntelligentSystemsLab/UrbanEV/tree/main/data) | The GitHub hourly `volume.csv`; other features only where an experiment explicitly requires them | Expected volume SHA-256 `a55a095ce75af33c59aece2643d5d71b5cd5a0dc73bb97bc553f0a48f40ace32` |
| Jiaxing | [Figshare v3](https://doi.org/10.6084/m9.figshare.28182251.v3) | `Dataset.zip`, containing the transaction data | MD5 `d2f43f19a3cfd5d364f97225e5f5d3ab` |
| Shanghai MP-EVData | [Figshare v3](https://doi.org/10.6084/m9.figshare.29882366.v3) | `station-level load Profile 1h.xlsx` | MD5 `ef44311080071c9a2473c41e53043346` |

CHARGED direct file: [Hourly.zip](https://zenodo.org/api/records/15814263/files/Hourly.zip/content).

UrbanEV can also be obtained with the [official GitHub repository ZIP](https://github.com/IntelligentSystemsLab/UrbanEV/archive/refs/heads/main.zip), then use its `data/` directory. The `main` URL can change. Match the downloaded files against the hashes in `manifests/DATA_REFERENCES.csv`.

## Optional UrbanEV data at other resolutions

The [official cloud folder](https://drive.google.com/drive/folders/1VUgdb8uNgmtvO93BHBK_OrSxjndrF-48?usp=sharing) provides:

- [UrbanEVDataset.zip](https://drive.google.com/file/d/1OEpo-XDocd33aK3MbgDt9S5bcqwiAFPQ/view): original/processed station data and 5-minute/hourly zone data. SHA-256 `9d3f0aec34434546d082509efcdeeaea116eaa841701cc5854a9b62a27881a79`.
- [UrbanEVSupplemental.zip](https://drive.google.com/file/d/1ojGWoQ13w2o59DjEAuUk2-CPzG08NRIT/view): weather, POI and geographic boundaries. SHA-256 `621992ceb9d6e0d4d01e3dde331853515a0f5e678dddc30a1c5b39392fe18afd`.

These optional archives are not required for the current volume-only UrbanEV experiment. Do not substitute their hourly volume for the GitHub file: some values differ. The [Dryad record](https://doi.org/10.5061/dryad.np5hqc04z) is another official access route with historical versions; it is not a guarantee of byte identity to the current GitHub file.

## Preparation and protocol identity

- CHARGED: six cities AMS/JHB/LOA/MEL/SPO/SZH; the primary LOCO model trains on April–July 2023 source-city labels and evaluates August target-city labels. Keep scaler, station order and time boundaries as recorded by each experiment. The evaluation is retrospective.
- UrbanEV: GitHub hourly zone data, 275 zones and 4,344 hours, September 2022–February 2023. `volume.csv` and `volume-11kW.csv` are different target definitions. The latter is not a replacement for the primary target.
- Jiaxing: build the hourly panel from transaction energy apportioned by overlap time. Keep `prepare_jiaxing_scprc_external.py` and the frozen construction settings; expected panel SHA-256 is recorded in the reference inventory. The hourly panel is not shipped.
- Shanghai: keep the existing panel-construction logic and frozen configuration. Use A1–A6 and A9 as specified; exclude swapping-count sheets A7/A8 and incomplete A10 for this protocol. Native target is hourly mean power in kW. The prepared history panel is not shipped.

Pass the external data location to the relevant preprocessing or experiment entry point. Some evaluation scripts also require recorded prediction arrays that are not distributed. Their availability is described in [release contents](docs/MIGRATION_SCOPE_ZH.md).

## Attribution and checks

CHARGED, Jiaxing and Shanghai retain their CC BY 4.0 dataset attribution. UrbanEV declares CC0 in its [official README](https://github.com/IntelligentSystemsLab/UrbanEV). Cite each original dataset record; the project code license does not replace dataset licenses.

`manifests/DATA_REFERENCES.csv` records source files and expected hashes rather than observations. The hashes identify the files used in the experiments; availability of the linked upstream files may change.
