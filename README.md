# SCPRC: multi-city EV charging demand forecasting

> **论文状态：正在准备期刊投稿。** 本仓库为 SCPRC 的投稿前研究版本，公开方法实现、实验协议、结果摘要和模型检查点，支持研究核验与复现。论文链接与正式引用信息将在可公开时补充。引用本研究时，请注明本仓库及所用 release／commit。
>
> **Paper status: manuscript in preparation for journal submission.** This pre-submission research release documents the SCPRC implementation, experimental protocols, results, and checkpoints for verification and reproducibility. A paper link and formal citation will be added when available. Please identify this repository and the release or commit used when referencing this work.

本仓库提供 SCPRC 当前论文实验的源码、最终检查点索引和复现材料。**不包含数据集；数据从官方来源下载。** 维护者的发布步骤见 [UPLOAD_GUIDE_ZH.md](UPLOAD_GUIDE_ZH.md)。

SCPRC 将 PhaseFormer 骨干、多源城市 PCGrad 训练和相位条件站点关系适配结合。适配器包括 Scale、Peer 和 Source 衰减机制。PCGrad 是既有训练方法；本地基线和适配器的具体调整、协议与限制见实验矩阵。

## 从这里开始

| 内容 | 入口 |
|---|---|
| 当前实验矩阵与发布范围 | [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md)、[docs/MIGRATION_SCOPE_ZH.md](docs/MIGRATION_SCOPE_ZH.md) |
| 官方数据下载链接、版本、准备规则 | [DATA_SOURCES.md](DATA_SOURCES.md) |
| 安装环境 | [environment/README.md](environment/README.md)、[requirements.txt](requirements.txt) |
| 最终权重索引 | [manifests/CHECKPOINTS.csv](manifests/CHECKPOINTS.csv) |
| 权重分包 | [下载与恢复说明](docs/WEIGHTS.md)；本地 `release_assets/` |
| 整体迁移状态 | [manifests/MIGRATION_COMPLETION.json](manifests/MIGRATION_COMPLETION.json) |
| 第三方代码版本、许可 | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |
| 发布命名与路径调整 | [docs/NAMING_REVIEW_ZH.md](docs/NAMING_REVIEW_ZH.md) |

## 模型与实验身份

主比较为 **Original PhaseFormer（MG）vs SCPRC**。Exact-PCGrad 是机制控制。主实验为六城 AMS/JHB/LOA/MEL/SPO/SZH、seeds 13/17/29、168小时输入、24小时输出。h=48 使用单独训练的输出头。

本次迁移包括当前基线、关系/独立开关/严格删除消融、设计控制配对实验和相应最终权重。SCPRC主实验、独立开关参照、设计控制配对参照与h=48版本 保留独立身份。不能按同名文件替换。

按“不上传历史模型”的范围，外部旧谱系 legacy Exact/SCPRC 权重未迁入；外部表可查的既有结果与配置保留，但不声称完整交付这些旧模型的冻结推理。外部新基线包含在当前发布候选中。R10固定距离Graph WaveNet仍有结果来源/质量审查事项，不包含在已验收主模型包；不宣称有Adaptive GWN或MTGNN正式结果。

所有结果是回顾性评估。8月结果曾参与部分关系模块设计，9月已查看的结果影响过后续研究方向；不能将这些月份重新称为项目级盲测。历史 no-PCGrad 运行的配对初始化记录不足也保留在限制说明中。SCPRC不在所有指标或外部数据集上优于简化模型；Shanghai日均基线的优势不删除。

## 数据准备

先按 [DATA_SOURCES.md](DATA_SOURCES.md) 下载到本仓库之外。仓库只保存来源、文件hash及构建代码，不保存ZIP、Excel、时序CSV、面板或标签。

UrbanEV当前论文使用官方GitHub小时版 `data/volume.csv`；云盘主包的同名文件有部分数值差异，不能直接替换。Jiaxing的小时标签是交易能量按时间重叠分配形成的代理，Shanghai目标是小时平均功率。不同数据集的原始RMSE不合并。

训练与评价入口位于 `experiments/`；CHARGED主实验支持显式 `--archive` 指向外部 `Hourly.zip`。部分旧评估入口还需要它们记录的冻结输入，本次不迁移这些数据。每个入口的具体可复现范围以清单和审计为准。

## 权重和推理

本地检查点在 `checkpoints/`，按实验角色、模型、城市和seed分组；它们被Git忽略，通过Releases分发。每个文件的来源、SHA-256、导出方式和包名见 `manifests/CHECKPOINTS.csv`。DLinear原混合容器导出时只保留对应模型和必要信息，转换核验单独记录。

查看便携入口与验证器的参数：

```bash
python scripts/infer.py --help
python scripts/check_runtime.py --help
python scripts/verify_checkpoints.py --help
python scripts/audit_publication.py
```

例如，用AMS/seed13的SCPRC主实验权重对仓库外输入推理：

```bash
python scripts/infer.py \
  --checkpoint checkpoints/forecasting/h24/scprc/AMS/seed_13/model.pt \
  --input /your-data/history.npy \
  --output /your-output/full_forecast.npy \
  --device cpu
```

输入为原始单位的 `[batch,168,stations]` 数组；输出为 `[batch,24,stations]`，h=48检查点输出48步。`/your-data` 和 `/your-output` 是需要替换的外部路径。主表两变量DLinear还要求 `--duration-input`，不能用仅有充电量的输入冒充原两变量协议。

推理输入、输出放在仓库外；`infer.py` 读取原始单位历史，按检查点记录使用源域scaler，再将预测还原到原始单位。默认保留raw预测，非负截断必须显式启用。预测形状和支持的检查点类型见入口帮助与runtime审计。

无需重新训练即可检查已有模型。原训练入口仍保留用于方法复现；不要把重新训练产生的差异覆盖到迁移包中的固定结果。

## 代码与权重分发

仓库入口为 [xubeiyou-fate/SCPRC](https://github.com/xubeiyou-fate/SCPRC)，主分支为 `main`。源码、配置、小型结果和清单进入普通Git；`checkpoints/`、`release_assets/`、`local_notes/` 和数据被 `.gitignore` 排除。

最终权重ZIP和校验文件通过同一仓库的 [Releases](https://github.com/xubeiyou-fate/SCPRC/releases) 分发。GitHub的“Download ZIP”不包含这些Releases附件。下载权重、校验及恢复方法见 [权重说明](docs/WEIGHTS.md)，维护者发布流程见 [上传说明](UPLOAD_GUIDE_ZH.md)。

各组件的来源与适用许可见 `THIRD_PARTY_NOTICES.md`。DLinear的固定版本Apache-2.0许可文本已补全；PhaseFormer固定快照未附许可文本的事项仍保留，公开可见不代表本仓库代上游授予复用或再分发许可。

## 发布目录

```text
scprc/                 SCPRC模型、训练组件、数据适配器和基线依赖
experiments/           主实验、消融、评价和基线入口
configs/               发布配置
checkpoints/           模型／城市／seed命名的本地权重
results/               指标表和实验结果
release_assets/        最终权重ZIP及校验文件
docs/                  模型说明、实验矩阵和边界
scripts/               推理、恢复与验证
manifests/             权重、元数据、来源和验收清单
```

模型公开名称统一为 **SCPRC**。PhaseFormer、PCGrad、PatchTST、TimeMixer等保留各自的方法归属。源服务器路径和原协议ID仅作为来源字段保存，不是发布目录或部署依赖。
