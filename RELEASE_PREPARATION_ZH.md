# 发布内容与文件校验

SCPRC的源码、配置、实验结果和文档保存在[主仓库](https://github.com/xubeiyou-fate/SCPRC)；模型权重通过[paper-v1](https://github.com/xubeiyou-fate/SCPRC/releases/tag/paper-v1)分发。数据下载方式见[数据来源](DATA_SOURCES.md)。

## 文件组成

- `scprc/`：模型、训练组件、数据适配器和基线依赖。
- `experiments/`：训练、对照实验与评估入口。
- `results/`、`docs/`：结果表、方法与实验说明。
- `manifests/`：权重清单、环境与文件哈希、运行来源。
- `scripts/`：推理、权重恢复和校验。

## 模型权重

14个ZIP保存528份唯一权重及六个别名，恢复后共534个检查点路径。主实验、独立开关、严格删除、配对设计、48小时和外部基线分别编号。各组内容见[权重说明](docs/WEIGHTS.md)。

检查点保留必要的归一化参数、阈值和实验身份。DLinear下载文件仅保留对应模型及推理信息；参数与固定输入的一致性记录见`manifests/CHECKPOINT_MIGRATION_AUDIT.json`。

## 校验

下载权重后，在仓库根目录执行：

```bash
(cd release_assets && sha256sum -c SHA256SUMS)
python scripts/restore_checkpoints.py
python scripts/verify_checkpoints.py
python scripts/verify_file_manifest.py
```

CPU合成输入检查覆盖534个检查点和38类模型、数据集、时域设置。该检查验证加载和推理执行的一致性，不能代替真实数据上的指标复算。结果范围见[实验说明](docs/EXPERIMENT_MATRIX.md)，完整环境见[环境说明](environment/README.md)。

## 数据与许可

数据集、原始标签、逐样本预测、校准残差和原始扰动掩码不包含在Git仓库或权重ZIP中。必要的外部产物及其来源列于[文件范围说明](docs/MIGRATION_SCOPE_ZH.md)。

第三方代码及数据集保留各自的许可和署名，见[第三方说明](THIRD_PARTY_NOTICES.md)。仓库公开可见并不改变上游许可。
