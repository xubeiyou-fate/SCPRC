# 目录与文件说明

源码按用途组织，检查点按实验、模型、城市和seed组织。实验协议ID、文件哈希和运行身份用于对应代码、权重与结果。

| 目录 | 内容 |
|---|---|
| `scprc/` | SCPRC模型、训练组件、数据适配器和基线依赖 |
| `experiments/` | 主实验、对照实验和评价入口 |
| `results/` | 指标表、协议和运行元数据 |
| `docs/` | 方法、数据和实验说明 |
| `scripts/` | 推理、权重恢复及文件校验 |
| `manifests/` | 检查点、来源和SHA-256索引 |

SCPRC的主实现为`scprc/models/scprc.py`。主比较使用Original PhaseFormer与SCPRC；Exact-PCGrad为机制对照。主实验、独立开关、严格删除、配对设计、48小时与外部实验分别保留运行身份。

`manifests/CHECKPOINTS.csv`记录534个检查点路径及其来源；`manifests/RELEASE_ASSETS.csv`记录14个权重ZIP。文件中的源服务器路径和检查点内部键用于追溯，不是用户机器上的读取路径。

文件对应关系见[命名与路径记录](../manifests/NAMING_AUDIT.json)。[源文件核对](../manifests/ORIGINAL_SOURCE_INTEGRITY.json)、[检查点执行检查](../manifests/RUNTIME_AUDIT.json)和[结果数值核对](../manifests/RESULT_VALUE_PARITY.json)分别记录各自检查范围。
