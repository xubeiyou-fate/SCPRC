# SCPRC 发布命名核对

本次调整范围仅为 `/data/TimeMixer/git` 发布副本。服务器的 `paper2` 原始实验、数据和权重来源保留。公开路径以方法和实验角色命名；源文件名、运行ID、协议ID和checkpoint内部兼容键保留在溯源字段中，以便核对实验身份。

## 发现并修正的命名

| 原命名问题 | 发布库中的入口 |
|---|---|
| 用 `paper2` 作为项目包及实验目录 | `experiments/`、`docs/`、`results/` 按用途组织 |
| 模型依赖分散在 `reproduction/` 与 `research/` | 统一到 `scprc/`，训练、模型、数据适配器和基线依赖各自分组 |
| 主方法文件与类使用实现阶段名称 | `scprc/models/scprc.py`，主类 `SCPRC` |
| 文件名和目录名含开发日期、`rebuild`、`legacy`、`scpcr` 等 | 用 `forecasting`、`evaluation`、`design_controls`、`gate_removal` 等实验角色命名 |
| 权重文件携带工作目录和任务批次名称 | `checkpoints/<实验角色>/<模型>/<城市>/seed_<seed>/model.pt`，h24/h48单独分组 |
| 权重ZIP使用工作批次名称 | `release_assets/scprc_h24_and_controls.zip` 等按实验组命名，共14包 |
| 文档目录树、命令示例和链接仍指向改名前路径 | 更新README、上传说明、矩阵链接与命令路径，并重新核对本地链接 |

## 方法名称与实验身份

完整方法统一称 **SCPRC**。主比较是 Original PhaseFormer（均值梯度训练）与SCPRC；Exact-PCGrad负责机制控制。PhaseFormer、PCGrad、PatchTST、TimeMixer、DLinear等保留各自的方法名称与来源。公开标签与检查点内部键的对应见 [model_names.json](../configs/model_names.json)。

h24主实验、独立开关参照、严格删除消融、配对设计对照、h48和外部数据实验继续作为不同运行身份保存。路径规范化不合并它们的结果，也不替换检查点。原协议文件的字节与身份不因命名整理而重写。

## 核对依据

- [命名审计及逐文件映射](../manifests/NAMING_AUDIT.json)：改名前后路径、实验组名称及发布命名扫描。
- [checkpoint索引](../manifests/CHECKPOINTS.csv)：534个路径与来源hash；[权重分包索引](../manifests/RELEASE_ASSETS.csv)列出14个ZIP。
- [结果数值一致性](../manifests/RESULT_VALUE_PARITY.json)：71个结果CSV、114039个数值单元格与整理前一致。
- [服务器原始材料核对](../manifests/ORIGINAL_SOURCE_INTEGRITY.json)：被迁移清单引用的790个paper2源文件hash一致；此检查不等同于服务器全盘快照。
- [源码路径与导入验证](../manifests/SOURCE_NAMING_VALIDATION.json)、[推理验证](../manifests/RUNTIME_AUDIT.json)、[文档链接核对](../manifests/DOCUMENT_NAMING_VALIDATION.json)记录各自范围及结果。

本轮不新增训练或数据集，不重算或筛选论文性能结果。CPU合成输入验证只用于检查重命名后的模型加载和执行，不代表重新完成真实数据评估。源码、摘要和清单进入普通Git；权重与ZIP由Releases分发。本命名审计记录本地整理阶段，仓库与权重发布入口见 [README](../README.md)。
