# 发布内容与复现范围

仓库提供SCPRC的模型实现、训练与评估入口、实验协议、结果表和文件索引。模型权重通过[Releases](https://github.com/xubeiyou-fate/SCPRC/releases/tag/paper-v1)下载；数据从[原始来源](../DATA_SOURCES.md)获取。

## 文件范围

| 材料 | 获取方式 |
|---|---|
| 模型、基线依赖、训练与评估代码 | `scprc/`、`experiments/`、`scripts/` |
| 实验协议、逐折结果与汇总表 | `docs/`、`results/`，见[结果索引](RESULTS_INDEX.md) |
| 模型权重 | 14个ZIP，恢复后为534个检查点路径，见[权重说明](WEIGHTS.md) |
| 数据集、原始标签、逐样本预测、校准残差、扰动掩码 | 不随仓库提供；已有记录列于[来源索引](SOURCE_ARTIFACT_INDEX.md) |
| 归一化参数、阈值和必要推理元数据 | 相应检查点及`manifests/INFERENCE_METADATA.json` |

外部数据上的v2.2 Exact/SCPRC实验提供汇总结果、配置和来源记录，其18份检查点不在下载列表中；该部分的完整冻结推理需要额外产物。外部基线权重的实际范围见`manifests/CHECKPOINTS.csv`。

部分审计JSON保留运行身份、指标、哈希和环境字段，省略重复展开的快照。源文件哈希与发布文件哈希分别记录；这些JSON是有明确来源的摘要。

## 结果解释

- 主比较为Original PhaseFormer与SCPRC。主实验等折平均RMSE/MAE分别为35.0675/8.2817和32.1907/7.0498；Exact-PCGrad用于机制分析。主实验、独立开关、严格删除、配对设计和48小时实验使用各自的运行记录。
- 结果属于回顾性评估。六个城市和三个固定seed构成18个city×seed折，不等于18个独立城市，也不构成未接触数据上的最终盲测。
- 部分Original/MG与Full-no-PCGrad运行在模型构造后设置seed，且缺少共同初始状态记录；其数值差异不能完全归因于PCGrad。配对设计实验的初始化与采样哈希另有记录，见[初始化说明](SOURCE_LIMITATIONS_ZH.md)。
- 独立开关与Full-drop是两组实验。严格删除实验未同批重训Full，GPU执行轨迹及缺失状态限制了微小差异的归因。Source-envelope的增量较弱，Scale和Peer存在指标权衡。
- 配对设计实验中，同批Full的raw Peak误差比严格相位共享对照低约5.08%，但比逐站对照高约10.04%、比拼接对照高约7.17%。逐站对照仍保留跨站公共统计，Peer因子退化为1；拼接对照增加105个参数。门控诊断使用七个固定起点，结果仅适用于该选择范围。
- Shanghai日均基线为175.4674/82.8385 kW，优于所列学习模型。Jiaxing目标是会话能量分摊后的小时kWh代理；UrbanEV与CHARGED-SZH同属深圳，属于跨数据集评估。不同目标和单位的误差分别报告。
- 固定距离Graph WaveNet完成了18折训练及数值、对齐检查；JHB/seed17平行运行差异和基线质量仍待确认，因此未纳入主比较。该实现不等同于Adaptive GWN、MTGNN、DCRNN或所比较文献的动态图复现。
- raw预测与非负截断预测对应不同的评价定义。区间、运行诊断与能源容量情景采用各自协议；容量指标是归一化预留风险代理，不涉及真实电价、变压器约束、排队或电网调度收益。
- 效率结果对应记录的硬件和FP32前向协议，不含数据传输、加载与预处理。配对设计效率记录包含216项测量，不能据此推算端到端服务延迟。

## 来源记录

[文件索引](../manifests/FILE_INDEX.csv)和[SHA-256清单](../manifests/MANIFEST_SHA256.csv)用于核对发布文件。[来源索引](SOURCE_ARTIFACT_INDEX.md)区分已提供材料与仅保留定位的产物；其中的服务器路径是来源信息，运行代码时应使用公开目录结构和外部数据路径。
