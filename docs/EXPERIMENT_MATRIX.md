# 当前论文证据入口

[迁移范围与结论边界](MIGRATION_SCOPE_ZH.md) · [排除核验](DATA_EXCLUSION_REPORT_ZH.md) · [审计 JSON](../manifests/EVIDENCE_MIGRATION_AUDIT.json)

| 证据 | 本包实际入口 | 解释 |
|---|---|---|
| 当前主矩阵 | [MAIN_EXPERIMENT_MATRIX.md](MAIN_EXPERIMENT_MATRIX.md) | 保留源版角色和状态；Original PhaseFormer 对 Full 为主比较 |
| 当前 energy 矩阵 | [ENERGY_EXPERIMENT_MATRIX.md](ENERGY_EXPERIMENT_MATRIX.md) | 保留严格删组件和 20261117 扩展 |
| 模型与协议 | [MODEL_AND_PROTOCOL.md](MODEL_AND_PROTOCOL.md)、[算法](MODEL_ARCHITECTURE.md) | Exact-PCGrad 为机制控制；Online 为独立释放层 |
| 最终表格清单 | [RESULTS_INDEX.md](RESULTS_INDEX.md) | 只含精选聚合结果、逐折 metrics 和协议/环境/运行元数据 |
| 20261105 严格组件结论 | [结果](experiments/energy_value/RESULTS_AND_CONCLUSIONS.md) | 不混用独立开关与 Full-drop；不声称每个组件必要 |
| 20261117 模型控制结论 | [结果](experiments/design_controls/RESULTS_AND_CONCLUSIONS_ZH.md) | 72 个历史训练、216 项历史冻结推理；同批 Full 身份独立 |
| 历史初始化限制 | [事实摘录](SOURCE_LIMITATIONS_ZH.md) | 旧 noPCGrad 构造后设 seed、无共同初态记录；归因受限 |
| 包外产物 | [SOURCE_ARTIFACT_INDEX.md](SOURCE_ARTIFACT_INDEX.md) | 明确仅索引未包含，不能当作包内复现依赖 |

源矩阵中外部 legacy 实验“完成”表示历史实验状态。本次依用户范围排除 18 个旧 weights，外部完整冻结推理不在包里；这不是可独立完整复现的外部模型包。R10 仍未验收，不宣称可复现或纳入已接受主比较。全部证据为回顾性。
