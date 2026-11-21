# R13 严格时间外推审计

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](../../../MIGRATION_SCOPE_ZH.md)。


状态：**completed_no_blind_test**。本目录只做时间边界和标签可见性审计，没有启动重训。

## 结论

当前 CHARGED 归档覆盖 2023-04-01 00:00 至 2023-09-30 23:00。canonical protocol 使用 4–7 月源城市训练、8 月目标城市评估；该协议本身已经把 August 标为 prior development 暴露区间。项目历史中还存在完成的 September final/nested readout，因此 9 月也不能再作为项目级盲测。归档结束于 9 月，没有更晚且未触碰的未来月份。

## 已核对的时间边界

- 原始归档：6 个城市均为 4,392 小时，2023-04-01 至 2023-09-30；每城 August 744 行、September 720 行。
- canonical 训练：2023-04-01 至 2023-07-31；评估：2023-08-01 至 2023-08-31；每城 721 个 24 小时滚动 origin。
- R3 区间：每 fold 使用同一 checkpoint 的 `target_history_same_fold_checkpoint` 残差校准，361 个 calibration origins；当前 fold JSON 没有保存这些 origins 的日历区间，故不能擅自声称具体的独立月份划分。
- August 标签：历史 release gate、strict nested audit 和当前旧结果均明确记录 August 被用于 release/development audit。
- September 标签：历史 `september_final` 和 nested September 产物记录了 September reads、外层预测和 September metrics；文档同时明确这些结果是 retrospective。

## 能证明什么

- 可以把现有 August 与 September 数字作为有时间顺序的回溯性审计结果，并完整报告其数据范围、切分和 lineage。
- 可以提出一个未来数据到达后的严格 protocol：早期月份训练、下一月份验证和校准、之后新到达月份一次性测试。

## 不能宣称什么

- 不能把 August 或 September 写成 project-level blind confirmation 或 untouched future test。
- 不能因为重新训练了 Original/Full，就把已经被读取过的 September 变成盲测。
- 不能把当前区间结果写成分布无关的 formal coverage guarantee；当前 R3 仍是锁定 protocol 下的经验 coverage/WIS/capacity risk。

## 后续可补做的严格 protocol

1. 在新的、晚于 2023-09 的真实月份或新数据版本到达前冻结模型、Original/Full、基线、预处理、区间校准规则、指标和决策回放规则。
2. 训练只使用早期连续月份；验证/校准只使用紧邻的后续月份；最后的新月份只读取输入历史和最终标签一次。
3. 仍按 6 个目标城市 × 3 seeds = 18 folds，保留 168 小时输入、24 小时预测和 paired moving-block bootstrap。
4. 保留 label-access log、raw timestamp manifest、checkpoint/prediction hashes，并把最终新月份结果标成 confirmatory only after the freeze。

## 可复现文件

- [raw_time_axis.csv](../../../SOURCE_ARTIFACT_INDEX.md#src-edea627a34c7cd13)：每城原始时间轴与成员哈希。
- [evidence_index.csv](../../../SOURCE_ARTIFACT_INDEX.md#src-27a81bf5db51f93d)：August/September 历史可见性证据索引。
- [audit_report.json](../../../SOURCE_ARTIFACT_INDEX.md#src-5efbd3028ff2eb81)：机器可读审计结论、边界与后续 protocol。
- [protocol.json](../../../../results/experiments/evaluation/temporal_evaluation/protocol.json)：本次审计和建议 protocol 的冻结摘要。
- [manifest.json](../../../SOURCE_ARTIFACT_INDEX.md#src-d3a5cb5b84c96322)：输入文件哈希和输出清单。
- [status.json](../../../../results/experiments/evaluation/temporal_evaluation/status.json)：当前任务状态。

本任务没有修改 [paper2/MAIN_EXPERIMENT_MATRIX.md](../../../MAIN_EXPERIMENT_MATRIX.md) 或任何主矩阵文件。
