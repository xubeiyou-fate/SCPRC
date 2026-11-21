# 验收工具兼容性更正

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](../../MIGRATION_SCOPE_ZH.md)。


1. Full首个fold准备过程中的source_cities为有序tuple，checkpoint保留tuple而JSON编码为list。逐字段核验确认唯一差异是容器表示，城市内容与顺序完全相同。审计器仅对该字段做有序tuple→list规范化，再严格比较所有协议字段。未修改checkpoint、预测、配置或指标。
2. 效率脚本的默认cuda设备未指定索引，当前Torch的set_device要求明确索引；默认别名改为cuda:0。首次测量没有执行GPU推理，无有效测量。
3. 第二次效率测量的Full读取器遇到同一source_cities的tuple/list表示差异，另外三个控制已测162项、Full的54项未能测量。读取器仅对该字段实施与独立审计相同的有序规范化；没有放宽其他协议比较。
4. 最终从头完成全部216项效率测量，72个checkpoint在batch=1/4/16下全部通过冻结预测一致性校验；missing_or_failed_cases为空。独立实验审计最终通过72/72。最终报告分别为efficiency/manifest.json和audit/completed_runs_audit.json。

初次失败报告保存在validation_history/，最终结果以重新验收和测量后的正式报告为准。
