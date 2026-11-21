# 数据未复制与逐 origin 排除核验

本报告只针对本代理写入的 `docs/` 和 `results/`，以及唯一证据审计 JSON。主侧已经独立迁移的 534 个权重不属于本报告零权重声明的统计范围。没有检查、改写或删除那些权重。

本次数据集、预测数组、校准残差、原始 mask、weights、病例轨迹/逐小时 obs、逐 origin raw truth、逐 origin 派生明细及原始 data CSV 的复制数均为 **0**。未迁入图片或任何二进制文件、压缩数组、数据缓存、symlink。未加载模型、未读预测数组、未训练、未推理、未远程操作。所有指标直接复用已有 summary/result 标量，没有数十 GB 预测重读。

只纳入 Markdown、精选 CSV 和 JSON。逐折 CSV 的列和行数、JSON 容器检查逐项记录在 [审计](../manifests/EVIDENCE_MIGRATION_AUDIT.json) 的 `data_exclusion_report`。CSV 不含逐起点键 `origin_time/label_start/target_time` 或 `obs/truth/y_true/y_pred/prediction` 数值列。少量固定效率起点列表为选择协议元数据，不含原始观测。JSON 中只允许短范围、shape 和协议列表；真值/预测/残差等只可出现文件指纹或命名，不能带原始数值载荷。最大数值列表长度为 4，未发现大数组或原始数值载荷。

以下计数仅扫描源 `paper2/` 的目录条目、文件大小，不读取被排除文件内容；其他源数据目录及历史迁移包也完全未复制。类别互斥，非全盘数据集清单，也不能用来推断全盘不存在其他数据。

| 源候选类别 | 未迁入文件数 | 未迁入总字节（stat） |
|---|---:|---:|
| historical_or_proposed_plans | 9 | 82705 |
| origin_or_raw_data_csv | 4 | 365080967 |
| datasets_arrays_or_compressed_payloads | 941 | 64362753014 |
| weights | 1048 | 255746178 |

外部 legacy：按用户范围排除 18 个旧 weights；源处其他诊断/源状态权重也未由本次复制。矩阵的“历史实验完成”语义保留，但外部完整冻结推理不在包里，不宣称可独立完整复现。R10 未验收，不宣称可复现。包外引用见 [仅索引清单](SOURCE_ARTIFACT_INDEX.md)。

源路径与 hash 保留；保留 hash/shape/计数不等于包含对应数据。大型审计 JSON 已做白名单精选投影，并记录原件及副本各自 hash。
