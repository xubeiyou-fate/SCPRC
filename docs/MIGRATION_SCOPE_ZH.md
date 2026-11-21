# 论文证据迁移范围

源：`/data/TimeMixer`；目标：`/data/TimeMixer/git`。本次只写 `docs/`、`results/` 和 `manifests/EVIDENCE_MIGRATION_AUDIT.json`。根 README、源码、weights、release、planning 和其他目标 manifests 由主侧处理。本次未训练、未推理、未远程操作，未修改源文件。

## 已包含

[当前主矩阵](MAIN_EXPERIMENT_MATRIX.md)、[energy 矩阵](ENERGY_EXPERIMENT_MATRIX.md)、算法与模型协议；主结果及算子、组件、区间、鲁棒性、h48、效率、外部案例和容量情景的精选最终汇总；11 月 5 日严格删组件与 11 月 17 日设计控制的完整聚合结论、逐折 metrics、协议、环境和运行元数据。见 [精选资产清单](RESULTS_INDEX.md)。不迁移历史过期计划或候选优化队列，本次不带图。

只复用已有 summary/result 中的标量指标，不加载预测、不从原始数据重算。大型 20261117 审计和分析 manifest 为明确标注的精选投影：保留状态、fold/run 身份、聚合指标、初始化/采样/源文件哈希及环境，省略重复展开快照。原 SHA256 与迁移后 SHA256 分开记录，不将精简版本称为原件字节副本。

## 不包含的数据与依赖

本次证据迁移复制的数据集、预测数组、校准残差、原始 mask、weights、病例轨迹/逐小时 obs、逐 origin raw truth 数量均为 **0**；也不包含逐 origin 派生明细、原始 data CSV、缓存或压缩数组。仅文件路径/哈希、shape/dtype 和计数是元数据，不是原始数组。外部训练/推理数据不因文档中出现路径而被包含。范围与核验计数见 [数据排除报告](DATA_EXCLUSION_REPORT_ZH.md)。主侧独立迁移的 534 个权重不计入本次证据资产或零权重声明。

源论文矩阵说 legacy 外部模型完成，保留其历史状态与角色语义；**本次按用户范围排除 18 个旧 weights，外部完整冻结推理不在包里，不宣称外部模型已可独立完整复现**。源目录还可能有其他源状态/诊断权重，亦未由本次复制。外部结论只能作历史聚合证据。

## 必须随结果保留的解释边界

- 主比较是 Original PhaseFormer (no-PCGrad) 对 SCPRC (main experiment)；Exact-PCGrad 是机制控制；legacy PhaseFormer Exact、independent-switch Full 与 20261117 同批 Full 各自保持身份。主 headline 为 35.0675/8.2817 对 32.1907/7.0498（RMSE/MAE）。
- 所有结果均为回顾性。六城三固定 seed 及重叠滚动窗口不等于 18 个独立城市，不提供盲测、未见城市、未来时期或重训练总体保证。
- 旧 Original/MG 与 Full-no-PCGrad 存在构造模型后才设 seed、缺少共同初态记录的问题；旧优化器数值差不能完全归因于 PCGrad。见 [初始化证据边界](SOURCE_LIMITATIONS_ZH.md)。不将此缺口传播到有初始化与采样哈希的 20261117 配对控制。
- 严格删组件不等于独立开关。Full-drop 严格组件未同批重训 Full，历史 GPU 轨迹及缺失状态限制微小差异的因果归因；Source-envelope 增量很弱，Scale/Peer 有权衡，不能声称全部组件必要。
- 20261117 同批 Full 对严格相位共享 raw Peak 低约 5.08%，但比逐站控制高约 10.04%、比拼接控制高约 7.17%。逐站控制仍保留跨站公共统计且 Peer 退化为 1；拼接增加 105 参数。诊断仅七个固定起点，不能外推全起点激活率或预测能量守恒。
- Shanghai daily-mean 为必须保留的负面结果（175.4674/82.8385 kW，优于所列学习模型）。Jiaxing 是会话分摊 kWh proxy，UrbanEV 是深圳跨数据集证据；不同单位不池化，不把 UrbanEV 当第七独立城市。保留 seed 波动。
- R10 fixed-distance Graph WaveNet 历史训练 18/18 完成、数值/对齐检查通过，但 JHB/seed17 平行运行差异与基线质量尚待验收。**R10 未验收，不宣称可复现，不纳入已接受主比较**；不称为 adaptive GWN，也不声称 MTGNN/DCRNN 或 Energy 动态图忠实复现。
- raw 与 nonnegative_clip 是不同 estimand；U1、O1 和 energy 容量场景各有协议，不替换 P0 分数。Full 不赢所有指标、扰动或容量 proxy。归一容量不是电费、真实调度、排队或电网可靠性。
- 冻结推理效率为历史记录的硬件/FP32 纯前向时间；216 项是源实验既有验收状态，本次未运行。数据传输、加载和预处理不计入，不能推成服务 SLA。

## 链接与审计

仅修改副本的链接和迁移注释，不更改源版矩阵的角色、结果数值或科学状态。包内文档/小表链接按实际位置改为相对路径。包外源码、图、数据和历史产物改链到 [仅索引清单](SOURCE_ARTIFACT_INDEX.md)，标明未包含；JSON/CSV 中记录的绝对路径、源 SHA、原运行输出清单属于历史 provenance，不是本包存在性声明。旧协议中 planned/ready 等状态为当时快照，当前结论以矩阵及最终验收/summary 为准。

审计逐项记录 source path、解析后路径、source SHA256、destination SHA256、复制/链接重写/精选投影/摘录类型和投影来源。哈希用于迁移一致性核验，不代替模型或实验重现实验。
