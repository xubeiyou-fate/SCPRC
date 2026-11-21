# 历史初始化证据边界

以下摘录源复盘第9–14行，只保留已核查事实与限制，不迁移后续优化计划。方括号 K 编号为源文定位标识。

先处理配对证据，而不是用旧 no-PCGrad 劣势替关系模块背书：

- **本次已复核的实验事实：** 直接读取本批 72 个 [initialization.json](SOURCE_ARTIFACT_INDEX.md#src-1fabee06c4396b33) 与 [sampling_trace.json](SOURCE_ARTIFACT_INDEX.md#src-61c9754e717b991d)，重新计算规范 JSON 哈希；18/18 折四臂 backbone 初始哈希一致，Full 与 strict-shared 的完整初始状态哈希一致，四臂采样与 PCGrad 顺序一致，每臂都有 2,048 个 source minibatch、410 个投影顺序记录。读取结果均与已有独立审计一致。不是重新构造模型的初始化探针，也没有重新读取 checkpoint tensor。证据 CSV 的 `PAIR_*` 保留逐折来源和哈希。[K5] [K6] [K7]
- **代码事实：** `build_model` 在 CPU `fork_rng` 内先按 seed 构造 canonical Full，再替换关系或加入融合层；训练入口另行重置 RNG。各变体本来就有不同形状或额外参数，不能要求四臂完整 state 哈希全相同。关键是共同 backbone 相同、Full/shared 全状态相同、采样和投影顺序相同。[K5] [K6]
- **旧 no-PCGrad 的真实缺口：** R8 runner 第 319 行先构造模型，第 320 行才进入 `_ordinary_gradient_train`，后者第 131 行才执行 `seed_everything`。本次检查全部 18 个 result，均无 `initialization` 记录；18/18 记录的 runner SHA256 与当前文件一致。训练前设 seed 能固定后续采样，不能追溯固定已生成的初始权重。不能将旧 Full/no-PCGrad 数值差完全归因为 PCGrad。[K8]
- **边界：** 上述旧缺口不传播到有实际初始化哈希和采样轨迹的 20261117 四臂。其三项设计比较仍可作固定协议下的配对实验解释；GPU 轨迹不保证逐位确定，四臂也不是 18 个独立城市。历史缺失状态本次无法恢复；任何新的 optimizer 析因须在构造前固定 seed、保存共同初始 state，并成对训练，不能补写历史证据。

原件：[paper2/optimization_review_20261118/MECHANISM_DIAGNOSIS_ZH.md](SOURCE_ARTIFACT_INDEX.md#src-0a98d23b2ddbd0ba)（仅索引，未包含全文）。

Original/MG 和 Full-no-PCGrad 的历史分数保留；缺少共同初态记录时，不把优化器差异解释为严格因果效应。20261117 的初始化与采样哈希证据单独保留。
