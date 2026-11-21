# 严格消融独立验收（已完成）

> 迁移说明：这是源文档的证据副本，保留原角色、指标和历史实验状态；只修复链接。外部 legacy 所需的 18 个旧 weights 已按本次范围排除，外部完整冻结推理不在包里；R10 未验收，不宣称可复现。数据/预测/图/源码等包外路径仅索引，未包含。大型审计 JSON 的链接指向明确标注的精选元数据投影；正文 SHA256 是源产物历史指纹，副本 SHA256 见迁移审计。详见 [迁移范围](../../../../MIGRATION_SCOPE_ZH.md)。


**结论：PASS_WITH_PROVENANCE_AND_CAUSAL_LIMITATIONS。** 18/18 folds、4 arms、72/72 checkpoints 的核心验收全部通过。机器结果见 [audit.json](../../../../../results/experiments/energy_value/gate_removal/audit/audit.json)、72 行 [audit.csv](../../../../../results/experiments/energy_value/gate_removal/audit/audit.csv) 和 [per_fold/](../../../../SOURCE_ARTIFACT_INDEX.md#src-feeb741c70e5a3db)。本次未启动训练、未调用 backward 或 optimizer step；所有写入仅限本 [audit/](../../../../SOURCE_ARTIFACT_INDEX.md#src-eee3f86400157096)，未修改 runner、manifest、result、原始预测或 canonical。

## 验收结果

- 实际读取全部 72 个 checkpoint 的 state tensor，全部有限；每个含 74 个 tensor、37,263 个参数。keys、shape、dtype 与目标模型匹配，并逐个 `load_state_dict(strict=True)` 成功。
- gate flags 和 relation mode 在 checkpoint、result、fold protocol、manifest 与实际构造模型之间全部一致：drop-scale=(false,true,true)，drop-peer=(true,false,true)，drop-source=(true,true,false)，Daily=(true,true,true)。前三者 relation=`phase`，Daily=`phase_agnostic_daily`；均保留 support_scale=1.0。
- 独立打开全部 18 个 strict NPZ 与对应 canonical NPZ，truth、label_starts、site_ids 的 dtype、shape、逐元素内容全部精确一致。721 个起点为 2928 至 3648，horizon=24，站点顺序一致。未以父评估器对齐表替代独立检查。
- 全部 72 个保存预测均为有限 float32，shape 与 truth 一致。以分块 FP64 计算未经裁剪的 raw RMSE/MAE，对原 result 最大绝对偏差分别为 1.137e-13、2.309e-14，均小于 1e-12。
- 每个 arm 均为 2,048 source minibatches × batch 4，即 8,192 source windows；410 PCGrad optimizer rounds。五 source 城市按固定顺序分配 410/410/410/409/409，ordered gradient pairs=8,186；AdamW lr=0.001、weight_decay=0.0001、损失及 PCGrad 实现匹配 canonical。
- manifest 的 runner/adapter hash、冻结协议 hash，以及全部 canonical result 记录的代码 hash 均匹配现文件。checkpoint、strict/canonical NPZ、result/protocol 文件及预测数组、truth/starts/sites、checkpoint state 的 SHA-256 已记录。数组 hash 包含 dtype、shape 与 C-order 内容，不能直接与只 hash 裸 bytes 的工具比较。

## Raw 点指标

先对每 fold 的 site × lead × origin 计算原始量纲指标，再对 18 folds 等权平均；不裁剪负预测，不替代 city-energy 指标。

| arm | RMSE | MAE |
|---|---:|---:|
| full_drop_scale | 32.753439525172 | 7.109027845313 |
| full_drop_peer | 32.161230125530 | 7.039355403643 |
| full_drop_source_envelope | 32.181134724831 | 7.048266236627 |
| daily_full_attenuation | 32.621931235849 | 7.044506563685 |

## 可匹配的信息预算、RNG 与初始化

新旧 runner 均在每 arm 构造前重置 Python、NumPy、Torch CPU/CUDA seed；训练入口再次重置。两套 seed 函数去除 docstring 后 AST 相同，两套 runner 实际引用同一个 `_train_candidate_pcgrad` 函数对象。

三 seeds × 四 arms 的 12 组 CPU 构造探针（11 站点、共享参数架构）均与旧 canonical Full 的初始 state 完全相同；构造后 Python/NumPy/Torch CPU/CUDA RNG 状态 hash 完全相同，relation correction 末层初始为零。这是当前代码的初始化验证，不是历史训练初始快照的恢复。

训练采样为 `default_rng(seed)`，PCGrad 顺序为 `default_rng(seed ^ 0x5C2C)`，threshold 使用独立的 `default_rng(seed + 7001)`。source 顺序、训练起点、归一化来源、预算、损失和评价 origin 可匹配。预算证据来自逐城更新记录和同一实现；checkpoint 没有 optimizer/RNG 快照或逐步 batch trace，因此最终参数不能独立证明执行过每一步。

## Source-drop 的 null threshold 边界

Full-drop-source 的 threshold 元数据为 null，构造器实际传入占位值 1.0，source gate 关闭。但 fold 在 arm 循环前仍为另外三个 arm 共用估计一次 source-only q90，其数值与 canonical 一致。逐 arm seed reset 隔离了该预处理对后续 RNG 的影响。

因此 `not_estimated_gate_disabled` 应解释为“该 arm 不采用估计阈值”，不能解释为“整个 fold 没有计算 threshold”。可确认删除了 source envelope 的 forward attenuation；不能声称整个 fold 的阈值估计计算预算被删除。审计保留原 metadata，单独披露这一用语范围。

## 跨运行时间确定性与因果精度

**没有同次重训的 Full control。** 评价对齐、初始化约定和信息预算匹配，不代表不同运行时间的 GPU 训练轨迹逐位确定，也不能把对旧 canonical 的全部差异都归因于组件移除。

父评估器报告旧 Full 独立开关重复运行与 canonical 仅 3/18 folds 预测逐位一致。[../../02_paired_stats/full_repeat_run_sensitivity.csv](../../../../../results/experiments/energy_value/02_paired_stats/full_repeat_run_sensitivity.csv) 量化的是观察到的跨运行敏感性，并非正式训练噪声估计。例如 raw peak magnitude MAE 的重复运行均值差约 +0.96581，source-drop 对 canonical 约 +2.32583；raw total-24h MAE 分别约 -20.32327 和 -68.37334。细小 source-envelope 效应必须与相应重复运行敏感性并列解释，不能提升为高精度组件因果结论。本审计不新增训练或噪声估计。

## Provenance 与验收限制

[environment.json](../../../../../results/experiments/energy_value/gate_removal/audit/environment.json) 记录当前审计环境：Python 3.12.7、Torch 2.7.1+cu118、NumPy 1.26.4、pandas 2.2.2、RTX 4080 SUPER。[imports.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-0dd3aa515c4c6c49) 记录实际导入的 1,414 个模块/兼容 shim，并对在盘模块保存实际解析路径与 SHA-256，包含仓库模型、vendor 和传递导入。无 Lightning 发行包，实际兼容 shim 已记录。

旧 canonical 记录的 Python/Torch/设备与当前一致；新 strict 没有逐 fold 训练环境快照或完整传递依赖 hash。当前 `deterministic_algorithms=false`、`cudnn_deterministic=false`，runner 未启用确定性约束。审计时版本、导入及现文件 hash 不能追溯证明未记录的历史训练环境或依赖。

gate/relation 是 Python 属性而非 state_dict tensor；验证依靠 checkpoint metadata、受 hash 约束的构造器及严格 state 加载。已完整检查保存预测与全部 checkpoint，但未执行 checkpoint→保存预测的全量推理回放，也未执行拟议的少量 origin forward 抽查；不能附加声称两者具有函数等价证明。

原 runs 为 symlink，实际存储为 `/home/cwadmin/exports/SCPRC_Energy_Value_20261104_strict_runs`。本次通过原路径读取，记录内容 hash，未修改存储或 symlink。

## 复核与改动文件

```bash
PYTHONDONTWRITEBYTECODE=1 /home/cwadmin/anaconda3/bin/python -u \
  /data/TimeMixer/paper2/experiments/energy_value_20261104/agent_strict_gap/audit/validate.py
```

该命令只验收既有产物并更新 audit 内机器结果，不调用训练。此次新增/修改文件均位于本目录：

- [README.md](README.md)
- [validate.py](../../../../SOURCE_ARTIFACT_INDEX.md#src-649d1de18e764116)
- [audit.json](../../../../../results/experiments/energy_value/gate_removal/audit/audit.json)
- [audit.csv](../../../../../results/experiments/energy_value/gate_removal/audit/audit.csv)
- [environment.json](../../../../../results/experiments/energy_value/gate_removal/audit/environment.json)
- [imports.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-0dd3aa515c4c6c49)
- [source_hashes.json](../../../../../results/experiments/energy_value/gate_removal/audit/source_hashes.json)
- [rng_initialization.json](../../../../../results/experiments/energy_value/gate_removal/audit/rng_initialization.json)
- [progress.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-48c6c77fe157e05e)
- `execution.log`
- [per_fold/target_ams_seed13.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-d1947da2635ef974)、[target_ams_seed17.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-3da03d70bc5a3b5d)、[target_ams_seed29.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-a3ca5da21a1478f4)
- [per_fold/target_jhb_seed13.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-563404599a14b5bc)、[target_jhb_seed17.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-3c9b50bd471d116e)、[target_jhb_seed29.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-9e3f23a494275606)
- [per_fold/target_loa_seed13.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-e07e233407a469a9)、[target_loa_seed17.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-2763931c9ea0a921)、[target_loa_seed29.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-ce5a10e6c398f9cc)
- [per_fold/target_mel_seed13.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-718061bed480d08b)、[target_mel_seed17.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-81774c96cb9d0bd9)、[target_mel_seed29.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-e298db14990e4e28)
- [per_fold/target_spo_seed13.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-9d52d0c462dd003f)、[target_spo_seed17.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-0d322fed0fb6003f)、[target_spo_seed29.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-02a71dc808dbac08)
- [per_fold/target_szh_seed13.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-e3c40c0a6f9719db)、[target_szh_seed17.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-474f1df2eae8c6e7)、[target_szh_seed29.json](../../../../SOURCE_ARTIFACT_INDEX.md#src-2a319b04b7697d74)
