# SCPRC GitHub 迁移与上传策略

制定日期：2026-11-21（Asia/Shanghai）。根据最新要求更新：**不迁移、不重新上传任何数据集；只保留官方链接、版本、校验信息和预处理代码。** 这是执行前策略记录。现已按方案迁入源码、最终权重和复现材料，并初始化本地 main 分支；实际数量与验收以 `manifests/MIGRATION_COMPLETION.json` 为准。没有创建远程仓库、提交或上传。

## 发布方式

采用同一个 GitHub 项目的“Git 源码仓库 + 最终权重 Releases”。源码、配置、协议、环境文件、小型汇总结果及清单进入普通 Git；最终权重在 Releases 提供下载。数据由使用者从官方来源自行获取，仓库用 `DATA_SOURCES.md` 标明链接、版本、文件名、校验信息、放置位置和预处理方法。下载源码 ZIP 不会自动包含权重或数据。

GitHub 普通 Git 拒绝超过 100 MiB 的单文件；推荐仓库小于 1 GB。Release 单资产必须小于 2 GiB。最终权重按实验组打包；只有超过限制的权重资产才分卷，分卷不改变权重精度。Git LFS 是备选，需要账户存储和下载流量额度，不作为默认发布途径。数据集不进入普通 Git、Git LFS 或 Releases。

依据：
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage
- https://docs.github.com/en/billing/concepts/product-billing/git-lfs

## 本地目标结构

发布材料位于 `/data/TimeMixer/git/`。模型包统一为 `scprc/`，实验入口统一为 `experiments/`，不沿用源服务器的工作目录名称。以下为当前结构；逐文件清单见 `manifests/FILE_INDEX.csv`：

```text
git/
  README.md                         # 安装、数据、模型角色、复现命令
  DATA_SOURCES.md                    # 官方数据链接、固定版本及准备方法
  RELEASE_PREPARATION_ZH.md             # 当前策略
  scprc/                            # SCPRC模型、训练组件、数据适配器及基线依赖
  experiments/                      # 主实验、消融、设计对照、能源评价入口
  docs/                             # 模型框架、实验矩阵、协议与边界
  results/                          # 已有小型指标表与诊断摘要
  configs/                          # 发布配置，路径相对仓库或可配置
  scripts/                          # 权重获取、数据校验/预处理、推理、评分、训练
  manifests/                        # 文件哈希、运行身份、模型和数据来源索引
  checkpoints/                      # 本地最终权重，普通Git忽略
  local_notes/                      # 本地清点记录，普通Git忽略
  THIRD_PARTY_NOTICES.md             # 许可、固定提交号、本地修改说明
  release_assets/                   # 仅待上传的最终权重包，普通 Git 忽略
```

依赖闭包需要包括 runner 实际导入的项目内模块、模型 layers、数据加载器及配置；不能仅复制模型入口。动态 import、共享 `layers` 命名空间、基于根目录发现路径的代码均需在迁移后检查。源码路径调整只用于迁移版本，不覆盖服务器冻结代码或结果。

## 筛选原则：按论文身份，不按日期删文件

发布库的矩阵入口是 `docs/EXPERIMENT_MATRIX.md`，详细主矩阵和能源矩阵分别为 `docs/MAIN_EXPERIMENT_MATRIX.md`、`docs/ENERGY_EXPERIMENT_MATRIX.md`，已纳入配对设计控制。源服务器的原始路径和运行ID保留在来源索引中，不能以文件改名替代实验身份核对。

主比较始终是 Original PhaseFormer（MG）vs SCPRC；Exact-PCGrad 是机制控制。独立开关 Full、严格消融、20261117 同批 Full、h=48 Full 和外部 legacy v2.2 各有独立身份，不能互相重命名、覆盖或换权重。

需要保留：
1. h=24 Original、Exact-PCGrad、Daily relation、Phase relation、Full，以及 Full-no-PCGrad；每个模型完整保留六城 × seeds 13/17/29。
2. 独立 Scale/Peer/Source 开关及其组内 Exact/Full，严格 Full-drop 三臂和 Daily+完整衰减。
3. Attention、DeepSets；20261117 Full/shared/stationwise/concat 四臂。保留配对参照，不能只保留胜出模型。
4. h=48 Original、Exact、Full；不能用 h=24 checkpoint 替代。
5. 主表 PatchTST、TimeMixer、DLinear、本地 N-BEATS 图残差；规则基线只需代码和配置，不存在待迁移权重。
6. Jiaxing、Shanghai、UrbanEV 的当前基线实现、配置、结果和对应基线权重。遵守“不上传历史版本模型”：外部表仍使用的 legacy v2.2 Exact/SCPRC 共18份旧谱系权重只登记来源，暂不列入上传白名单。相应外部旧表可以提供已有结果与溯源，但不能声称已经交付其完整冻结推理复现，也不能改用 canonical 权重冒充。
7. U1/U2、效率、能源指标、配对统计等需要的协议、校准/扰动生成代码、掩码 hash 及小型统计。原始标签、输入、校准残差、逐样本预测和原始掩码数组不迁移；记录其再生条件及无法完全再生的限制。重用同一 checkpoint 的实验只在索引增加引用，避免重复存储。

不迁移：废弃版本、smoke/benchmark、失败或中断训练、按 epoch 的中间权重、独立优化器缓存、历史写作包副本、无关预训练模型、嵌套 `.git` 和运行缓存。服务器原材料不删除。负面结果若仍在论文矩阵中，应保留，不因指标较差排除。

已定位主表两变量 DLinear 的18份保存容器：seed13/17 位于 `charged_linear_loco_2048/seed{seed}`，seed29 位于 `charged_linear_loco_20261021_seed29`。容器同时含 DLinear/NLinear 和优化器状态；公开推理包只导出当前矩阵 DLinear，记录原容器 hash，并在导出后核对张量及推理一致性，不引入无关 NLinear 模型。可选 Online 的额外资产仍需逐项核对。固定距离 Graph WaveNet 当前仍有待解决的运行来源和质量问题，单列 review-pending；不把它写成已完成 Adaptive GWN/MTGNN。

## 预训练权重

已核对的核心 SCPRC/PhaseFormer 和主要基线从初始化开始训练；尚未发现当前主路径必须加载额外预训练权重。旧 `reproduction/runs` 约 45 GiB 权重，不整体复制。只有配置/代码确实引用且属于发布范围的预训练文件才入包；训练后 checkpoint 和预训练权重分开登记。不能为了满足目录名称而把旧模型伪装成必需预训练模型。

## 数据只保留链接与版本，不复制内容

所有原始数据包、解压后的时序、Excel、加工面板和数据缓存均留在现有服务器路径，不复制到 `/data/TimeMixer/git`。`local_notes/DATA_REFERENCES.csv` 仅保存来源、文件身份、体积和 hash，不是数据迁移白名单。已生成的旧名 `DATA_CANDIDATES.csv` 将由清点脚本撤下，防止误用。

| 数据 | 仓库中保留的信息 | 官方链接及版本 |
|---|---|---|
| CHARGED | Hourly.zip 文件名、MD5、加载器、使用范围 | https://doi.org/10.5281/zenodo.15814263 ，v3 |
| UrbanEV GitHub | 23文件的名称及 hash；论文所用 volume 身份 | https://github.com/IntelligentSystemsLab/UrbanEV/tree/main/data |
| UrbanEV 云盘扩展 | 主包、补充包链接和 hash；注明非主实验必需 | https://drive.google.com/drive/folders/1VUgdb8uNgmtvO93BHBK_OrSxjndrF-48 |
| Jiaxing | Dataset.zip 来源与 MD5、小时面板构建代码和目标 hash | https://doi.org/10.6084/m9.figshare.28182251.v3 |
| Shanghai | Excel来源与 MD5、面板构建代码和目标 hash | https://doi.org/10.6084/m9.figshare.29882366.v3 |

详见 `DATA_SOURCES.md`。使用者下载后以原始精度、完整城市和固定规则构建面板；仓库保存原始单位、station/zone 索引定义、时间边界、文件 hash 和预处理规则。UrbanEV GitHub 与云盘的部分 volume 不同，不能互换；移动的 main 链接需配合固定提交或逐文件 hash，不能无校验地宣称版本一致。CHARGED/Jiaxing/Shanghai 保留 CC BY 4.0 来源署名；UrbanEV 保留官方 CC0 声明来源。

数据根目录设为显式 `--data-root` 或配置项，由使用者指向仓库外的下载目录。`git/data/` 即使创建也只放说明，不放数据文件。当前仓库不打包任何已有加工面板；面板应在用户的数据目录从官方原始文件重建并核对 hash。

## 权重与结果必须能对应

每个 checkpoint 的索引至少包括 model_id、matrix_id、run_id、city、seed、context、horizon、source path、SHA-256、代码/配置 hash、scaler、source threshold、站点顺序、起点时间和指标定义。需要多个代码版本时提供各自固定快照/补丁，不能只提供一个修改后的模型文件解释所有历史权重。

原始 checkpoint 作为权威文件；如另导出仅含 state_dict 的推理权重，必须记录来源、转换代码和冻结推理一致性校验。未经核对，不删除 scaler/阈值等非参数信息。不得为发布修正既有训练或选模记录；旧 no-PCGrad 初始化记录不足应作为复现边界说明。

原始输入、真实标签、冻结预测、校准残差和原始扰动掩码不进入此次迁移或 Releases。保留生成代码、参数、随机数信息和 hash；无法根据公开原始数据及记录严格再生的旧结果，明确标为“已有汇总可查，严格再生尚未验收”，不能重建不同数组后冒充原配对结果。模型必要的 scaler/阈值可作为小型推理 metadata 保留。

## 参考同类项目的做法

- TimeMixer：GitHub 放实现和实验脚本，README 指向外部数据下载。https://github.com/kwuking/TimeMixer
- PatchTST：区分 supervised/self-supervised 路径，给出数据放置和训练/预训练命令，不把所有训练历史混成仓库。https://github.com/yuqinie98/PatchTST
- UrbanEV：GitHub 提供小时基准目录，README 指向 Dryad/云盘的多粒度数据。https://github.com/IntelligentSystemsLab/UrbanEV
- Chronos：GitHub 维护推理实现和使用说明，模型表链接到单独权重托管。用户本项目优先使用同一 GitHub 项目的 Releases，实现代码和大资产分发分离。https://github.com/amazon-science/chronos-forecasting

## 正式执行顺序与验收

1. 确定逐文件 allowlist，完成 DLinear 容器拆分和 Online 等待核对项；记录保留与排除理由。
2. 并行整理源码依赖、最终权重、论文证据三组，另整理数据链接/版本说明；不复制数据文件。权重复制实际文件，不留下指向 `/home/cwadmin/exports` 的服务器软链接。
3. 锁定源仓库提交号和本地修改；保留上游许可证。PhaseFormer 与 DLinear 许可文本在当前局部快照中未见，需要核对上游；许可不明代码可先本地迁移，不能直接按本项目许可公开再分发。
4. 从实际运行环境形成统一依赖文件，保留训练时环境记录；不能拼接互相冲突的旧 vendor requirements。改为仓库相对路径/显式 data-root、checkpoint 参数；扫描公开文件里的凭据和无关私人路径。
5. 在隔离目录验证 import、加载 checkpoint、CPU 最小推理、GPU 冻结推理及指标 parity；使用显式数据根目录指向现有服务器数据，不把数据迁入仓库。不训练新模型，不覆盖结果；各模型/实验组有独立验收状态。
6. 普通 Git 添加代码/小表/清单/数据链接说明；Release 仅上传按模型组分装的最终权重。每个权重资产附 SHA-256、版本标识、下载/解压说明，执行下载后复验；检查发布目录内没有数据包、时序CSV或加工面板。
7. 新建代码版本 tag 和同版本 Release。仓库地址和公开状态由实际创建时确定，不编造 URL。正式发布之前，先核对本地验收结果；本轮未执行远程操作。

`local_notes/` 中的清点记录不进入普通Git。当前发布白名单由 `manifests/CHECKPOINTS.csv` 定义；源服务器扫描过的重复校准、smoke和中间产物不因此进入发布范围。验收范围以逐模型身份、逐文件索引和runtime记录为准。


## 本轮执行结果

本地迁移已完成。共534个检查点路径、14个权重ZIP；数据集与原始预测/标签数组迁移数为0。534/534 CPU合成推理验收通过，记录见 `manifests/MIGRATION_COMPLETION.json`。本地main分支已初始化，无远程、无提交、无上传。原生权重保留内嵌优化器/PRNG状态以维持原始hash；DLinear另行导出并验参。
