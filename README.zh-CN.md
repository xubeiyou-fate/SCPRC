# SCPRC

跨城市电动汽车充电需求预测。

[English](README.md) · [实验说明](docs/EXPERIMENT_MATRIX.md) · [模型权重](docs/WEIGHTS.md) · [数据来源](DATA_SOURCES.md)

**论文状态：正在准备期刊投稿。** 论文链接与引用信息将在可公开时补充。

SCPRC以PhaseFormer为预测骨干，将多源城市的PCGrad训练与相位条件站点关系适配结合。关系适配器通过站点尺度、邻站可靠性和源域分歧信号调节残差修正。

仓库包含模型实现、训练与评估代码、实验协议和结果表。训练好的权重见[Releases](https://github.com/xubeiyou-fate/SCPRC/releases/tag/paper-v1)；数据请从[原始来源](DATA_SOURCES.md)另行下载。

## 安装

推理代码已在Python 3.12、PyTorch 2.7.1、NumPy 1.26.4和pandas 2.2.2下检查。CPU安装示例：

```bash
git clone https://github.com/xubeiyou-fate/SCPRC.git
cd SCPRC
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

CUDA安装及环境记录见[环境说明](environment/README.md)。

## 数据

主实验使用CHARGED六城小时数据：AMS、JHB、LOA、MEL、SPO和SZH。外部数据包括Jiaxing、Shanghai MP-EVData和UrbanEV。下载链接、文件哈希、单位和预处理规则见[数据来源](DATA_SOURCES.md)，数据应放在仓库之外。

UrbanEV实验使用官方GitHub的小时版`volume.csv`。云盘中的同名文件部分数值不同，请按记录的哈希选择版本。

## 下载权重

从[paper-v1](https://github.com/xubeiyou-fate/SCPRC/releases/tag/paper-v1)下载14个ZIP和`SHA256SUMS`，放入`release_assets/`。使用GitHub CLI时可执行：

```bash
mkdir -p release_assets
gh release download paper-v1 --repo xubeiyou-fate/SCPRC \
  --pattern '*.zip' --pattern SHA256SUMS --dir release_assets
(cd release_assets && sha256sum -c SHA256SUMS)
python scripts/restore_checkpoints.py
python scripts/verify_checkpoints.py
```

恢复后共有534个检查点路径，其中六个路径对应相同权重的别名。模型、城市、seed、预测时域和文件哈希见[权重索引](manifests/CHECKPOINTS.csv)；分组说明见[权重文档](docs/WEIGHTS.md)。

## 推理

以下命令使用AMS、seed 13的24小时SCPRC权重：

```bash
python scripts/infer.py \
  --checkpoint checkpoints/forecasting/h24/scprc/AMS/seed_13/model.pt \
  --input /path/to/history.npy \
  --output /path/to/forecast.npy \
  --device cpu
```

输入为原始单位的`[batch, 168, stations]`数组，输出为`[batch, 24, stations]`；48小时权重输出48步。输入与输出数组应放在仓库之外。推理程序自动读取源域归一化参数，默认返回未经截断的预测；非负截断需显式启用。

两变量DLinear权重还需要`--duration-input`。完整选项见`python scripts/infer.py --help`。

## 实验

主实验采用六城留一评估，seeds为13、17、29，输入168小时、预测24小时。模型使用2023年4—7月的源城市数据训练，在目标城市8月数据上评估。

主比较为Original PhaseFormer与SCPRC。Exact-PCGrad用于分析训练与关系适配器的作用；其余实验包括组件及关系算子对照、区间预测、输入扰动、48小时预测、能源指标和外部数据评估。

[实验总览](docs/EXPERIMENT_MATRIX.md) · [模型与协议](docs/MODEL_AND_PROTOCOL.md) · [主实验矩阵](docs/MAIN_EXPERIMENT_MATRIX.md) · [能源实验](docs/ENERGY_EXPERIMENT_MATRIX.md)

### 结果的适用范围

这些结果属于回顾性评估。8月结果曾用于部分关系适配器的设计，已查看的9月结果影响过后续研究方向，因此上述时段不属于完全未接触的最终测试。

不同城市和指标存在权衡：部分简化适配器在个别指标上表现相当或更好，Shanghai日均基线优于该数据集上所列的学习模型。外部数据的目标和单位不同，结果分别报告。

部分均值梯度对照缺少与PCGrad对照共同初始化的记录，优化器归因因此受限；具有初始化记录的配对设计实验另行说明，见[初始化与比较限制](docs/SOURCE_LIMITATIONS_ZH.md)。容量实验衡量预留风险情景，不代表实际电费收益或电网调度效果。

## 引用与许可

论文公开前，请注明本仓库及使用的release或commit。PhaseFormer、PCGrad及其他基线保留各自的来源署名。各组件许可见[第三方说明](THIRD_PARTY_NOTICES.md)；固定PhaseFormer版本的复用许可仍待核实。
