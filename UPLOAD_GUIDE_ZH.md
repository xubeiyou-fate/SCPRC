# 维护与发布

仓库：[`xubeiyou-fate/SCPRC`](https://github.com/xubeiyou-fate/SCPRC)。主分支为`main`，模型权重单独保存在Releases。

## 更新代码与文档

检查更改和文件清单后提交：

```bash
git status --short
git diff
git add README.md docs/
git diff --cached --stat
git commit -m "Update documentation"
git push origin main
```

以上`git add`路径应按实际改动调整。发布文件发生变化时，同步更新`manifests/FILE_INDEX.csv`与`manifests/MANIFEST_SHA256.csv`，并运行：

```bash
python scripts/verify_file_manifest.py
```

## 发布权重

现有权重版本为`paper-v1`。新增或更换权重时，建立新的版本，记录对应代码提交、检查点身份和SHA-256。发布前核对训练协议、模型参数、归一化信息与文件清单。

`checkpoints/`与`release_assets/`由Git忽略；权重ZIP和`SHA256SUMS`作为Release附件上传。下载与恢复方式见[权重说明](docs/WEIGHTS.md)。

## 发布前检查

- 确认文档中的模型、实验与权重路径有效。
- 确认未提交数据集、原始标签、逐样本预测、访问凭据或运行缓存。
- 保留第三方代码与数据的来源、许可和必要署名。
- 区分文档修订、代码修订和重新训练产生的权重变化。

代码依赖、实验结果及其解释范围见[实验说明](docs/EXPERIMENT_MATRIX.md)和[发布内容](docs/MIGRATION_SCOPE_ZH.md)。
