# 登录后的上传步骤

本文件提供维护者发布步骤。当前仓库为 `xubeiyou-fate/SCPRC`；以下初次建库命令仅适用于新仓库，现有仓库无需重复执行 `gh repo create`。数据不在发布范围内。先阅读 `README.md`、`DATA_SOURCES.md`、`docs/MIGRATION_SCOPE_ZH.md` 和 `THIRD_PARTY_NOTICES.md`，查看 `manifests/MIGRATION_COMPLETION.json` 中的本地准备阶段验收状态。

## 上传代码

在本目录执行；把 `YOUR_ACCOUNT/SCPRC` 换成你的实际仓库：

```bash
gh auth login
git status --short
git add .
git diff --cached --stat
git commit -m "Prepare SCPRC paper reproduction release"
gh repo create YOUR_ACCOUNT/SCPRC --public --source=. --remote=origin
git push -u origin main
```

如果仓库已经存在，省略 `gh repo create`，设置你实际的 `origin` 后推送。按你的投稿/发布安排选 public 或 private。公开前需处理第三方许可说明中尚未解决的条目，不要把第三方代码统一声明成你的原创许可。

`.gitignore` 会排除本地检查点、权重ZIP、数据、缓存及服务器规划清单。`git add .` 上传的是代码、文档、结果摘要和公开索引，不包括这些大资产。

## 上传最终权重

权重已按实验组准备在 `release_assets/`。它们不会随普通Git提交上传，需要发布到同一仓库的 Releases：

```bash
gh release create paper-v1 release_assets/*.zip release_assets/SHA256SUMS \
  --repo YOUR_ACCOUNT/SCPRC \
  --title "SCPRC paper checkpoints" \
  --notes "Final paper checkpoints. Dataset links and preparation instructions are in DATA_SOURCES.md."
```

将 `paper-v1` 作为本次发布标识；若已存在，应新建版本或明确管理已有Release，不使用覆盖参数偷偷替换旧资产。权重包如何解压以 `release_assets/README.md` 和恢复脚本为准。下载之后先校验 `SHA256SUMS`。

## 数据

不上传数据集、原始/处理后时序、标签或逐样本预测数组。访问者按 `DATA_SOURCES.md` 从官方来源下载到仓库之外，再按配置提供数据路径。UrbanEV GitHub小时版须核对hash，不能用云盘同名文件替代。
