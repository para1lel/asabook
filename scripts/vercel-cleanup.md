# Vercel 部署清理

`.github/workflows/cleanup-vercel-deployments.yml` 在 Vercel 报告 Production 部署成功后运行，也可从 GitHub Actions 手动运行。

脚本只处理 `asabook` 项目。它通过 `www.asabook.cc`、`asabook.cc` 和 `asabook.vercel.app` 三个域名核对实际线上部署，核对成功事件的提交 SHA，然后删除比该版本更早的已结束部署（包含旧预览、失败和取消的部署）。当前线上版本、正在构建的部署以及更新的部署会保留；域名指向不一致或线上版本在清理过程中改变时立即停止。部署失败不会触发清理。旧部署删除后不能再通过其 URL 访问，也不能依靠它们即时回滚。

GitHub 仓库需要 Actions secret `VERCEL_CLEANUP_TOKEN`。在 Vercel 的账户设置中创建专用 Access Token，尽量限制到该项目或所在团队，并将令牌保存在此 secret 中。不要把令牌写入仓库。令牌到期或撤销后需要更新 secret；失败详情可在 GitHub Actions 查看。

本地检查（事先在环境变量中设置令牌）：

```sh
node scripts/cleanup-vercel-deployments.mjs
```

实际清理：

```sh
node scripts/cleanup-vercel-deployments.mjs --apply
```

不带 `--apply` 时只显示待删除列表。Vercel 的存储用量统计可能延迟更新；删除部署不代表已经产生的计量用量会归零。
