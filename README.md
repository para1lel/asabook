# ASa Book

基于 VuePress 2、Vite bundler 与 Plume 主题的中文文档 / 博客站。

## 开发

```sh
npm install
npm run docs:dev
```

如果修改了 VuePress 配置或缓存相关内容，可以运行：

```sh
npm run docs:clean-dev
```

## 构建

```sh
npm run docs:build
```

构建产物位于 `docs/.vuepress/dist`，不要手动编辑或提交生成目录。

## 部署

`main` 分支推送后，GitHub Actions 触发内网服务器 `192.168.80.116` 拉取完整历史仓库并构建，通过 Cloudflare Tunnel 发布。也可在 Actions 中手动运行 **Deploy ASa Book**。

服务器仓库路径为 `/data0/shared/dongwu.chen/asabook`。部署、日志与回滚操作见 [部署说明](scripts/deploy/README.md)。站内搜索只索引页面标题，避免增加客户端搜索索引负担。
