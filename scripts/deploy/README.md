# ASa Book 部署

站点在 `192.168.80.116`（用户 `dongwu.chen`）构建和发布，Cloudflare Tunnel 直接连接公网，不加载代理。`asabook.cc` 保留路径和查询参数，以 308 跳转到 `https://www.asabook.cc`。

## 目录与服务

| 路径 | 用途 |
| --- | --- |
| `/data0/shared/dongwu.chen/asabook` | `main` 分支完整历史仓库 |
| `/data0/shared/dongwu.chen/asabook-deploy/releases` | 已完成的独立静态版本 |
| `/data0/shared/dongwu.chen/asabook-deploy/current` | 当前版本的符号链接 |
| `/data0/shared/dongwu.chen/asabook-deploy/assets` | 跨版本保留的构建静态资源 |
| `/data0/shared/dongwu.chen/asabook-deploy/jobs` | 部署任务状态 |
| `/data0/shared/dongwu.chen/asabook-deploy/logs` | 按任务 ID 保存的构建日志 |
| `~/.config/asabook/deploy.env` | 部署接口密钥，权限 `600` |
| `~/.config/asabook/tunnel.yml` | 实际 Tunnel 配置 |
| `~/.cloudflared` | Cloudflare 证书与 Tunnel 凭据，不提交到仓库 |

三个 systemd 用户服务分别为 `asabook-web`、`asabook-deploy` 和 `asabook-tunnel`。用户启用了 linger，退出 SSH 后仍运行，并随机器启动。

```sh
systemctl --user status asabook-web asabook-deploy asabook-tunnel
journalctl --user -u asabook-tunnel -n 100 --no-pager
```

Node.js 使用 `~/.local/opt/node-v24.18.0-linux-x64`，不替换系统 Node。部署服务为 V8 提供 64 GiB 堆上限。Caddy 与 cloudflared 位于 `~/.local/bin`。

## 自动发布

`.github/workflows/deploy.yml` 只处理 `main` 的 push 和在 `main` 上手动运行的 workflow，不运行 PR 代码。GitHub 托管 runner 通过 HTTPS 调用 `deploy.asabook.cc`，使用仓库 Actions secret `ASABOOK_DEPLOY_TOKEN`；只有服务器执行拉取和构建。

服务器验证请求、串行部署并使用 `flock` 防止重叠。它拒绝浅克隆、脏工作区、错误 origin 或非 `main` 分支，仅拉取远端当前 `main`，不接受任意命令或分支。过期提交标记为 `superseded`。每次运行：

```sh
git pull --ff-only origin main
npm ci --no-audit --no-fund
npm run paper:check-config
npm run docs:build -- --clean-cache
```

构建完成后复制到独立 release，再原子切换 `current`。构建失败不切换；切换后的 HTTP 版本检查失败会恢复前一个链接。Actions 会等待最终结果，服务重启中断的任务会标记为失败，需要重新运行 workflow。

`staging.asabook.cc` 用于验证同一个当前版本；它不是独立的预发布构建环境。静态服务只监听 `127.0.0.1:8787`，部署接口只监听 `127.0.0.1:8788`。Tunnel 按域名转发，其他域名返回 404。

## 初次安装与配置更新

首次准备仓库时使用完整 `git clone`，不要加 `--depth`。也可从完整 `git bundle --all` 克隆，再将 origin 设置为 GitHub 地址并 fetch。确认：

```sh
cd /data0/shared/dongwu.chen/asabook
git rev-parse --is-shallow-repository
git remote -v
```

第一个命令必须输出 `false`。安装 Node 24、Caddy、cloudflared 时使用官方发布文件并校验 SHA-256。复制 `tunnel.example.yml` 到 `~/.config/asabook/tunnel.yml`，填入 `cloudflared tunnel create` 返回的 UUID 和凭据路径；用 `cloudflared tunnel route dns` 配置对应域名。

部署密钥须随机生成至少 32 字符，写入 `deploy.env` 的 `ASABOOK_DEPLOY_TOKEN`，同时保存为 GitHub Actions 同名 secret。不要把密钥粘贴到日志或提交到仓库。

```sh
mkdir -p ~/.config/systemd/user
cp scripts/deploy/asabook-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now asabook-web asabook-deploy asabook-tunnel
```

修改 Caddy、部署 API 或服务文件后，等待当前部署结束，再复制更新的 unit 并运行：

```sh
systemctl --user daemon-reload
systemctl --user restart asabook-web asabook-deploy asabook-tunnel
```

## 回滚与存储

旧 release 和资源不会自动删除。先暂停自动发布或等部署结束，选择已验证的 release，通过下面脚本切换（参数是 `releases` 下的目录名）：

```sh
node scripts/deploy/rollback.mjs <release-directory>
```

脚本使用同一部署锁，检查 release 的版本信息和页面，再原子切换并验证 HTTP 版本；失败则恢复原版本。随后检查站点，再通过 Git revert 修复 `main`，避免下次自动部署重新发布问题版本。

定期检查 `du -sh /data0/shared/dongwu.chen/asabook-deploy`。删除历史 release 前确认它不是 `current` 指向的目录，并保留可回滚版本；跨版本 `assets` 还服务于用户已经打开的旧页面，不能随 release 一起立即删除。

静态 HTML 每次重新验证，`/assets/*` 使用一年 immutable 缓存。标题搜索保持原样；字体、论文内容和依赖兼容配置不因部署方式变化而调整。
