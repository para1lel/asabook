# ASa Book 部署

站点在 `192.168.80.152`（`node13`，用户 `dongwu.chen`）自动同步 GitHub 并构建，Cloudflare Tunnel 从本机直接连接公网。`asabook.cc` 保留路径和查询参数，以 308 跳转到 `https://www.asabook.cc`。

## 目录与服务

所有软件、配置、缓存与发布版本都在 `/data1/shared/dongwu.chen`：

| 路径 | 用途 |
| --- | --- |
| `asabook` | `main` 分支完整历史仓库 |
| `software/node24` | 独立的 Node.js 24 运行时 |
| `bin/caddy`、`bin/cloudflared` | 官方二进制的符号链接 |
| `state/asabook/releases` | 独立的静态版本 |
| `state/asabook/current` | 当前版本的符号链接 |
| `state/asabook/assets` | 跨版本保留的构建静态资源 |
| `state/asabook/jobs`、`state/asabook/logs` | 部署任务状态与构建日志 |
| `config/cloudflared/asabook.yml` | Tunnel 配置 |
| `config/cloudflared/<UUID>.json` | 此 Tunnel 的凭据，权限 `600` |
| `config/systemd/user/asabook-*` | 服务与定时器文件 |

`~/.config/systemd/user` 仅保存 unit 的符号链接。用户启用了 linger，退出 SSH 后仍运行，并随机器启动。

```sh
systemctl --user status asabook-web asabook-tunnel asabook-deploy.timer
systemctl --user list-timers asabook-deploy.timer
journalctl --user -u asabook-deploy -n 100 --no-pager
```

## 自动发布

GitHub Actions workflow 已移除。`asabook-deploy.timer` 启动后首次等待 45 秒，此后每次同步结束两分钟后检查，附加最多 15 秒的随机延迟。同步直接从 GitHub 拉取，不依赖其他内网服务器、部署 API 或 GitHub Actions secret。

同步使用 `flock` 串行执行，拒绝浅克隆、脏工作区、错误 origin 或非 `main` 分支。已发布版本与远端 `main` 相同时不重建。新提交运行：

```sh
git fetch --prune origin
git merge --ff-only <已验证的-origin/main-SHA>
npm ci --no-audit --no-fund
npm run paper:check-config
npm run docs:build -- --clean-cache
```

构建完成后复制到独立 release，再原子切换 `current`。构建失败不切换；切换后的 HTTP 版本检查失败会恢复前一个链接。服务中断后，定时器下次检查会重新尝试尚未成功发布的版本。任务详情和完整输出保存在 `jobs` 和 `logs`。

手动立即检查并发布：

```sh
systemctl --user start asabook-deploy.service
```

静态服务只监听 `127.0.0.1:8787`。Tunnel 仅发布 `asabook.cc`、`www.asabook.cc` 和 `ssh.asabook.cc`；SSH 转到服务器实际监听的 `127.0.0.1:22`，沿用现有 Cloudflare Access 策略。本机通过 LAN 地址连接 152 时仍使用外部映射端口 `54321`。其他域名返回 404。旧部署 API 和 trigger 脚本保留供历史参考，152 不运行 API、不开放 `8788`。

GitHub 拉取优先直连；暂时性 TLS、连接和低速错误最多尝试四次，间隔 10、20、30 秒。传输速度连续 30 秒低于 64 KiB/s 时结束当前尝试。重试使用 `ASABOOK_GIT_PROXY` 指定的 152 本机 Mihomo 回退端口 `127.0.0.1:7891`，仅在直连操作失败后使用香港 x0.8 节点。常规代理端口 `7890` 仍按规则直连优先。认证、证书校验和仓库错误直接失败。不要关闭证书校验或固定 GitHub IP。同步和 Tunnel 服务清除通用代理变量，Tunnel 独立直连公网。

## 安装与配置更新

首次使用完整 `git clone https://github.com/para1lel/asabook.git`，不要加 `--depth`。确认：

```sh
cd /data1/shared/dongwu.chen/asabook
git rev-parse --is-shallow-repository
git remote -v
```

第一个命令必须输出 `false`。Node.js 24、Caddy、cloudflared 使用官方发布文件并校验 SHA-256。复制 `tunnel.example.yml` 到 `config/cloudflared/asabook.yml`，填入实际 UUID 和凭据路径。复用原 Tunnel 时须停用旧 connector，避免请求被分配到旧源站。

复制 unit 到 `config/systemd/user/`，并在 systemd 用户目录创建同名符号链接：

```sh
base=/data1/shared/dongwu.chen
mkdir -p "$base/config/systemd/user" "$HOME/.config/systemd/user"
cp scripts/deploy/asabook-*.service scripts/deploy/asabook-*.timer "$base/config/systemd/user/"
for unit in "$base"/config/systemd/user/asabook-*; do
  ln -sfn "$unit" "$HOME/.config/systemd/user/$(basename "$unit")"
done
systemctl --user daemon-reload
systemctl --user enable --now asabook-web asabook-tunnel asabook-deploy.timer
```

变更脚本和 Caddyfile 会随仓库同步。变更已安装的 unit 时，等待部署结束，重新复制 unit，再执行 `daemon-reload` 并重启对应服务或定时器。

## 回滚与存储

先停止定时器并等待当前构建结束，再选择已验证的 release：

```sh
systemctl --user stop asabook-deploy.timer
systemctl --user is-active asabook-deploy.service
cd /data1/shared/dongwu.chen/asabook
/data1/shared/dongwu.chen/software/node24/bin/node scripts/deploy/rollback.mjs <release-directory>
```

回滚脚本使用同一锁，检查 release 元数据和页面，再原子切换并验证 HTTP 版本；失败则恢复原版本。通过 Git revert 修复并推送 `main`，再启动 `asabook-deploy.timer`，否则下次检查会重新发布远端当前版本。

旧 release 和资源不会自动删除。定期检查 `du -sh /data1/shared/dongwu.chen/state/asabook`；删除前确认不是 `current`，并保留回滚版本。跨版本 `assets` 服务于用户已打开的旧页面，不能随 release 立即删除。

静态 HTML 每次重新验证，`/assets/*` 使用一年 immutable 缓存。标题搜索、字体、论文内容和依赖兼容配置保持现有设置。
