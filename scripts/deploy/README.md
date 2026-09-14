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
git fetch --prune origin
git merge --ff-only <已验证的-origin/main-SHA>
npm ci --no-audit --no-fund
npm run paper:check-config
npm run docs:build -- --clean-cache
```

构建完成后复制到独立 release，再原子切换 `current`。构建失败不切换；切换后的 HTTP 版本检查失败会恢复前一个链接。Actions 会等待最终结果，服务重启中断的任务会标记为失败，需要重新运行 workflow。

`staging.asabook.cc` 用于验证同一个当前版本；它不是独立的预发布构建环境。静态服务只监听 `127.0.0.1:8787`，部署接口只监听 `127.0.0.1:8788`。Tunnel 按域名转发，其他域名返回 404。

## 初次安装与配置更新

### GitHub 连接故障

服务器直连 GitHub，不使用代理。拉取遇到 TLS 意外中断、连接超时等暂时性网络错误时，最多尝试四次，间隔为 10、20、30 秒；传输速度连续 60 秒低于 1 KiB/s 时结束当前尝试。认证、证书校验和仓库错误直接失败。Actions 会显示失败命令和最后 4 KiB 的标准错误输出，完整日志仍保存在服务器。

拉取成功后只对已验证的提交执行本地 fast-forward，避免 `pull` 再次建立网络连接。不要关闭证书校验，也不要将 GitHub 的某个 IP 永久写入 hosts；临时节点故障应结合 DNS、TLS 和 Git 实际传输日志排查。

2026-09-14 排查记录：默认解析的 `20.205.243.166` 曾发生 TCP 443 连接超时，随后同一地址的 TLS、HTTP/1.1、HTTP/2 和 Git 引用查询均恢复；替代 GitHub 节点的证书校验及引用查询也成功，但完整拉取仍发生低速超时，强制 HTTP/1.1 也未解决。证据指向直连路径的间歇性故障，未能确定具体断连设备。原任务日志没有 TLS 跟踪，不能仅凭 `GnuTLS recv error (-110)` 判定为证书、协议版本或 DNS 污染问题。

若持续无法下载缺失对象，可按下方完整 bundle 的安装方式，经可信内网传输增量 bundle，在服务器运行 `git bundle verify` 后用 `git fetch <bundle-path> main` 导入对象，再让部署流程从 GitHub 验证最新 `main`。这属于故障恢复，不替代日常直连拉取。

### 安装步骤

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
