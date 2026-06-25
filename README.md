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
