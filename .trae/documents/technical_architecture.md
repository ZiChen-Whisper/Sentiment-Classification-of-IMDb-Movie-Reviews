## 1. 架构设计

```mermaid
flowchart TB
    "前端（单页HTML）" --> "CSS动画引擎"
    "前端（单页HTML）" --> "JS翻页控制器"
    "前端（单页HTML）" --> "内嵌数据可视化"
```

纯前端单文件方案，无需后端服务和数据库。

## 2. 技术说明

- 前端：纯HTML5 + CSS3 + Vanilla JavaScript
- 无需构建工具、无需npm、无需框架
- 动画：CSS3 Transitions + Keyframe Animations + 少量JS控制
- 数据可视化：纯CSS/SVG实现（柱状图、流程图等）
- 字体：Google Fonts CDN加载 Noto Serif SC + Noto Sans SC
- 图标：内联SVG

## 3. 路由定义

无路由，单页面内通过JS控制幻灯片切换。

## 4. 文件结构

```
output/
  defense_presentation.html   # 单文件演示页面
```

## 5. 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 技术栈 | 纯HTML/CSS/JS | 便携性，双击即可打开，无需服务器 |
| 动画方案 | CSS3优先 | 性能好，GPU加速，代码简洁 |
| 翻页效果 | 3D滑动+淡入淡出 | 视觉冲击力强，流畅自然 |
| 数据可视化 | 纯CSS/SVG | 无需引入Chart.js等库，保持单文件 |
| 字体加载 | Google Fonts CDN | 中文衬线/无衬线字体搭配 |
