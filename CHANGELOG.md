# Changelog

本项目的重要变化记录在此。后续更新遵循“能力变化、正式实验结论、接口变化、验证规则变化”四类记录；纯缓存和临时输出不进入日志。

## Unreleased

### Planned

- 在 627.8 秒 incumbent 上比较 causal closure、local branching 与 shifting bottleneck。
- 统一轨迹增强 Tool 的可插拔输入输出契约。
- 扩充 JSP、FSP、FJSP、HFSP 多实例回归集。
- 完善 Oracle 缓存、Cut 复用和结构化实验存储。

## 0.1.1 — 2026-07-20

### Documentation

- 将 README 重构为标准 GitHub 项目结构：徽章、目录、功能、结果、安装、使用、架构、文档、路线图、贡献和许可。
- 增加 `README_EN.md` 完整英文版，并在中英文 README 顶部提供语言切换。
- 增加当前结果、成熟度、快速开始、项目结构和维护规范。
- 增加 `EXPERIMENTS.md`，统一记录接受、拒绝和计划实验。
- 增加本 Changelog，作为后续版本更新入口。

## 0.1.0 — 2026-07-20

### Added

- 建立 JSP、FSP、FJSP、HFSP 的统一调度表示和适配器。
- 接入派工启发式、PyJobShop、OR-Tools CP-SAT 和通用验证器。
- 建立 fast → balanced 自适应 incumbent-improvement 工作流。
- 加入 VNS、有界 ALNS、因果闭包、Tabu 去重和贝叶斯证据排序。
- 接入舰载机旧领域轨迹/碰撞 Oracle。
- 建立固定 MAT 路线目录、离散时空求解和 Oracle Cut 原型。
- 保存 675.5 → 627.8 秒的已验证舰载机优化链。
- 增加甘特图、同时间轴对比视频、CLI、MCP Server 和项目 Skill。
- 增加 `SCHEDULE_LAB_PLAN.md` 技术计划与长期路线。
