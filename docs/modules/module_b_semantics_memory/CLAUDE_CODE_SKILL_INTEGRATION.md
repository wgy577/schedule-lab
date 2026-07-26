# Claude Code 只读代码语义 Skill

> **所属模块**：B — 语义理解与知识记忆  
> **文档职责**：定义 Claude Code 与 Harness 统一只读证据路径。

## 目标

让 Opus 不只总结静态摘录，而是按受控流程追踪调度项目的入口、调用关系、环境、
约束、目标、决策与 Oracle；同时保持只读、有限预算、可追踪和证据可审计。

项目 Skill：

```text
.claude/skills/scheduling-code-semantics/
├── SKILL.md
├── agents/openai.yaml
└── references/review-checklist.md
```

Claude Code 官方支持项目级 `.claude/skills/<name>/SKILL.md`，直接调用使用
`/skill-name`。本项目通过 `/scheduling-code-semantics` 显式调用，避免依赖模型自行
判断是否触发。

## 从现有 Sortie Skill 迁移了什么

本机盘点到的项目级 Claude Skill 只有：

`Sortie/sortie code/comparision/.claude/skills/sortie3d-harness`

没有复制其 Blender、Three.js、舰载机、渲染资产或 797 行领域 SOP。只迁移以下通用
Harness 机制：

| 原机制 | 调度代码语义版本 |
|---|---|
| 场景规划后再分部件读取 | 先建 repository navigation，再读取语义锚点 |
| 渐进式披露 | 先入口/配置，再按调用关系读取关键函数 |
| choreography 单一契约 | 固定 JSON schema、不可变 batch ID 和枚举 |
| 几何真值优先 | 可达代码事实优先于文件名、评论、论文和模型推断 |
| verify/audit 双门 | Schema 校验 + evidence audit |
| Core/Working/Archival memory | 问题族知识、批次短期记忆、SQLite 长期记忆 |
| 每轮产物归档 | JSON Artifact + content-free JSONL 调用事件 |
| 有限 react loop | 固定批次、有限轮次、有限 schema repair、读预算和费用上限 |
| 独立复核 | 默认 Opus constraint-impact critic；影响审核 CLI 暂时禁止 DeepSeek |

对 `~/.claude/projects/**/*.jsonl` 的历史 `Skill` 工具调用做了只读扫描：当前只找到
一次明确记录，调用的是 Claude Code 内置 `/loop`，用于连续完成三个病理项目任务。
没有找到 `sortie3d-harness` 经 `Skill` 工具显式调用的历史记录；它可能作为项目上下文
被人工读取过，但不能据此声称“实际 Skill 调用”。`/loop` 是 Claude Code 内置 Skill，
不需要复制。本项目只借用“有终止条件的循环”思想，仍由自己的轮次、repair、读取和
费用上限控制，不直接启动无界 `/loop`。

## 权限边界

Skill 模式仅开放：

- `Read`
- `Glob`
- `Grep`

同时使用 Claude Code `--allowedTools` 预批准这些只读工具。禁止 Bash、Edit、Write、
网络、安装、训练和求解器执行；Skill 还明确禁止读取 `.env`、密钥、权重、大日志和
生成结果。

这些限制是双层的：

1. CLI 层根本不向模型提供写入和执行工具；
2. Skill 层规定读取范围、证据规则和有限复读条件。

## 证据流程

```text
Repository map
  → semantic anchors
  → caller/callee trace
  → evidence ledger
  → reachable-code check
  → fact classification
  → requested JSON schema
  → mechanical schema/evidence audit
```

每条判断区分：

- executable hard constraint；
- objective / secondary mechanism；
- controllable decision；
- conventional construction policy；
- training-only；
- evaluation/baseline-only；
- paper/comment claim；
- unknown/conflict。

## 调用审计

Skill 模式使用 Claude Code `stream-json`。Provider 只抽取工具名称，不保存工具参数、
文件正文或响应中间内容。事件日志记录：

- 配置的 Skill；
- 实际工具序列；
- 工具调用次数；
- provider/model/task/batch/round/repair；
- Token、耗时、费用、错误类型和推测原因。

因此可以区分：

- Skill 未发现；
- Skill 已发现但 Read 被权限层拒绝；
- Read/Grep/Glob 正常执行；
- 模型完成阅读但最终 JSON schema 失败。

## 真实探针

2026-07-25 对 `pyproject.toml` 做最小只读验证：

| 项目 | 结果 |
|---|---|
| Skill | `scheduling-code-semantics`，成功识别 |
| 实际工具 | `Read` |
| 读取范围 | 仅 `pyproject.toml` |
| 结果 | `causal-schedule-lab` / `0.5.0` |
| 轮次 | 2 |
| 客户端报告成本 | `$0.06351975` |

第一次探针使用 `$skill-name` 且没有 `--allowedTools`，结果为 Skill 未发现且 Read
被 `dontAsk` 拒绝。根据 Claude Code 规则修正为 `/skill-name` 并预批准只读工具后，
第二次通过。该失败已经证明权限层确实生效，不是形式上的只读声明。
