# 外部代码理解 Skill / Agent 能力调研

> **所属模块**：B — 语义理解与知识记忆  
> **文档职责**：记录可借鉴的外部代码理解模式及采用边界。

> 日期：2026-07-25  
> 目的：只借鉴成熟机制，不直接复制第三方 Skill 文本、脚本或领域结论。

## 1. Anthropic 官方 feature-dev / code-explorer

来源：

- <https://github.com/anthropics/claude-code/blob/main/plugins/feature-dev/agents/code-explorer.md>

可借鉴能力：

- 从 API、CLI、UI 等入口追踪到最终输出；
- 每一步记录数据转换、状态变化和依赖；
- 区分架构层与组件职责；
- 最后列出理解目标功能真正必要的文件。

本项目采用：

- `entrypoint → environment/solver → transition → objective/Oracle` 连续执行流；
- `minimum essential files`，避免把整个仓库清单当成核心上下文。

未采用：

- WebSearch、WebFetch、Bash 等宽权限；
- 面向功能开发的架构设计与代码修改阶段。

## 2. Trail of Bits audit-context-building

来源：

- <https://github.com/trailofbits/skills/tree/main/plugins/audit-context-building>

可借鉴能力：

- Orientation 与深度函数分析分阶段；
- 显式输入、隐式状态、假设、输出、副作用、不变量；
- 跨函数时保持同一条执行流，不在文件边界重置上下文；
- 新证据推翻旧理解时留下 correction；
- 定期形成稳定的上下文锚点。

本项目采用：

- 只对目标、约束、决策、环境转移和 Oracle 的核心函数做 micro-analysis；
- correction ledger 与短期记忆锚点。

未采用：

- 全仓库逐行分析；
- 每个函数固定数量的 5 Whys/5 Hows；
- 安全漏洞、攻击面和外部敌手模型。

原因：全量逐行模式 Token 成本过高，会把简单调度项目分析成安全审计；应只在
高影响函数上升级深度。

## 3. Trail of Bits fp-check

来源：

- <https://github.com/trailofbits/skills/tree/main/plugins/fp-check>

可借鉴能力：

- 模型结论必须通过独立证据门；
- 特别检查可达性、环境条件和数学上是否可能；
- 即使中间阶段失败也保留失败证据，不静默忽略。

本项目改写为六道语义门：

1. direct evidence；
2. runtime reachability；
3. instance/candidate applicability；
4. decision controllability；
5. primary/secondary objective relevance；
6. Oracle coverage。

这些门用于降级不可靠结论，不让 LLM 自己给自己的判断打满分。

## 4. Trail of Bits variant-analysis

来源：

- <https://github.com/trailofbits/skills/tree/main/plugins/variant-analysis>

可借鉴能力：

- 从一个已证明的精确实现开始；
- 一次只抽象一个名称、变量或结构元素；
- 每次扩大搜索后立即检查新增匹配；
- 可达性和上下文不同的匹配必须独立验证。

本项目采用：

- 找到一处 eligibility、capacity、precedence、mask、insert 或 validation 实现后，
  对同义函数和替代路径做 exact-to-general 搜索；
- 防止只读到训练环境中的一处约束，却漏掉生成器、修复器或最终 Oracle 的另一套实现。

暂不引入 Semgrep/CodeQL；当前先用 AST 索引 + Grep。跨语言或大型项目再升级工具。

## 5. Trailmark 代码图

来源：

- <https://github.com/trailofbits/skills/tree/main/plugins/trailmark>

可借鉴能力：

- 解析器生成调用图、模块依赖图和 entrypoint；
- 用复杂度、blast radius、data flow 等指标排序阅读热点；
- Mermaid 只负责展示解析结果，不让模型凭阅读手画调用图。

本项目当前已有：

- Python AST symbol/import/call 索引；
- Navigator 模块图；
- 受控 caller/callee 复读。

因此暂不安装 Trailmark。未来把代码图定义为可选 `CodeGraphTool`：

```text
AST builtin → default
Trailmark adapter → optional polyglot enhancement
Semgrep/CodeQL → optional semantic/data-flow enhancement
```

工具不存在时必须明确报告能力缺口，不允许假装完成跨语言数据流分析。

## 6. 采用后的读取流程

```text
快速导航
  → 找语义锚点
  → 完整执行流追踪
  → 仅核心函数 micro-analysis
  → 精确到泛化的约束变体搜索
  → 六道语义结论门
  → correction ledger
  → 固定 JSON + evidence audit
```

## 7. 安全与许可证边界

- 不整包安装第三方插件；
- 不复制其脚本、模板和长段落；
- 只用公开方法论重新设计本项目流程；
- Trail of Bits 仓库使用 CC-BY-SA-4.0，若未来直接引入其文件或衍生脚本，必须单独
  核查署名、相同方式共享和仓库发布边界；
- 任何外部 Skill 真正安装前先进行静态安全扫描和人工权限审查。
