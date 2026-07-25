# 项目适配指南

## 1. 适配器职责

一个项目适配器必须输出稳定的 canonical `Problem` 和 incumbent `Schedule`：

- 作业、工序与稳定 ID；
- 工序优先关系；
- 可选模式、处理时间和所需资源；
- 资源容量；
- release、due date；
- setup、日历、blocking、no-wait、路径和时间窗等已建模约束；
- 未建模但必须由 Oracle 检查的领域约束；
- 项目目标的顺序、方向和容差。

不得为了接入而静默丢弃约束。

## 2. 自定义来源适配器

```python
class MyAdapter:
    def load(self, manifest, *, manifest_path):
        problem = ...
        incumbent = ...
        return problem, incumbent
```

在 manifest 中使用：

```json
{"adapter": "my_package.adapters:MyAdapter"}
```

适配器可以读取数据库、Excel、MES、求解器输出或项目自有对象。核心框架不关心
原始来源，只要求输出统一 IR 并保留 source ID。

## 3. 项目语义

项目语义 DSL 声明：

- 问题族；
- 资源容量与角色；
- 绑定关系；
- 硬约束；
- 允许的干预；
- Oracle 门；
- 有代码、测试或文档证据的 verified 声明；
- 特定诊断对应的责任阶段。

LLM 或 Agent 可以提出 DSL 草案，但只有证据审计通过的事实才能成为
`verified`。没有证据的内容应标成 `unknown`。

## 4. 条件生成器

生成器实现：

```python
def generate(*, problem, incumbent, cip, proposal) -> GeneratedCandidate:
    ...
```

必须遵守：

- 只开放 `proposal.released_operations`；
- 固定 `proposal.frozen_operations`；
- 使用固定 seed、单 worker 和确定性预算；
- 不伪造可行性；
- 领域约束未验证时标记 provisional。

默认 `generic-cp-sat` 适合统一 IR 已完整表达的 JSP/FSP/FJSP/HFSP。setup、
运输或自定义主目标如果不能被内置 CP-SAT 完整编码，可以提供
`module:function` 工厂：

```json
{"generator": "my_package.repair:build_generator"}
```

## 5. 领域 Oracle

Oracle 接口：

```python
def validate(problem, incumbent, candidate):
    return {
        "passed": True,
        "failureLabels": [],
        "details": {}
    }
```

工厂在 manifest 中配置：

```json
{"domain_oracle": "my_package.oracle:build_oracle"}
```

一旦配置，完整验证阶段必须调用它，即使 `Problem.metadata` 没有额外标志。
Oracle 可以检查仿真状态、车辆连续性、路径、碰撞、能耗、日历或外部系统规则。

## 6. 自定义目标

简单项目直接在 `Problem.objective` 中按词典序声明
`ObjectiveComponent(name, sense, tolerance, weight)`。

复杂项目声明：

```json
{"objective_evaluator": "my_package.objectives:evaluate"}
```

函数签名：

```python
def evaluate(problem, schedule, baseline) -> dict[str, float]:
    ...
```

候选与 incumbent 必须返回相同指标集合。核心接受器按词典序比较，第一个超过
容差的分量决定优劣。

## 7. 接入验收矩阵

新项目至少增加以下测试：

1. problem/schedule ID 与数量稳定；
2. incumbent 通过项目现有验证器；
3. 每个硬约束有统一 IR 或 Oracle 落点；
4. 目标向量和容差符合业务意图；
5. CIP 输出包含诊断、责任、路径和闭包；
6. 闭包外任意 start/end/mode 变化都会被拒绝；
7. 非法候选被通用验证拒绝；
8. 领域候选在 Oracle 前保持 provisional；
9. 固定 seed 两次运行哈希一致；
10. 只有完整目标严格改善才更新 incumbent。
