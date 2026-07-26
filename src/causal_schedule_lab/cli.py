from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from .agent import MaskedPPOAgent
from .audit import load_records
from .cip import CausalCoreDiscoverer
from .constraint_impact import assess_constraint_impacts_with_llm
from .controller import AgenticImprovementController
from .dataset import pairwise_ranking_labels, record_to_training_row
from .experiments import build_control_plan
from .graph import build_scheduling_graph
from .llm_semantics import (
    RepositoryEvidencePacket,
    SemanticAnalysis,
    build_repository_evidence_packet,
    choice_library,
    compile_project_semantics_with_llm,
)
from .models import AgentAction, ControlAction
from .mechanisms import measure_mechanisms
from .operators import ActionIndex
from .objective import evaluate_objective
from .plugins import build_domain_oracle, build_generator
from .project import ProjectContext, load_project
from .semantics import audit_semantic_evidence, load_semantics
from .semantic_compiler import compile_project_semantics
from .semantic_agent import (
    build_navigation_inventory,
    compile_project_semantics_staged,
)
from .semantic_harness import (
    SemanticHarnessResult,
    run_blind_harness,
    run_code_only_harness,
)
from .semantic_knowledge import SchedulingKnowledgeBase
from .storage import HierarchicalMemory
from .providers.anthropic_compatible import (
    AnthropicCompatibleProvider,
    load_anthropic_configuration,
)
from .providers.claude_code_cli import (
    ClaudeCodeCLIProvider,
    load_claude_code_configuration,
)
from .providers.openai_compatible import (
    OpenAICompatibleProvider,
    load_provider_configuration,
)
from .providers.tracing import LLMRunTrace
from .validation import MultiFidelityValidator, schedule_hash


def _lab_root() -> Path:
    configured = os.environ.get("CAUSAL_SCHEDULE_LAB_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / "configs").is_dir() and (current / "examples").is_dir():
        return current
    source_candidate = Path(__file__).resolve().parents[2]
    if (source_candidate / "configs").is_dir():
        return source_candidate
    return current


LAB_ROOT = _lab_root()


def _write(payload: Any, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def validate_impact_review_route(protocol: str, provider_prefix: str) -> None:
    """Temporarily prevent a conservative model from gating metric discovery."""

    if protocol == "openai" and provider_prefix.strip().upper() == "DEEPSEEK":
        raise ValueError(
            "DeepSeek is temporarily disabled for constraint-impact review; "
            "use Opus via anthropic or claude-code"
        )


def _evidence_root(context: ProjectContext) -> Path:
    configured = context.manifest.metadata.get("evidence_root")
    if configured is None:
        return context.project_root
    path = Path(str(configured)).expanduser()
    return path.resolve() if path.is_absolute() else (context.project_root / path).resolve()


def _discover(context: ProjectContext, *, top_k: int):
    graph = build_scheduling_graph(
        context.problem,
        context.incumbent,
        project_id=context.manifest.project_id,
    )
    discoverer = CausalCoreDiscoverer(context.semantics)
    candidates = discoverer.discover(
        problem=context.problem,
        schedule=context.incumbent,
        graph=graph,
        top_k=top_k,
    )
    return graph, discoverer, candidates


def _policy(context: ProjectContext, candidates, *, epochs: int):
    action_index = ActionIndex.from_semantics(context.semantics)
    policy = MaskedPPOAgent(action_index, seed=0)
    demonstrations = [
        (
            candidate,
            AgentAction(
                operator=candidate.recommended_operators[0],
                closure_level=candidate.closure.level,
                control=ControlAction.FULL_ORACLE,
            ),
        )
        for candidate in candidates
        if candidate.recommended_operators
    ]
    losses = policy.behavior_clone(demonstrations, epochs=epochs)
    return action_index, policy, losses


def _project_summary(context: ProjectContext) -> dict[str, Any]:
    objective = evaluate_objective(
        context.problem,
        context.incumbent,
        baseline=context.incumbent,
    )
    return {
        "projectId": context.manifest.project_id,
        "problemId": context.problem.id,
        "problemFamily": context.problem.kind,
        "adapter": context.manifest.adapter,
        "generator": context.manifest.generator,
        "domainOracle": context.manifest.domain_oracle,
        "operationCount": len(context.problem.operations),
        "resourceCount": len(context.problem.resources),
        "incumbentObjective": objective.as_dict(),
        "incumbentHash": schedule_hash(context.incumbent),
    }


def cmd_audit_semantics(args: argparse.Namespace) -> None:
    semantics = load_semantics(args.semantics)
    result = audit_semantic_evidence(
        semantics,
        project_root=args.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        _write(result, args.output)
    if not result["passed"]:
        raise SystemExit(2)


def cmd_compile_semantics(args: argparse.Namespace) -> None:
    result = compile_project_semantics(args.project_root)
    payload = result.model_dump(mode="json")
    output = args.output or (
        LAB_ROOT / "outputs" / "semantic_compilation.json"
    )
    print(f"wrote {_write(payload, output)}")


def cmd_compile_semantics_llm(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).expanduser().resolve()
    output = args.output or (
        LAB_ROOT / "outputs" / "llm_semantic_compilation.json"
    )
    if args.dry_run:
        packet = build_repository_evidence_packet(
            project_root,
            max_characters=args.max_input_characters,
        )
        knowledge = SchedulingKnowledgeBase.load(args.knowledge_base)
        packet_text = "\n".join(
            f"{item.path}\n{item.excerpt}" for item in packet.files
        )
        payload = {
            "mode": "dry-run",
            "packet": packet.model_dump(mode="json"),
            "knowledgeHits": [
                item.model_dump(mode="json")
                for item in knowledge.retrieve(packet_text, top_k=2)
            ],
            "engineeringPatternHits": [
                item.model_dump(mode="json")
                for item in knowledge.retrieve_engineering_patterns(
                    packet_text,
                    top_k=8,
                )
            ],
            "choiceLibrary": choice_library(),
            "apiCalled": False,
        }
        print(f"wrote {_write(payload, output)}")
        return
    configuration = load_provider_configuration(
        env_file=args.env_file,
        prefix=args.provider_prefix,
        timeout_seconds=args.timeout,
        max_attempts=args.max_attempts,
    )
    provider = OpenAICompatibleProvider(configuration)
    impact_provider = None
    if args.assess_constraint_impact:
        validate_impact_review_route(
            args.impact_protocol,
            args.impact_provider_prefix,
        )
        if args.impact_protocol == "anthropic":
            impact_provider = AnthropicCompatibleProvider(
                load_anthropic_configuration(
                    env_file=args.env_file,
                    timeout_seconds=args.timeout,
                    max_attempts=args.max_attempts,
                ),
                provider_name="constraint-impact-critic",
            )
        else:
            impact_provider = OpenAICompatibleProvider(
                load_provider_configuration(
                    env_file=args.env_file,
                    prefix=args.impact_provider_prefix,
                    timeout_seconds=args.timeout,
                    max_attempts=args.max_attempts,
                ),
                provider_name="constraint-impact-critic",
            )
    result = compile_project_semantics_with_llm(
        project_root,
        provider=provider,
        max_input_characters=args.max_input_characters,
        max_output_tokens=args.max_output_tokens,
        max_schema_repairs=args.max_schema_repairs,
        knowledge_base_path=args.knowledge_base,
        impact_provider=impact_provider,
        impact_max_output_tokens=args.impact_max_output_tokens,
    )
    payload = result.model_dump(mode="json")
    target = _write(payload, output)
    print(
        f"wrote {target}; provider={result.metadata.provider}; "
        f"requested_model={result.metadata.requested_model}; "
        f"response_model={result.metadata.response_model}; "
        f"evidence_pass_rate={result.evidence_pass_rate:.3f}; "
        f"review_status={result.review_status}; "
        f"knowledge_hits={','.join(item.family for item in result.knowledge_hits) or 'none'}; "
        f"impact_critic={result.constraint_impact.response_model if result.constraint_impact else 'not-run'}"
    )


def cmd_compile_semantics_staged(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).expanduser().resolve()
    output = args.output or (
        LAB_ROOT / "outputs" / "staged_semantic_compilation.json"
    )
    if args.dry_run:
        inventory = build_navigation_inventory(project_root)
        payload = {
            "mode": "dry-run",
            "inventory": inventory.model_dump(mode="json"),
            "navigatorApiCalled": False,
            "analystApiCalled": False,
        }
        print(f"wrote {_write(payload, output)}")
        return

    trace_path = args.event_log or f"{output}.events.jsonl"
    trace = LLMRunTrace(trace_path)
    navigator_base = OpenAICompatibleProvider(
        load_provider_configuration(
            env_file=args.env_file,
            prefix=args.navigator_provider_prefix,
            timeout_seconds=args.timeout,
            max_attempts=args.max_attempts,
        ),
        provider_name="semantic-navigator",
    )
    navigator = trace.wrap(navigator_base, label="navigator")
    if args.analyst_protocol == "claude-code":
        analyst_base = ClaudeCodeCLIProvider(
            load_claude_code_configuration(
                env_file=args.env_file,
                timeout_seconds=args.timeout,
                max_budget_usd=args.claude_code_max_budget_usd,
                base_url_override=args.claude_code_base_url,
                enable_project_skill=args.claude_code_project_skill,
                enable_read_tools=args.claude_code_read_mode == "claude",
                skill_name=args.claude_code_skill_name,
                read_budget=args.claude_code_read_budget,
            ),
            provider_name="semantic-analyst-opus",
        )
    elif args.analyst_protocol == "anthropic":
        analyst_base = AnthropicCompatibleProvider(
            load_anthropic_configuration(
                env_file=args.env_file,
                timeout_seconds=args.timeout,
                max_attempts=args.max_attempts,
            ),
            provider_name="semantic-analyst",
        )
    else:
        analyst_base = OpenAICompatibleProvider(
            load_provider_configuration(
                env_file=args.env_file,
                prefix=args.analyst_provider_prefix,
                timeout_seconds=args.timeout,
                max_attempts=args.max_attempts,
            ),
            provider_name="semantic-analyst",
        )
    analyst = trace.wrap(analyst_base, label="analyst")
    impact_provider = None
    if args.assess_constraint_impact:
        validate_impact_review_route(
            args.impact_protocol,
            args.impact_provider_prefix,
        )
        if args.impact_protocol == "claude-code":
            impact_base = ClaudeCodeCLIProvider(
                load_claude_code_configuration(
                    env_file=args.env_file,
                    timeout_seconds=args.timeout,
                    max_budget_usd=args.claude_code_max_budget_usd,
                    base_url_override=args.claude_code_base_url,
                    enable_project_skill=args.claude_code_project_skill,
                    enable_read_tools=args.claude_code_read_mode == "claude",
                    skill_name=args.claude_code_skill_name,
                    read_budget=args.claude_code_read_budget,
                ),
                provider_name="staged-constraint-impact-opus",
            )
        elif args.impact_protocol == "anthropic":
            impact_base = AnthropicCompatibleProvider(
                load_anthropic_configuration(
                    env_file=args.env_file,
                    timeout_seconds=args.timeout,
                    max_attempts=args.max_attempts,
                ),
                provider_name="staged-constraint-impact-critic",
            )
        else:
            impact_base = OpenAICompatibleProvider(
                load_provider_configuration(
                    env_file=args.env_file,
                    prefix=args.impact_provider_prefix,
                    timeout_seconds=args.timeout,
                    max_attempts=args.max_attempts,
                ),
                provider_name="staged-constraint-impact-critic",
            )
        impact_provider = trace.wrap(impact_base, label="impact")
    print(
        f"LLM event log: {Path(trace_path).expanduser().resolve()}",
        flush=True,
    )
    result = compile_project_semantics_staged(
        project_root,
        navigator_provider=navigator,
        analyst_provider=analyst,
        max_rounds_per_batch=args.max_rounds_per_batch,
        max_reads=args.max_reads,
        max_output_tokens=args.max_output_tokens,
        max_schema_repairs=args.max_schema_repairs,
        knowledge_base_path=args.knowledge_base,
        memory_database=args.memory_database,
        impact_provider=impact_provider,
        impact_max_output_tokens=args.impact_max_output_tokens,
        analyst_reasoning_effort=args.analyst_reasoning_effort,
        impact_reasoning_effort=args.impact_reasoning_effort,
    )
    target = _write(result.model_dump(mode="json"), output)
    print(
        f"wrote {target}; navigator={result.metadata.navigator_model}; "
        f"analyst={result.metadata.analyst_model}; "
        f"approved_reads={result.metadata.approved_reads}; "
        f"rejected_reads={result.metadata.rejected_reads}; "
        f"evidence_pass_rate={result.final.evidence_pass_rate:.3f}; "
        f"review_status={result.review_status}; event_log={trace_path}"
    )


def cmd_review_constraint_impact(args: argparse.Namespace) -> None:
    """Re-run only the impact/metric critic over a saved semantic extraction."""

    source = Path(args.semantic_compilation).expanduser().resolve()
    saved = json.loads(source.read_text(encoding="utf-8"))
    analysis = SemanticAnalysis.model_validate(saved["final"]["analysis"])
    packet = RepositoryEvidencePacket.model_validate(saved["final"]["packet"])
    source_semantic_tokens = int(saved.get("metadata", {}).get("total_tokens", 0))
    validate_impact_review_route(args.protocol, args.provider_prefix)
    if args.protocol == "claude-code":
        base_provider = ClaudeCodeCLIProvider(
            load_claude_code_configuration(
                env_file=args.env_file,
                timeout_seconds=args.timeout,
                max_budget_usd=args.claude_code_max_budget_usd,
                base_url_override=args.claude_code_base_url,
                enable_project_skill=False,
                enable_read_tools=False,
            ),
            provider_name="replay-constraint-impact-opus",
        )
    elif args.protocol == "anthropic":
        base_provider = AnthropicCompatibleProvider(
            load_anthropic_configuration(
                env_file=args.env_file,
                timeout_seconds=args.timeout,
                max_attempts=args.max_attempts,
            ),
            provider_name="replay-constraint-impact-critic",
        )
    else:
        base_provider = OpenAICompatibleProvider(
            load_provider_configuration(
                env_file=args.env_file,
                prefix=args.provider_prefix,
                timeout_seconds=args.timeout,
                max_attempts=args.max_attempts,
            ),
            provider_name="replay-constraint-impact-critic",
        )
    trace_path = args.event_log or f"{args.output}.events.jsonl"
    provider = LLMRunTrace(trace_path).wrap(base_provider, label="impact-replay")
    knowledge = SchedulingKnowledgeBase.load(args.knowledge_base)
    knowledge_hits, engineering_hits = knowledge.retrieve_conditioned_on_analysis(
        analysis,
        top_k=4,
    )
    report = assess_constraint_impacts_with_llm(
        analysis,
        knowledge_hits,
        provider=provider,
        engineering_hits=engineering_hits,
        max_output_tokens=args.max_output_tokens,
        thinking_mode="enabled",
        reasoning_effort=args.reasoning_effort,
    )
    payload = {
        "schema_version": "1.0",
        "comparison_mode": "same_saved_semantics_impact_replay",
        "source_semantic_compilation": str(source),
        "source_semantic_tokens": source_semantic_tokens,
        "report": report.model_dump(mode="json"),
    }
    target = _write(payload, args.output)
    print(
        f"wrote {target}; model={report.response_model or report.requested_model}; "
        f"options={len(report.metric_recall.options)}; "
        f"selected={len(report.secondary_targets)}; "
        f"critic_tokens={report.usage.total_tokens}; event_log={trace_path}"
    )


def cmd_run_semantic_harness(args: argparse.Namespace) -> None:
    configuration = load_provider_configuration(
        env_file=args.env_file,
        prefix=args.provider_prefix,
        timeout_seconds=args.timeout,
        max_attempts=args.max_attempts,
    )
    provider = OpenAICompatibleProvider(
        configuration,
        provider_name=args.provider_prefix.lower(),
    )
    result = run_blind_harness(
        args.case,
        provider=provider,
        max_paper_characters=args.max_paper_characters,
        max_code_characters=args.max_code_characters,
    )
    output = args.output or (
        LAB_ROOT / "outputs" / f"{result.case.case_id}_semantic_harness.json"
    )
    target = _write(result.model_dump(mode="json"), output)
    print(
        f"wrote {target}; case={result.case.case_id}; blind=true; "
        f"label_status={result.label.label_status}; "
        f"macro_f1={result.evaluation.macro_f1:.3f}; "
        f"exact_match_rate={result.evaluation.exact_match_rate:.3f}; "
        f"code_evidence_pass_rate={result.evaluation.evidence_pass_rate:.3f}"
    )


def _benchmark_provider(args: argparse.Namespace):
    if args.protocol == "anthropic":
        configuration = load_anthropic_configuration(
            env_file=args.env_file,
            timeout_seconds=args.timeout,
            max_attempts=args.max_attempts,
        )
        return AnthropicCompatibleProvider(
            configuration,
            provider_name="anthropic",
        )
    configuration = load_provider_configuration(
        env_file=args.env_file,
        prefix=args.provider_prefix,
        timeout_seconds=args.timeout,
        max_attempts=args.max_attempts,
    )
    return OpenAICompatibleProvider(
        configuration,
        provider_name=args.provider_prefix.lower(),
    )


def cmd_run_code_semantic_benchmark(args: argparse.Namespace) -> None:
    source = Path(args.label_artifact).expanduser().resolve()
    baseline = SemanticHarnessResult.model_validate_json(
        source.read_text(encoding="utf-8")
    )
    result = run_code_only_harness(
        args.case,
        label=baseline.label,
        provider=_benchmark_provider(args),
        max_code_characters=args.max_code_characters,
    )
    output = args.output or (
        LAB_ROOT
        / "outputs"
        / f"{result.case.case_id}_{args.provider_prefix.lower()}_benchmark.json"
    )
    target = _write(result.model_dump(mode="json"), output)
    print(
        f"wrote {target}; provider={result.prediction.metadata.provider}; "
        f"model={result.prediction.metadata.response_model or result.prediction.metadata.requested_model}; "
        f"macro_f1={result.evaluation.macro_f1:.3f}; "
        f"exact_match_rate={result.evaluation.exact_match_rate:.3f}; "
        f"evidence_pass_rate={result.evaluation.evidence_pass_rate:.3f}"
    )


def cmd_inspect(args: argparse.Namespace) -> None:
    context = load_project(args.project)
    audit = audit_semantic_evidence(
        context.semantics,
        project_root=_evidence_root(context),
    )
    graph, _, candidates = _discover(context, top_k=args.top_k)
    action_index, policy, losses = _policy(
        context,
        candidates,
        epochs=args.bc_epochs,
    )
    actions = []
    for candidate in candidates:
        action, policy_info = policy.choose(candidate, deterministic=True)
        actions.append(
            {
                "cip": candidate.id,
                "action": action.model_dump(mode="json"),
                "policy": policy_info,
                "legalMask": action_index.mask_for(candidate),
            }
        )
    payload = {
        "framework": "project-conditioned-causal-agentic-scheduling",
        "implementationStage": "research-platform",
        "project": _project_summary(context),
        "semanticsAudit": audit,
        "graph": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "causalSkeleton": graph.metadata["sharedCausalSkeleton"],
        },
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "agentActions": actions,
        "behaviorCloning": {
            "epochs": args.bc_epochs,
            "initialLoss": losses[0] if losses else None,
            "finalLoss": losses[-1] if losses else None,
        },
        "causalControls": build_control_plan(candidates, seed=0),
        "oracleExecuted": False,
        "incumbentUnchanged": True,
    }
    output = args.output or (
        LAB_ROOT
        / "outputs"
        / f"{context.manifest.project_id}_{context.problem.id}_inspection.json"
    )
    print(f"wrote {_write(payload, output)}")


def cmd_optimize(args: argparse.Namespace) -> None:
    context = load_project(args.project)
    _, discoverer, candidates = _discover(context, top_k=args.top_k)
    action_index, policy, losses = _policy(
        context,
        candidates,
        epochs=args.bc_epochs,
    )
    validator = MultiFidelityValidator(
        semantics=context.semantics,
        project_root=str(_evidence_root(context)),
        domain_oracle=build_domain_oracle(context),
    )
    controller = AgenticImprovementController(
        project_id=context.manifest.project_id,
        discoverer=discoverer,
        action_index=action_index,
        policy=policy,
        generator=build_generator(context),
        validator=validator,
        audit_log=args.audit_log,
    )
    result = controller.run(
        problem=context.problem,
        incumbent=context.incumbent,
        max_iterations=args.iterations,
        top_k=args.top_k,
        candidate_budget=args.candidate_budget,
        full_oracle_budget=args.full_oracle_budget,
    )
    initial_objective = evaluate_objective(
        context.problem,
        context.incumbent,
        baseline=context.incumbent,
    )
    best_objective = evaluate_objective(
        context.problem,
        result.best_schedule,
        baseline=context.incumbent,
    )
    payload = {
        "framework": "project-conditioned-causal-agentic-scheduling",
        "project": _project_summary(context),
        "initialObjective": initial_objective.as_dict(),
        "bestObjective": best_objective.as_dict(),
        "bestHash": schedule_hash(result.best_schedule),
        "stopReason": result.stop_reason,
        "behaviorCloningFinalLoss": losses[-1] if losses else None,
        "records": [record.model_dump(mode="json") for record in result.records],
        "bestSchedule": result.best_schedule.model_dump(mode="json"),
        "sourceIncumbentUnchanged": True,
    }
    output = args.output or (
        LAB_ROOT
        / "outputs"
        / f"{context.manifest.project_id}_{context.problem.id}_optimization.json"
    )
    print(f"wrote {_write(payload, output)}")


def cmd_demo(args: argparse.Namespace) -> None:
    manifest = LAB_ROOT / "examples" / "manifests" / f"{args.family}.json"
    args.project = str(manifest)
    if args.mode == "inspect":
        cmd_inspect(args)
    else:
        cmd_optimize(args)


def cmd_train_policy(args: argparse.Namespace) -> None:
    context = load_project(args.project)
    _, _, candidates = _discover(context, top_k=args.top_k)
    action_index, policy, losses = _policy(
        context,
        candidates,
        epochs=args.epochs,
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.model.state_dict(),
            "actions": action_index.actions,
            "epochs": args.epochs,
            "final_loss": losses[-1] if losses else None,
            "project": _project_summary(context),
        },
        checkpoint,
    )
    print(f"wrote {checkpoint}")


def cmd_export_dataset(args: argparse.Namespace) -> None:
    records = load_records(args.audit_log)
    payload = {
        "rows": [record_to_training_row(record) for record in records],
        "pairwiseRanking": pairwise_ranking_labels(records),
    }
    print(f"wrote {_write(payload, args.output)}")


def cmd_memory_build(args: argparse.Namespace) -> None:
    memory = HierarchicalMemory.open(args.database)
    report = memory.ingest_curated_seed(
        knowledge_path=args.knowledge_base,
        mechanism_path=args.mechanism_base,
    )
    payload = {
        "database": str(Path(args.database).expanduser().resolve()),
        "ingestion": report.model_dump(mode="json"),
        "stats": memory.store.stats().model_dump(mode="json"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        _write(payload, args.output)


def cmd_memory_ingest_document(args: argparse.Namespace) -> None:
    memory = HierarchicalMemory.open(args.database)
    report = memory.ingest_external_document(
        args.file,
        title=args.title,
        source_kind=args.source_kind,
        source_uri=args.source_uri,
        scope=args.scope,
        review_status=args.review_status,
        node_ids=tuple(args.node_id),
        chunk_characters=args.chunk_characters,
    )
    payload = {
        "database": str(Path(args.database).expanduser().resolve()),
        "ingestion": report.model_dump(mode="json"),
        "stats": memory.store.stats().model_dump(mode="json"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        _write(payload, args.output)


def cmd_memory_search(args: argparse.Namespace) -> None:
    memory = HierarchicalMemory.open(args.database)
    result = memory.retrieve(
        args.query,
        family_hint=args.family,
        layers=tuple(args.layer) if args.layer else (0, 1, 2, 3, 4),
        graph_depth=args.graph_depth,
        limit=args.limit,
    )
    payload = result.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        _write(payload, args.output)


def cmd_memory_status(args: argparse.Namespace) -> None:
    memory = HierarchicalMemory.open(args.database)
    payload = memory.store.stats().model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        _write(payload, args.output)


def cmd_memory_taxonomy(args: argparse.Namespace) -> None:
    memory = HierarchicalMemory.open(args.database)
    profile = memory.resolve_taxonomy(args.node, scope=args.family)
    payload = profile.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        _write(payload, args.output)


def cmd_measure_mechanisms(args: argparse.Namespace) -> None:
    context = load_project(args.project)
    payload = {
        "projectId": context.manifest.project_id,
        "problemId": context.problem.id,
        "problemFamily": context.problem.kind,
        "incumbentHash": schedule_hash(context.incumbent),
        "objective": "makespan",
        "measurements": [
            item.model_dump(mode="json")
            for item in measure_mechanisms(context.problem, context.incumbent)
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        _write(payload, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Domain-independent project-conditioned scheduling optimizer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-semantics")
    audit.add_argument("--semantics", required=True)
    audit.add_argument("--project-root", required=True)
    audit.add_argument("--output")
    audit.set_defaults(func=cmd_audit_semantics)

    compiler = subparsers.add_parser("compile-semantics")
    compiler.add_argument("--project-root", required=True)
    compiler.add_argument("--output")
    compiler.set_defaults(func=cmd_compile_semantics)

    llm_compiler = subparsers.add_parser("compile-semantics-llm")
    llm_compiler.add_argument("--project-root", required=True)
    llm_compiler.add_argument(
        "--env-file",
        default=str(LAB_ROOT / ".env"),
    )
    llm_compiler.add_argument("--provider-prefix", default="SEED")
    llm_compiler.add_argument("--max-input-characters", type=int, default=60000)
    llm_compiler.add_argument("--max-output-tokens", type=int, default=6000)
    llm_compiler.add_argument("--max-schema-repairs", type=int, default=1)
    llm_compiler.add_argument(
        "--knowledge-base",
        default=str(
            Path(__file__).resolve().parent
            / "knowledge"
            / "scheduling_families.json"
        ),
    )
    llm_compiler.add_argument(
        "--assess-constraint-impact",
        action="store_true",
        help="run a second high-reasoning critic over extracted constraints",
    )
    llm_compiler.add_argument(
        "--impact-protocol",
        choices=["openai", "anthropic"],
        default="anthropic",
    )
    llm_compiler.add_argument(
        "--impact-provider-prefix",
        default="SEED",
        help="OpenAI-compatible provider prefix when impact protocol is openai; DeepSeek is opt-in only",
    )
    llm_compiler.add_argument(
        "--impact-max-output-tokens",
        type=int,
        default=12000,
    )
    llm_compiler.add_argument("--timeout", type=float, default=120.0)
    llm_compiler.add_argument("--max-attempts", type=int, default=3)
    llm_compiler.add_argument("--dry-run", action="store_true")
    llm_compiler.add_argument("--output")
    llm_compiler.set_defaults(func=cmd_compile_semantics_llm)

    staged = subparsers.add_parser("compile-semantics-staged")
    staged.add_argument("--project-root", required=True)
    staged.add_argument(
        "--env-file",
        default=str(LAB_ROOT / ".env"),
    )
    staged.add_argument(
        "--navigator-provider-prefix",
        default="MIMO",
        help="low-cost OpenAI-compatible provider used only for repository navigation",
    )
    staged.add_argument(
        "--analyst-protocol",
        choices=["openai", "anthropic", "claude-code"],
        default="claude-code",
        help="high-capability provider protocol for batched analysis and synthesis",
    )
    staged.add_argument(
        "--analyst-provider-prefix",
        default="SEED",
        help="provider prefix when analyst protocol is openai",
    )
    staged.add_argument("--max-rounds-per-batch", type=int, default=3)
    staged.add_argument("--max-reads", type=int, default=18)
    staged.add_argument("--max-output-tokens", type=int, default=7000)
    staged.add_argument("--max-schema-repairs", type=int, default=1)
    staged.add_argument(
        "--knowledge-base",
        default=str(
            Path(__file__).resolve().parent
            / "knowledge"
            / "scheduling_families.json"
        ),
    )
    staged.add_argument(
        "--memory-database",
        default=str(LAB_ROOT / "outputs" / "memory" / "schedule_memory.sqlite3"),
        help="optional existing SQLite graph/FTS memory; absent files are not created",
    )
    staged.add_argument(
        "--assess-constraint-impact",
        action="store_true",
        help="run an independent structured-choice critic after final synthesis",
    )
    staged.add_argument(
        "--impact-protocol",
        choices=["openai", "anthropic", "claude-code"],
        default="claude-code",
        help="high-recall impact review provider; defaults to Opus through Claude Code",
    )
    staged.add_argument(
        "--impact-provider-prefix",
        default="SEED",
        help="provider prefix only when impact protocol is openai; DeepSeek is not a default reviewer",
    )
    staged.add_argument("--impact-max-output-tokens", type=int, default=12000)
    staged.add_argument("--timeout", type=float, default=240.0)
    staged.add_argument("--max-attempts", type=int, default=3)
    staged.add_argument(
        "--claude-code-base-url",
        help="optional official-client relay node; defaults to .env routing",
    )
    staged.add_argument(
        "--claude-code-max-budget-usd",
        type=float,
        default=8.0,
        help="hard Claude Code budget cap per individual call",
    )
    staged.add_argument(
        "--claude-code-project-skill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the project scheduling code-review Skill with read-only tools",
    )
    staged.add_argument(
        "--claude-code-skill-name",
        default="scheduling-code-semantics",
    )
    staged.add_argument(
        "--claude-code-read-budget",
        type=int,
        default=18,
        help="read/search budget stated to the Claude Code project Skill",
    )
    staged.add_argument(
        "--claude-code-read-mode",
        choices=["harness", "claude"],
        default="harness",
        help="single authoritative reread path; harness is the default",
    )
    staged.add_argument(
        "--analyst-reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="high",
    )
    staged.add_argument(
        "--impact-reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="high",
    )
    staged.add_argument(
        "--event-log",
        help="JSONL path for model/stage/batch/round call tracing",
    )
    staged.add_argument("--dry-run", action="store_true")
    staged.add_argument("--output")
    staged.set_defaults(func=cmd_compile_semantics_staged)

    impact_replay = subparsers.add_parser("review-constraint-impact")
    impact_replay.add_argument("--semantic-compilation", required=True)
    impact_replay.add_argument("--output", required=True)
    impact_replay.add_argument(
        "--env-file",
        default=str(LAB_ROOT / ".env"),
    )
    impact_replay.add_argument(
        "--protocol",
        choices=["openai", "anthropic", "claude-code"],
        default="claude-code",
    )
    impact_replay.add_argument("--provider-prefix", default="SEED")
    impact_replay.add_argument(
        "--knowledge-base",
        default=str(
            Path(__file__).resolve().parent
            / "knowledge"
            / "scheduling_families.json"
        ),
    )
    impact_replay.add_argument("--max-output-tokens", type=int, default=12000)
    impact_replay.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="high",
    )
    impact_replay.add_argument("--timeout", type=float, default=240.0)
    impact_replay.add_argument("--max-attempts", type=int, default=3)
    impact_replay.add_argument("--claude-code-base-url")
    impact_replay.add_argument("--claude-code-max-budget-usd", type=float, default=8.0)
    impact_replay.add_argument("--event-log")
    impact_replay.set_defaults(func=cmd_review_constraint_impact)

    harness = subparsers.add_parser("run-semantic-harness")
    harness.add_argument("--case", required=True)
    harness.add_argument(
        "--env-file",
        default=str(LAB_ROOT / ".env"),
    )
    harness.add_argument("--provider-prefix", default="SEED")
    harness.add_argument("--max-paper-characters", type=int, default=40000)
    harness.add_argument("--max-code-characters", type=int, default=60000)
    harness.add_argument("--timeout", type=float, default=180.0)
    harness.add_argument("--max-attempts", type=int, default=3)
    harness.add_argument("--output")
    harness.set_defaults(func=cmd_run_semantic_harness)

    benchmark = subparsers.add_parser("run-code-semantic-benchmark")
    benchmark.add_argument("--case", required=True)
    benchmark.add_argument("--label-artifact", required=True)
    benchmark.add_argument(
        "--env-file",
        default=str(LAB_ROOT / ".env"),
    )
    benchmark.add_argument("--provider-prefix", required=True)
    benchmark.add_argument(
        "--protocol",
        choices=["openai", "anthropic"],
        default="openai",
    )
    benchmark.add_argument("--max-code-characters", type=int, default=60000)
    benchmark.add_argument("--timeout", type=float, default=240.0)
    benchmark.add_argument("--max-attempts", type=int, default=3)
    benchmark.add_argument("--output")
    benchmark.set_defaults(func=cmd_run_code_semantic_benchmark)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--project", required=True)
    inspect.add_argument("--top-k", type=int, default=8)
    inspect.add_argument("--bc-epochs", type=int, default=20)
    inspect.add_argument("--output")
    inspect.set_defaults(func=cmd_inspect)

    optimize = subparsers.add_parser("optimize")
    optimize.add_argument("--project", required=True)
    optimize.add_argument("--top-k", type=int, default=6)
    optimize.add_argument("--bc-epochs", type=int, default=20)
    optimize.add_argument("--iterations", type=int, default=2)
    optimize.add_argument("--candidate-budget", type=int, default=6)
    optimize.add_argument("--full-oracle-budget", type=int)
    optimize.add_argument("--audit-log")
    optimize.add_argument("--output")
    optimize.set_defaults(func=cmd_optimize)

    demo = subparsers.add_parser("demo")
    demo.add_argument("--family", choices=["jsp", "fsp", "fjsp", "hfsp"], required=True)
    demo.add_argument("--mode", choices=["inspect", "optimize"], default="optimize")
    demo.add_argument("--top-k", type=int, default=4)
    demo.add_argument("--bc-epochs", type=int, default=8)
    demo.add_argument("--iterations", type=int, default=1)
    demo.add_argument("--candidate-budget", type=int, default=2)
    demo.add_argument("--full-oracle-budget", type=int)
    demo.add_argument("--audit-log")
    demo.add_argument("--output")
    demo.set_defaults(func=cmd_demo)

    train = subparsers.add_parser("train-policy")
    train.add_argument("--project", required=True)
    train.add_argument("--top-k", type=int, default=8)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument(
        "--checkpoint",
        default=str(LAB_ROOT / "checkpoints" / "masked_ppo_bc.pt"),
    )
    train.set_defaults(func=cmd_train_policy)

    export = subparsers.add_parser("export-dataset")
    export.add_argument("--audit-log", required=True)
    export.add_argument(
        "--output",
        default=str(LAB_ROOT / "outputs" / "training_dataset.json"),
    )
    export.set_defaults(func=cmd_export_dataset)

    default_memory = str(LAB_ROOT / "outputs" / "memory" / "schedule_memory.sqlite3")
    default_knowledge = str(
        Path(__file__).resolve().parent / "knowledge" / "scheduling_families.json"
    )
    default_mechanisms = str(
        Path(__file__).resolve().parent / "knowledge" / "mechanism_targets.json"
    )

    memory_build = subparsers.add_parser("memory-build")
    memory_build.add_argument("--database", default=default_memory)
    memory_build.add_argument("--knowledge-base", default=default_knowledge)
    memory_build.add_argument("--mechanism-base", default=default_mechanisms)
    memory_build.add_argument("--output")
    memory_build.set_defaults(func=cmd_memory_build)

    memory_ingest = subparsers.add_parser("memory-ingest-document")
    memory_ingest.add_argument("--database", default=default_memory)
    memory_ingest.add_argument("--file", required=True)
    memory_ingest.add_argument("--title")
    memory_ingest.add_argument("--source-kind", default="other")
    memory_ingest.add_argument("--source-uri")
    memory_ingest.add_argument("--scope", default="global")
    memory_ingest.add_argument(
        "--review-status",
        choices=[
            "proposed",
            "canonicalized",
            "source_verified",
            "conflict_checked",
            "active",
            "rejected",
            "superseded",
        ],
        default="proposed",
    )
    memory_ingest.add_argument("--node-id", action="append", default=[])
    memory_ingest.add_argument("--chunk-characters", type=int, default=4000)
    memory_ingest.add_argument("--output")
    memory_ingest.set_defaults(func=cmd_memory_ingest_document)

    memory_search = subparsers.add_parser("memory-search")
    memory_search.add_argument("--database", default=default_memory)
    memory_search.add_argument("--query", required=True)
    memory_search.add_argument("--family")
    memory_search.add_argument(
        "--layer",
        action="append",
        type=int,
        choices=range(0, 5),
        default=[],
    )
    memory_search.add_argument("--graph-depth", type=int, default=1)
    memory_search.add_argument("--limit", type=int, default=12)
    memory_search.add_argument("--output")
    memory_search.set_defaults(func=cmd_memory_search)

    memory_status = subparsers.add_parser("memory-status")
    memory_status.add_argument("--database", default=default_memory)
    memory_status.add_argument("--output")
    memory_status.set_defaults(func=cmd_memory_status)

    memory_taxonomy = subparsers.add_parser("memory-taxonomy")
    memory_taxonomy.add_argument("--database", default=default_memory)
    memory_taxonomy.add_argument("--node", required=True)
    memory_taxonomy.add_argument("--family")
    memory_taxonomy.add_argument("--output")
    memory_taxonomy.set_defaults(func=cmd_memory_taxonomy)

    mechanisms = subparsers.add_parser("measure-mechanisms")
    mechanisms.add_argument("--project", required=True)
    mechanisms.add_argument("--output")
    mechanisms.set_defaults(func=cmd_measure_mechanisms)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
