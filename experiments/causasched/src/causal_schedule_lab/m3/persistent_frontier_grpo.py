"""T2-L unified net-intervention persistent-frontier GRPO.

M2 and M3 sample from frozen-SFT plus trainable-residual scores.  Candidate-wide
G1 probes, GH continuation and PA/PB/PC/PF oracle-like tiers are not run.  Twenty
of K=24 siblings remain pure on-policy GRPO; four cheap anchors inspect only the
actor's top candidates.  M2 and M3 share one scalar return: best-within-horizon
makespan gain minus best-to-terminal regression and infeasibility downside.
The frontier retains the best visited state.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from causal_schedule_lab.validation import schedule_hash
from . import joint_grpo as JG
from .fixed_cohort_metrics import FixedCohortMetrics


class ActivePoolWriter:
    """Prevent changing-population statistics looking like fixed-set progress."""
    def __init__(self, writer):
        self.writer = writer

    def add_scalar(self, tag, value, step):
        if tag.startswith(("cumulative/", "generation/")):
            tag = "active_pool/" + tag
        self.writer.add_scalar(tag, value, step)

    def flush(self):
        self.writer.flush()


@dataclass
class FrontierState:
    schedule: object
    ms: int
    gstep: int = 0
    search_steps: int = 0
    acted_steps: int = 0
    records: tuple = field(default_factory=tuple)
    generation: int = 0
    parent_hash: str | None = None

    @property
    def state_hash(self):
        return schedule_hash(self.schedule)


@dataclass
class ArenaGraph:
    iid: str
    episode_id: int
    problem: object
    src: str
    s0_schedule: object
    base_progmem: object
    current: FrontierState
    best: FrontierState
    elites: list = field(default_factory=list)
    state_pool: list = field(default_factory=list)

    @classmethod
    def from_graph(cls, graph):
        s0 = copy.deepcopy(graph.s0_schedule)
        state = FrontierState(schedule=copy.deepcopy(s0), ms=int(graph.ms0))
        return cls(graph.iid, int(graph.episode_id), graph.problem, graph.src, s0,
                   graph.pristine_progmem, state, state, [])

    def materialize_memory(self, state):
        pm = copy.deepcopy(self.base_progmem)
        for rec in state.records:
            pm.add_executed(self.iid, int(rec["written_at_step"]), copy.deepcopy(rec))
        return pm

    def rollout_graph(self, state):
        return SimpleNamespace(
            iid=self.iid, episode_id=self.episode_id, problem=self.problem,
            schedule=state.schedule, ms=int(state.ms), gstep=int(state.gstep),
            progmem=self.materialize_memory(state), done=False, src=self.src)


def _seed(*parts):
    raw = "::".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & 0x7fffffff


def update_state_pool(ag, root, trajectories, generation, capacity, rng):
    """Keep bounded, quality-stratified states, never old policy log-probs.

    Best-prefix and terminal snapshots are already collected.  Memory is paired
    with the exact prefix that produced each state; no future records are used.
    """
    candidates = list(ag.state_pool)
    for tr in trajectories:
        if tr.get("is_anchor", False) or int(tr.get("n_acted", 0)) == 0:
            continue
        candidates.append(_candidate(root, tr, generation)["state"])
        sample = tr.get("state_sample")
        if sample:
            candidates.append(FrontierState(
                schedule=sample["schedule"], ms=sample["ms"],
                gstep=root.gstep + sample["n_acted"],
                search_steps=root.search_steps + sample["n_steps"],
                acted_steps=root.acted_steps + sample["n_acted"],
                records=tuple(root.records) + tuple(sample["records"]),
                generation=generation, parent_hash=root.state_hash))
        candidates.append(FrontierState(
            schedule=tr["terminal_schedule"], ms=int(tr["final_ms"]),
            gstep=root.gstep + int(tr["n_acted"]),
            search_steps=root.search_steps + int(tr.get("n_steps", 0)),
            acted_steps=root.acted_steps + int(tr["n_acted"]),
            records=tuple(root.records) + tuple(tr.get("terminal_memory_records", ())),
            generation=generation, parent_hash=root.state_hash))
    unique = {s.state_hash: s for s in candidates}
    bins = [[], [], []]
    ms0 = int(ag.s0_schedule.makespan)
    for s in unique.values():
        bins[0 if s.ms < ms0 else 1 if s.ms == ms0 else 2].append(s)
    for bucket in bins:
        rng.shuffle(bucket)
    kept = []
    while len(kept) < capacity and any(bins):
        for bucket in bins:
            if bucket and len(kept) < capacity:
                kept.append(bucket.pop())
    ag.state_pool = kept


def diverse_tasks(arena, count, rng, s0_fraction, elite_fraction):
    """Instance-balanced draws; sampling categories do not alter action scores."""
    tasks = []
    indices = []
    while len(indices) < count:
        cycle = list(range(len(arena)))
        rng.shuffle(cycle)
        indices.extend(cycle)
    for gi in indices[:count]:
        ag = arena[gi]
        q = rng.random()
        if q < s0_fraction:
            kind, state = "s0", FrontierState(ag.s0_schedule, int(ag.s0_schedule.makespan))
        elif q < s0_fraction + elite_fraction:
            kind, state = "best", ag.best
        elif ag.state_pool:
            kind, state = "pool", rng.choice(ag.state_pool)
        else:
            kind, state = "s0", FrontierState(ag.s0_schedule, int(ag.s0_schedule.makespan))
        tasks.append((kind, gi, state))
    rng.shuffle(tasks)
    return tasks


def resolve_best_tasks(arena, tasks):
    """Resolve incumbents at collection time, after preceding batch promotions."""
    return [(kind, gi, arena[gi].best if kind == "best" else state)
            for kind, gi, state in tasks]


def _candidate(root, tr, generation):
    attempted = max(1, int(tr.get("n_steps", 0)))
    # Promote the best state actually visited, not blindly the H-step terminal
    # state.  Full attempted search is still charged to search_steps, so this
    # does not extend the configured budget or hide rollout cost.
    best_step = int(tr.get("best_step", tr.get("n_acted", 0)))
    schedule = tr.get("best_schedule", tr["terminal_schedule"])
    ms = int(tr.get("best_ms", tr["final_ms"]))
    records = tuple(root.records) + tuple(
        tr.get("best_memory_records", tr.get("terminal_memory_records", ())))
    st = FrontierState(
        schedule=schedule, ms=ms,
        gstep=int(root.gstep) + best_step,
        search_steps=int(root.search_steps) + attempted,
        acted_steps=int(root.acted_steps) + best_step,
        records=records, generation=int(generation), parent_hash=root.state_hash)
    mem_gain = sum(max(0.0, float(r.get("true_U") or 0.0))
                   for r in tr.get("terminal_memory_records", ()))
    return {"state": st, "score": float(root.ms - st.ms),
            "memory_score": float(mem_gain), "terminal": tr.get("terminal"),
            "n_steps": attempted, "n_acted": best_step,
            "best_step": best_step, "terminal_ms": int(tr["final_ms"])}


def retain_four(candidates, n=4):
    """Makespan first, Memory only breaks equal-makespan ties; unique states."""
    unique = {}
    for c in candidates:
        h = c["state"].state_hash
        old = unique.get(h)
        key = (c["score"], c["memory_score"], c["n_acted"])
        if old is None or key > (old["score"], old["memory_score"], old["n_acted"]):
            unique[h] = c
    ranked = sorted(unique.values(),
                    key=lambda c: (-c["score"], -c["memory_score"],
                                   c["state"].ms, c["state"].state_hash))
    return ranked[:max(1, int(n))]


def score_probabilities(candidates):
    """Turn the observed makespan reductions directly into a probability vector.

    A translation by the batch minimum is the only normalization needed for
    negative scores.  It preserves score gaps and contains no fixed rank weights.
    Equal scores are sampled uniformly.
    """
    scores = np.asarray([float(c["score"]) for c in candidates], dtype=np.float64)
    if len(scores) == 0:
        return np.asarray([], dtype=np.float64)
    if float(scores.max() - scores.min()) <= 1e-12:
        return np.full(len(scores), 1.0 / len(scores), dtype=np.float64)
    weights = scores - scores.min()
    weights += max(float(weights.max()) * 1e-6, 1e-9)
    return weights / weights.sum()


def _sample_candidate(candidates, rng):
    probs = score_probabilities(candidates)
    idx = int(rng.choices(range(len(candidates)), weights=probs.tolist(), k=1)[0])
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return candidates[idx], probs.tolist(), entropy


def _anchor_tasks(arena, n_elite, n_s0, rng):
    elite_pool = [(gi, st) for gi, ag in enumerate(arena)
                  for st in (ag.elites or [ag.best])]
    out = []
    for _ in range(int(n_elite)):
        gi, state = rng.choice(elite_pool)
        out.append(("elite", gi, state))
    for _ in range(int(n_s0)):
        gi = rng.randrange(len(arena))
        ag = arena[gi]
        out.append(("s0", gi, FrontierState(
            schedule=ag.s0_schedule, ms=int(ag.s0_schedule.makespan))))
    return out


def arena_state_dict(arena, generation, history, cycle_history=None, fixed_cohort=None):
    # Problems and frozen base Memory are rebuilt by the canonical startup.  Do
    # not duplicate them in every checkpoint (the old per-graph Memory copies are
    # the largest CPU objects in this project).
    states = {ag.iid: {"current": ag.current, "best": ag.best,
                       "elites": ag.elites, "state_pool": ag.state_pool} for ag in arena}
    return {"generation": int(generation), "history": history,
            "cycle_history": list(cycle_history or []), "states": states,
            "active_definitions": [(ag.iid, ag.episode_id, ag.problem, ag.src,
                                    ag.s0_schedule) for ag in arena],
            "fixed_cohort": fixed_cohort,
            "format": "t2l_diverse_state_v4"}


def _step_stats(values):
    a = np.asarray(list(values), dtype=np.float64)
    if not a.size:
        return {"min": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0}
    return {"min": int(a.min()), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "max": int(a.max())}


def _cycle_diagnostics(groups, *, cycle, generation, repeat, horizon, batch):
    """Paper/debug trace for one completed collection batch.

    This contains no model parameters and is written by the parent process only.
    It makes early termination attributable to a concrete reason and depth.
    """
    trajs = [tr for group in groups for tr in group.get("trajs", ())]
    acted = [int(tr.get("n_acted", 0)) for tr in trajs]
    attempted = [int(tr.get("n_steps", 0)) for tr in trajs]
    terminal_counts = {}
    terminal_depth_counts = {}
    layer_counts = {}
    sampled_class_counts = {}
    selected_operator_counts = {}
    selected_action_family_counts = {}
    execution_failure_counts = {}
    profile_cpu_sum = {}
    nonfinite_g1 = nonfinite_gh = bypass_attempts = decision_steps = 0
    preselection_replays = selected_action_replays = 0
    worker_cpu_sum = 0.0
    worker_wall_max = 0.0
    candidate_hops = []
    selected_hops = []
    selected_supports = []
    selected_prior_trust = []
    selected_root_values = []
    selected_trace_probabilities = []
    appearance_potential_entropies = []
    active_appearance_counts = []
    trace_expected_depths = []
    selected_differs = []
    root_policy_entropies = []
    root_policy_effective_counts = []
    root_candidate_counts = []
    for tr in trajs:
        bypass_attempts += int(tr.get("post_m2_root_gate_bypass_attempts", 0))
        nonfinite_g1 += int(tr.get("pval_nonfinite_g1_count", 0))
        nonfinite_gh += int(tr.get("pval_nonfinite_gh_count", 0))
        preselection_replays += int(tr.get("preselection_replay_count") or 0)
        selected_action_replays += int(tr.get("selected_action_replay_count") or 0)
        terminal = tr.get("terminal") or "horizon_complete"
        terminal_counts[terminal] = terminal_counts.get(terminal, 0) + 1
        depth_key = f"{terminal}@acted_{int(tr.get('n_acted', 0))}"
        terminal_depth_counts[depth_key] = terminal_depth_counts.get(depth_key, 0) + 1
        for key, value in (tr.get("prof") or {}).items():
            profile_cpu_sum[key] = profile_cpu_sum.get(key, 0.0) + float(value)
        for rec in tr.get("steps", ()):
            decision_steps += 1
            layer = str(rec.get("active_layer") or "unknown")
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            sampled = str(rec.get("sampled_class") or "unknown")
            sampled_class_counts[sampled] = sampled_class_counts.get(sampled, 0) + 1
            for edit_type in rec.get("edit_types", ()):
                key = str(edit_type)
                selected_operator_counts[key] = selected_operator_counts.get(key, 0) + 1
            family = str(rec.get("action_family") or "unknown")
            selected_action_family_counts[family] = (
                selected_action_family_counts.get(family, 0) + 1)
            reason = str(rec.get("execution_reason") or "unknown")
            if reason not in ("ok", "cached_validated", "cached_anchor_probe"):
                execution_failure_counts[reason] = execution_failure_counts.get(reason, 0) + 1
            m2d = rec.get("m2_diag") or {}
            candidate_hops.extend(int(v) for v in m2d.get("candidate_hops", ()))
            if m2d.get("root_candidate_count", 0) > 0:
                root_policy_entropies.append(float(
                    m2d.get("root_policy_entropy", 0.0)))
                root_policy_effective_counts.append(float(
                    m2d.get("root_policy_effective_count", 0.0)))
                root_candidate_counts.append(int(m2d.get("root_candidate_count", 0)))
            if m2d.get("selected_root_index", -1) >= 0:
                selected_hops.append(int(m2d.get("selected_min_hop", -1)))
                selected_supports.append(int(m2d.get("selected_support_count", 0)))
                selected_prior_trust.append(float(m2d.get("selected_prior_trust", 1.0)))
                selected_root_values.append(float(m2d.get("selected_root_value", 0.0)))
                selected_trace_probabilities.append(float(
                    m2d.get("selected_trace_probability", 0.0)))
                appearance_potential_entropies.append(float(
                    m2d.get("appearance_potential_entropy", 0.0)))
                active_appearance_counts.append(int(
                    m2d.get("active_appearance_count", 0)))
                trace_expected_depths.append(float(
                    m2d.get("trace_expected_depth", 0.0)))
                selected_differs.append(bool(m2d.get(
                    "selected_differs_from_b5_top1", False)))
    for group in groups:
        telemetry = group.get("worker_telemetry") or {}
        worker_cpu_sum += float(telemetry.get("worker_cpu_s", 0.0))
        worker_wall_max = max(worker_wall_max,
                              float(telemetry.get("worker_wall_s", 0.0)))
    n = max(len(trajs), 1)
    survival = {str(depth): float(sum(v >= depth for v in acted) / n)
                for depth in range(1, int(horizon) + 1)}
    root_rows = []
    for (kind, gi, _state), group in zip(batch, groups):
        gtr = list(group.get("trajs", ()))
        telemetry = group.get("worker_telemetry") or {}
        root_rows.append({
            "kind": kind, "graph_index": int(gi), "iid": group.get("iid"),
            "root_makespan": int(group.get("root_ms", 0)),
            "trajectory_count": len(gtr),
            "acted_steps_mean": (float(np.mean([t.get("n_acted", 0) for t in gtr]))
                                 if gtr else 0.0),
            "reward_mean": float(group.get("mean_reward", 0.0)),
            "reward_std": float(group.get("std_reward", 0.0)),
            "unique_trajectory_count": int(group.get("unique_trajectory_count", 0)),
            "unique_trajectory_rate": float(group.get("unique_trajectory_rate", 0.0)),
            "pure_unique_trajectory_count": int(
                group.get("pure_unique_trajectory_count", 0)),
            "pure_unique_trajectory_rate": float(
                group.get("pure_unique_trajectory_rate", 0.0)),
            "unique_terminal_state_count": int(
                group.get("unique_terminal_state_count", 0)),
            "unique_terminal_state_rate": float(
                group.get("unique_terminal_state_rate", 0.0)),
            "diversity_v4": dict(group.get("diversity_v4", {})),
            "worker_wall_seconds": float(telemetry.get("worker_wall_s", 0.0)),
            "worker_cpu_seconds": float(telemetry.get("worker_cpu_s", 0.0)),
            "terminal_counts": {
                str(k if k is not None else "horizon_complete"): int(v)
                for k, v in (group.get("terminal_counts") or {}).items()},
        })
    slowest_roots = sorted(
        [{"iid": row["iid"], "kind": row["kind"],
          "worker_wall_seconds": row["worker_wall_seconds"]}
         for row in root_rows],
        key=lambda row: row["worker_wall_seconds"], reverse=True)[:5]
    diversity = {
        "unique_path_rate_mean": float(np.mean(
            [row["unique_trajectory_rate"] for row in root_rows])) if root_rows else 0.0,
        "pure_unique_path_rate_mean": float(np.mean(
            [row["pure_unique_trajectory_rate"] for row in root_rows])) if root_rows else 0.0,
        "unique_terminal_state_rate_mean": float(np.mean(
            [row["unique_terminal_state_rate"] for row in root_rows])) if root_rows else 0.0,
    }
    return {
        "schema": "t2g-policy-sampled-rollout-diagnostics-v1", "cycle": int(cycle),
        "generation": int(generation), "repeat": int(repeat),
        "horizon": int(horizon), "root_groups": len(groups),
        "trajectory_count": len(trajs), "attempted_steps": _step_stats(attempted),
        "acted_steps": _step_stats(acted), "survival_rate": survival,
        "terminal_counts": terminal_counts,
        "terminal_depth_counts": terminal_depth_counts,
        "active_layer_counts": layer_counts,
        "sampled_class_counts": sampled_class_counts,
        "selected_operator_counts": selected_operator_counts,
        "selected_action_family_counts": selected_action_family_counts,
        "execution_failure_counts": execution_failure_counts,
        "decision_steps": decision_steps,
        "post_m2_root_gate_bypass_attempts": bypass_attempts,
        "post_m2_root_gate_bypass_rate": bypass_attempts / max(decision_steps, 1),
        "nonfinite_g1_count": nonfinite_g1,
        "nonfinite_gh_count": nonfinite_gh,
        "preselection_replay_count": preselection_replays,
        "selected_action_replay_count": selected_action_replays,
        "m2_candidate_hops": candidate_hops,
        "m2_selected_hops": selected_hops,
        "m2_selected_support_counts": selected_supports,
        "m2_selected_prior_trust": selected_prior_trust,
        "m2_selected_root_values": selected_root_values,
        "m2_selected_trace_probabilities": selected_trace_probabilities,
        "m2_appearance_potential_entropies": appearance_potential_entropies,
        "m2_active_appearance_counts": active_appearance_counts,
        "m2_trace_expected_depths": trace_expected_depths,
        "m2_selected_differs_from_b5": selected_differs,
        "m2_root_policy_entropies": root_policy_entropies,
        "m2_root_policy_effective_counts": root_policy_effective_counts,
        "m2_root_candidate_counts": root_candidate_counts,
        "profile_worker_cpu_seconds_sum": profile_cpu_sum,
        "worker_process_cpu_seconds_sum": worker_cpu_sum,
        "slowest_worker_job_wall_seconds": worker_wall_max,
        "trajectory_diversity": diversity,
        "diversity_v4": {
            key: (float(np.mean([g.get("diversity_v4", {}).get(key, 0.0) for g in groups]))
                  if groups else 0.0)
            for key in ("initial_root_set_rate", "first_action_unique_rate",
                        "prefix2_duplicate_rate", "prefix3_duplicate_rate",
                        "state_prefix3_duplicate_rate", "replay_cache_hit_rate")
        },
        "slowest_roots": slowest_roots,
        "roots": root_rows,
    }


def _write_cycle_diagnostics(directory, payload):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cycle_{int(payload['cycle']):04d}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    with (directory / "cycles.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _survival_text(survival, depths):
    return "/".join(f"{float(survival.get(str(d), 0.0)):.3f}" for d in depths)


def run(jpol, scorer, env, graphs, *, workers=16, branches=24, horizon=12,
        recollects=1, elites=4, target_steps=312, groups_per_update=48,
        grpo_epochs=3,
        long_horizon=20, long_horizon_every=5,
        max_generations=512, max_optimizer_cycles=500, seed=0,
        tensorboard_writer=None, rollout_pool=None,
        optimizer_persistent=True, checkpoint_cb=None, resume_state=None,
        evaluation_cb=None, evaluation_interval=5, log_prefix="[t2g]",
        diagnostics_dir=None, timing_log_path=None,
        diverse_training=False, state_pool_capacity=12, s0_fraction=0.2,
        elite_fraction=0.3, refresh_every=5, refresh_cb=None):
    """Run complete generations up to a fixed optimizer-cycle budget.

    ``target_steps`` is a frontier milestone/diagnostic, not a training stop.
    The requested optimizer-cycle budget is rounded down to a whole generation
    so every current graph receives exactly the same one-pass exposure.
    """
    if resume_state:
        arena = [ArenaGraph.from_graph(g) for g in graphs]
        definitions = resume_state.get("active_definitions")
        if definitions:
            if len(definitions) != len(arena):
                raise ValueError("resume active arena size changed")
            for ag, definition in zip(arena, definitions):
                ag.iid, ag.episode_id, ag.problem, ag.src, ag.s0_schedule = definition
        saved = resume_state.get("states", {})
        missing = [ag.iid for ag in arena if ag.iid not in saved]
        if missing:
            raise ValueError(f"T2-G resume is missing {len(missing)} arena states")
        for ag in arena:
            row = saved[ag.iid]
            ag.current, ag.best = row["current"], row["best"]
            ag.elites = row.get("elites", [])
            ag.state_pool = row.get("state_pool", [])
        start_generation = int(resume_state.get("generation", -1)) + 1
        history = list(resume_state.get("history", []))
        cycle_history = list(resume_state.get("cycle_history", []))
    else:
        arena = [ArenaGraph.from_graph(g) for g in graphs]
        start_generation, history = 0, []
        cycle_history = []
    if not arena:
        raise ValueError("T2-G arena is empty")
    saved_cohort = (resume_state or {}).get("fixed_cohort")
    if resume_state and saved_cohort is None and {ag.iid for ag in arena} != {g.iid for g in graphs}:
        raise ValueError("old checkpoint has refreshed instances but no fixed-cohort ledger; "
                         "cannot reconstruct retired best results, start a new run")
    fixed_cohort = FixedCohortMetrics(graphs, saved_cohort)
    fixed_cohort.update(arena)
    fixed_writer = tensorboard_writer
    if tensorboard_writer is not None:
        tensorboard_writer = ActivePoolWriter(tensorboard_writer)

    diagnostics_dir = Path(diagnostics_dir) if diagnostics_dir is not None else None
    if diagnostics_dir is not None:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
    timing_log_path = Path(timing_log_path) if timing_log_path is not None else None
    if timing_log_path is not None:
        timing_log_path.parent.mkdir(parents=True, exist_ok=True)

    def timing_event(stage, status, *, cycle=None, generation=None, seconds=None,
                     **details):
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": str(stage), "status": str(status),
            "cycle": cycle, "generation": generation,
        }
        if seconds is not None:
            row["seconds"] = float(seconds)
        row.update(details)
        suffix = (f" duration={float(seconds):.1f}s" if seconds is not None else "")
        detail_text = " ".join(f"{key}={value}" for key, value in details.items())
        print(f"{log_prefix} [{row['timestamp']}] {stage} {status}{suffix}"
              f"{(' ' + detail_text) if detail_text else ''}", flush=True)
        if timing_log_path is not None:
            with timing_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    wall0 = time.time()
    cycle_index = max((int(row.get("cycle", 0)) for row in cycle_history), default=0)
    fixed_cohort.publish(fixed_writer, cycle_index, arena)
    cycle_at_start = cycle_index
    nominal_generations = max(1, int(math.ceil(float(target_steps) /
                                               max(int(horizon), 1))))
    nominal_roots_per_repeat = (len(arena) + int(round(len(arena) * 2.0 / 7.0)) +
                                int(round(len(arena) * 1.0 / 7.0)))
    nominal_cycles_per_generation = (
        int(math.ceil(nominal_roots_per_repeat / max(int(groups_per_update), 1))) *
        int(recollects))
    nominal_total_cycles = nominal_cycles_per_generation * nominal_generations
    requested_cycle_cap = max(1, int(max_optimizer_cycles))
    full_generation_cycle_limit = (
        requested_cycle_cap // nominal_cycles_per_generation *
        nominal_cycles_per_generation)
    if full_generation_cycle_limit < nominal_cycles_per_generation:
        raise ValueError(
            f"max_optimizer_cycles={requested_cycle_cap} cannot fit one complete "
            f"generation ({nominal_cycles_per_generation} cycles)")
    planned_generations = min(
        int(max_generations),
        full_generation_cycle_limit // nominal_cycles_per_generation)
    estimated_total_cycles = planned_generations * nominal_cycles_per_generation

    def print_progress():
        total = max(estimated_total_cycles, cycle_index)
        base_h = max(int(horizon), 1)
        long_every = max(int(long_horizon_every), 0)
        long_h = max(int(long_horizon), 1)
        planned_long = (total // long_every) if long_every else 0
        step_total = total * base_h + planned_long * (long_h - base_h)
        recorded_steps = sum(int(row.get("trajectory_horizon", base_h))
                             for row in cycle_history)
        # Old resume payloads lack trajectory_horizon and are counted as base H.
        step_done = recorded_steps
        done_run = cycle_index - cycle_at_start
        if done_run <= 0:
            print(f"{log_prefix} progress [>-----------------------------] "
                  f"search_step_batches={step_done}/{step_total} "
                  f"optimizer_updates={cycle_index}/{total} "
                  "(0.0%) avg=calculating ETA=calculating",
                  flush=True)
            return
        mean_s = (time.time() - wall0) / done_run
        remaining = max(total - cycle_index, 0)
        eta_s = mean_s * remaining
        width = 30
        filled = min(width, int(width * step_done / max(step_total, 1)))
        bar = ("=" * filled + (">" if filled < width else "")).ljust(width, "-")
        eta_h, eta_rem = divmod(int(eta_s), 3600)
        eta_m, eta_sec = divmod(eta_rem, 60)
        finish_at = time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(time.time() + eta_s))
        print(f"{log_prefix} progress [{bar}] "
              f"search_step_batches={step_done}/{step_total} "
              f"optimizer_updates={cycle_index}/{total} "
              f"({100.0 * step_done / max(step_total, 1):.1f}%) "
              f"avg={mean_s:.1f}s ETA={eta_h:02d}:{eta_m:02d}:{eta_sec:02d} "
              f"finish~{finish_at}", flush=True)

    print_progress()
    # Match the policy activation used by generation 0 before recording the
    # pre-GRPO validation baseline.
    if start_generation == 0:
        for actor in (jpol.m2, jpol.m3):
            if hasattr(actor, "set_progress"):
                actor.set_progress(1, planned_generations)
    if tensorboard_writer is not None and start_generation == 0:
        tensorboard_writer.add_scalar(
            "cumulative/mean_best_makespan_reduction_per_graph", 0.0, 0)
        tensorboard_writer.add_scalar(
            "cumulative/sum_best_makespan_reduction_all_graphs", 0.0, 0)
        tensorboard_writer.add_scalar("cumulative/improved_graph_rate", 0.0, 0)
        tensorboard_writer.flush()
    initial_validation = None
    if start_generation == 0 and evaluation_cb is not None:
        timing_event("validation", "SKIPPED", cycle=0, generation=0,
                     reason="pre_GRPO_disabled_train_first")
    for generation in range(start_generation, planned_generations):
        if diverse_training and generation > 0 and refresh_every > 0 and generation % refresh_every == 0 and refresh_cb:
            refresh0 = time.time()
            replacements = refresh_cb(arena, generation)
            for gi, graph in replacements:
                arena[gi] = ArenaGraph.from_graph(graph)
            timing_event("active_pool_refresh", "END", generation=generation,
                         seconds=time.time() - refresh0, replaced=len(replacements))
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("state_pool/refreshed_instances", len(replacements), cycle_index)
        # Preserve T2-D's monotone residual activation across the complete fixed
        # optimizer budget, rather than ending activation at the 312-step milestone.
        for actor in (jpol.m2, jpol.m3):
            if hasattr(actor, "set_progress"):
                actor.set_progress(generation + 1, planned_generations)
        # Every graph receives one current-root visit in every sweep.  This fixed
        # allocation gives a paper-stable denominator: ceil(target/H) sweeps,
        # independent of rare safety terminals.  Actual executed/attempted steps
        # remain separately reported rather than changing the training length.
        active = list(range(len(arena)))
        # Never start a generation that cannot finish inside the hard cycle cap.
        # A partial sweep would expose only a shuffled subset of instances and is
        # unsuitable for a paper learning curve.
        if cycle_index + nominal_cycles_per_generation > full_generation_cycle_limit:
            print(f"{log_prefix} optimizer-cycle budget reached: "
                  f"{cycle_index}/{requested_cycle_cap}; stopped at complete "
                  f"generation boundary ({nominal_cycles_per_generation} "
                  "cycles/generation)", flush=True)
            break
        # FrontierState is append-only/immutable in this runner, so snapshots can
        # share old schedules and record prefixes.  This keeps long-run RAM and
        # checkpoint size linear instead of copying 300-step histories repeatedly.
        generation_roots = {i: arena[i].current for i in active}
        banks = {i: [] for i in active}
        phase = {"collect": 0.0, "update": 0.0, "validation": 0.0,
                 "promote": 0.0, "checkpoint": 0.0}
        anchor_counts = {"current": 0, "elite": 0, "best": 0, "s0": 0, "pool": 0}
        update_rows = []
        rollout_rewards = []
        rollout_best_rewards = []
        generation_cycle_start = cycle_index + 1

        # Every current root is freshly collected once. Elite/S0 anchors keep
        # earlier useful state distributions represented across frontier sweeps.
        for repeat in range(int(recollects)):
            rr = random.Random(_seed(seed, generation, repeat, "task-order"))
            tasks = [("current", i, generation_roots[i]) for i in active]
            n_elite = int(round(len(tasks) * 2.0 / 7.0))
            n_s0 = int(round(len(tasks) * 1.0 / 7.0))
            tasks.extend(_anchor_tasks(arena, n_elite, n_s0, rr))
            tail_fill = (-len(tasks)) % int(groups_per_update)
            if tail_fill:
                tasks.extend(_anchor_tasks(arena, tail_fill, 0, rr))
            if diverse_training:
                tasks = diverse_tasks(arena, len(tasks), rr, s0_fraction, elite_fraction)
            rr.shuffle(tasks)
            for pos in range(0, len(tasks), int(groups_per_update)):
                batch = tasks[pos:pos + int(groups_per_update)]
                if diverse_training:
                    batch = resolve_best_tasks(arena, batch)
                    timing_event("state_pool_batch", "SUMMARY", cycle=cycle_index + 1,
                        s0=sum(k == "s0" for k, _, _ in batch),
                        pool=sum(k == "pool" for k, _, _ in batch),
                        elite=sum(k == "elite" for k, _, _ in batch),
                        best=sum(k == "best" for k, _, _ in batch),
                        stored_states=sum(len(ag.state_pool) for ag in arena))
                cycle0 = time.time()
                next_cycle = cycle_index + 1
                cycle_horizon = (
                    int(long_horizon)
                    if int(long_horizon_every) > 0 and
                    next_cycle % int(long_horizon_every) == 0
                    else int(horizon))
                timing_event("cycle", "START", cycle=next_cycle,
                             generation=generation, roots=len(batch),
                             trajectories=len(batch) * int(branches),
                             horizon=cycle_horizon,
                             horizon_mode=("long" if cycle_horizon != int(horizon)
                                           else "base"))
                best_sum_before = float(sum(
                    int(ag.s0_schedule.makespan) - int(ag.best.ms) for ag in arena))
                root_graphs = [arena[gi].rollout_graph(state)
                               for _kind, gi, state in batch]
                coll0 = time.time()
                timing_event("collect", "START", cycle=next_cycle,
                             generation=generation, roots=len(batch),
                             trajectories=len(batch) * int(branches), workers=int(workers))
                def collection_progress(done, total, elapsed, *, pending_iids=None,
                                        waiting=False):
                    stride = max(1, int(math.ceil(total / 8.0)))
                    if waiting or done == 1 or done == total or done % stride == 0:
                        details = {"completed_jobs": done, "total_jobs": total,
                                   "elapsed_s": round(elapsed, 1)}
                        if pending_iids:
                            details["pending_instances"] = ",".join(pending_iids[:12])
                            details["pending_instance_count"] = len(pending_iids)
                        timing_event("collect", "WAITING" if waiting else "PROGRESS",
                                     cycle=next_cycle,
                                     generation=generation, completed_jobs=done,
                                     total_jobs=total, elapsed_s=round(elapsed, 1),
                                     **({key: value for key, value in details.items()
                                        if key not in {"completed_jobs", "total_jobs",
                                                       "elapsed_s"}}))

                groups = JG.collect_depth_groups_rollouts_r14(
                    jpol, scorer, env["executor"], env["model_b5"],
                    env["single_head"], env["direct_head"], root_graphs,
                    k=int(branches), T=None, eps=None, horizon=cycle_horizon,
                    seed=_seed(seed, generation, repeat, pos), workers=int(workers),
                    step0_cache=env.get("cache"), mp_ctx=None,
                    action_space="policy_sampled", val_cache={},
                    critical_gate=False, torch_threads=1, rollout_pool=rollout_pool,
                    stop_on_negative=False, allow_policy_stop=False,
                    feasible_fallback=True, progress_callback=collection_progress)
                coll_done = time.time()
                collect_only_s = coll_done - coll0
                phase["collect"] += collect_only_s
                timing_event("collect", "END", cycle=next_cycle,
                             generation=generation, seconds=collect_only_s)
                # Persist collection evidence before backward/update.  This is
                # intentionally parent-only, so worker processes never contend
                # on a shared log file.
                diag_payload = _cycle_diagnostics(
                    groups, cycle=cycle_index + 1, generation=generation,
                    repeat=repeat, horizon=cycle_horizon, batch=batch)
                prof = diag_payload["profile_worker_cpu_seconds_sum"]
                tracked_cpu = sum(float(value) for value in prof.values())
                all_worker_cpu = float(diag_payload["worker_process_cpu_seconds_sum"])
                untracked_cpu = max(0.0, all_worker_cpu - tracked_cpu)
                prof = dict(prof)
                prof["other_overhead"] = untracked_cpu
                ranked_prof = sorted(prof.items(), key=lambda item: item[1], reverse=True)
                timing_event(
                    "collect_breakdown", "SUMMARY", cycle=next_cycle,
                    generation=generation, seconds=collect_only_s,
                    aggregate_worker_cpu_s=round(all_worker_cpu, 3),
                    ranked=" > ".join(f"{key}:{value:.1f}s" for key, value in ranked_prof),
                    slowest_roots=";".join(
                        f"{row['iid']}:{row['worker_wall_seconds']:.1f}s"
                        for row in diag_payload["slowest_roots"]))
                if diagnostics_dir is not None:
                    diag_path = _write_cycle_diagnostics(diagnostics_dir, diag_payload)
                    print(
                        f"{log_prefix} rollout-diagnostics cycle={cycle_index + 1} "
                        f"acted(min/mean/p50/p90/max)="
                        f"{diag_payload['acted_steps']['min']}/"
                        f"{diag_payload['acted_steps']['mean']:.2f}/"
                        f"{diag_payload['acted_steps']['p50']:.0f}/"
                        f"{diag_payload['acted_steps']['p90']:.0f}/"
                        f"{diag_payload['acted_steps']['max']} "
                        f"survival@1/3/6/9/12="
                        f"{_survival_text(diag_payload['survival_rate'], (1, 3, 6, 9, 12))} "
                        f"terminals={diag_payload['terminal_counts']} "
                        f"operators={diag_payload['selected_operator_counts']} "
                        f"execution_failures={diag_payload['execution_failure_counts']} "
                        f"root_gate_bypass_attempts="
                        f"{diag_payload['post_m2_root_gate_bypass_attempts']} "
                        f"nonfinite(G1/GH)="
                        f"{diag_payload['nonfinite_g1_count']}/"
                        f"{diag_payload['nonfinite_gh_count']} "
                        f"replay(pre/select)="
                        f"{diag_payload['preselection_replay_count']}/"
                        f"{diag_payload['selected_action_replay_count']} "
                        f"v4(rootset/action/prefix3dup)="
                        f"{diag_payload['diversity_v4']['initial_root_set_rate']:.3f}/"
                        f"{diag_payload['diversity_v4']['first_action_unique_rate']:.3f}/"
                        f"{diag_payload['diversity_v4']['prefix3_duplicate_rate']:.3f} "
                        f"diversity(path/pure/terminal)="
                        f"{diag_payload['trajectory_diversity']['unique_path_rate_mean']:.3f}/"
                        f"{diag_payload['trajectory_diversity']['pure_unique_path_rate_mean']:.3f}/"
                        f"{diag_payload['trajectory_diversity']['unique_terminal_state_rate_mean']:.3f} "
                        f"saved={diag_path}", flush=True)
                for (kind, gi, state), group in zip(batch, groups):
                    anchor_counts[kind] += 1
                    rollout_rewards.extend(map(
                        float, group.get("terminal_rewards", group.get("rewards", ()))))
                    rollout_best_rewards.extend(map(
                        float, group.get("best_rewards", group.get("rewards", ()))))
                    if diverse_training:
                        update_state_pool(arena[gi], state, group["trajs"], generation,
                                          state_pool_capacity, random.Random(_seed(seed, next_cycle, gi, state.state_hash)))
                    if kind == "current" or diverse_training:
                        candidates = [_candidate(state, tr, generation)
                                      for tr in group["trajs"]]
                        if diverse_training:
                            for candidate in candidates:
                                candidate["score"] = float(arena[gi].s0_schedule.makespan - candidate["state"].ms)
                        # The generation ultimately consumes only retain_four.
                        # Top-k under its deterministic total ordering is merge
                        # associative, so pruning after each 64-trajectory batch
                        # is exactly equivalent to retaining all candidates
                        # until generation end, while avoiding schedule/history
                        # retention and GC pressure.
                        banks[gi] = retain_four(banks[gi] + candidates, elites)
                        if candidates:
                            batch_best = min((c["state"] for c in candidates),
                                             key=lambda s: s.ms)
                            if batch_best.ms < arena[gi].best.ms:
                                arena[gi].best = batch_best
                postprocess_done = time.time()
                postprocess_s = postprocess_done - coll_done
                timing_event("gpu_update", "START", cycle=next_cycle,
                             generation=generation, groups=len(groups),
                             epochs=int(grpo_epochs))
                upd0 = time.time()
                upd = JG.grpo_update_joint(
                    jpol, groups, stage="C", seeds=_seed(seed, generation, repeat, pos,
                                                         "update"),
                    epochs=int(grpo_epochs), log_prefix=log_prefix, credit="stagewise",
                    optimizer_persistent=bool(optimizer_persistent))
                upd_done = time.time()
                update_only_s = upd_done - upd0
                phase["update"] += update_only_s
                timing_event("gpu_update", "END", cycle=next_cycle,
                             generation=generation, seconds=update_only_s)
                update_rows.append(upd)
                cycle_index += 1
                batch_objective_rewards = np.asarray([
                    float(reward) for group in groups
                    for reward in group.get("rewards", ())], dtype=np.float64)
                batch_rewards = np.asarray([
                    float(reward) for group in groups
                    for reward in group.get("terminal_rewards",
                                             group.get("rewards", ()))],
                    dtype=np.float64)
                batch_best_rewards = np.asarray([
                    float(reward) for group in groups
                    for reward in group.get("best_rewards", ())], dtype=np.float64)
                batch_best_steps = np.asarray([
                    float(tr.get("best_step", 0)) for group in groups
                    for tr in group.get("trajs", ())], dtype=np.float64)
                batch_regression = np.asarray([
                    float(tr.get("best_to_terminal_regression", 0.0))
                    for group in groups for tr in group.get("trajs", ())],
                    dtype=np.float64)
                anchor_rewards = np.asarray([
                    float(tr.get("reward", 0.0)) for group in groups
                    for tr in group.get("trajs", ()) if tr.get("is_anchor", False)],
                    dtype=np.float64)
                pure_rewards = np.asarray([
                    float(tr.get("reward", 0.0)) for group in groups
                    for tr in group.get("trajs", ()) if not tr.get("is_anchor", False)],
                    dtype=np.float64)
                anchor_best_rewards = np.asarray([
                    float(tr.get("best_reward", tr.get("reward", 0.0)))
                    for group in groups for tr in group.get("trajs", ())
                    if tr.get("is_anchor", False)], dtype=np.float64)
                pure_best_rewards = np.asarray([
                    float(tr.get("best_reward", tr.get("reward", 0.0)))
                    for group in groups for tr in group.get("trajs", ())
                    if not tr.get("is_anchor", False)], dtype=np.float64)
                normalized_rewards = np.asarray([
                    float(reward) / max(float(group.get("root_ms", 1.0)), 1.0)
                    for group in groups for reward in group.get("rewards", ())],
                    dtype=np.float64)
                normalized_best_rewards = np.asarray([
                    float(reward) / max(float(group.get("root_ms", 1.0)), 1.0)
                    for group in groups
                    for reward in group.get("best_rewards", ())], dtype=np.float64)
                zero_std_group_rate = float(np.mean([
                    float(group.get("std_reward", 0.0)) <= 1e-12 for group in groups]))
                n_batch_traj = sum(len(group.get("trajs", ())) for group in groups)
                horizon_complete_rate = (sum(
                    tr.get("terminal") is None for group in groups
                    for tr in group.get("trajs", ())) / max(n_batch_traj, 1))
                policy_stop_rate = (sum(
                    tr.get("terminal") == "stop" for group in groups
                    for tr in group.get("trajs", ())) / max(n_batch_traj, 1))
                empty_feasible_pool_rate = (sum(
                    tr.get("terminal") in {
                        "no_proposals", "no_pool_m2", "no_pool_wide",
                        "no_pool_shortlist", "no_validated_pool",
                        "no_multistep_pool", "no_lexicographic_pool"}
                        | {"no_policy_sampled_pool", "no_executable_policy_candidates"}
                    for group in groups for tr in group.get("trajs", ())) /
                    max(n_batch_traj, 1))
                decision_steps = [rec for group in groups
                                  for tr in group.get("trajs", ())
                                  for rec in tr.get("steps", ())]
                pure_decision_steps = [rec for group in groups
                                       for tr in group.get("trajs", ())
                                       if not tr.get("is_anchor", False)
                                       for rec in tr.get("steps", ())
                                       if not rec.get("is_stop", False)]
                anchor_decision_steps = [rec for group in groups
                                         for tr in group.get("trajs", ())
                                         if tr.get("is_anchor", False)
                                         for rec in tr.get("steps", ())
                                         if not rec.get("is_stop", False)]
                single_steps = [rec for rec in decision_steps
                                if not rec.get("is_stop", False)
                                and not rec.get("selected_is_pair", False)]
                pair_steps = [rec for rec in decision_steps
                              if not rec.get("is_stop", False)
                              and rec.get("selected_is_pair", False)]
                post_m2_root_gate_bypass_attempts = sum(
                    int(tr.get("post_m2_root_gate_bypass_attempts", 0))
                    for group in groups for tr in group.get("trajs", ()))
                analyzed_action_states = sum(
                    int(tr.get("vprobe_n", 0))
                    for group in groups for tr in group.get("trajs", ()))
                post_m2_root_gate_bypass_rate = (
                    post_m2_root_gate_bypass_attempts /
                    max(analyzed_action_states, 1))
                best_gain_cycle = np.asarray([
                    int(ag.s0_schedule.makespan) - int(ag.best.ms) for ag in arena],
                    dtype=np.float64)
                best_sum_after = float(best_gain_cycle.sum())
                fixed_cohort.update(arena)
                fixed_metrics = fixed_cohort.metrics()
                fixed_cohort.publish(fixed_writer, cycle_index, arena)
                increment_sum = best_sum_after - best_sum_before
                monitor = upd.get("_monitor", {})
                last_epoch = upd.get("epochs", [])[-1] if upd.get("epochs") else {}
                metrics_done = time.time()
                metrics_s = metrics_done - upd_done
                collect_s = collect_only_s
                update_s = update_only_s
                current_count = sum(kind == "current" for kind, _gi, _state in batch)
                eval_metrics = None
                validation_s = 0.0
                if (evaluation_cb is not None and int(evaluation_interval) > 0 and
                        cycle_index % int(evaluation_interval) == 0):
                    eval0 = time.time()
                    timing_event("validation", "START", cycle=cycle_index,
                                 generation=generation)
                    eval_metrics = evaluation_cb(jpol, cycle_index)
                    validation_s = time.time() - eval0
                    phase["validation"] += validation_s
                    timing_event("validation", "END", cycle=cycle_index,
                                 generation=generation, seconds=validation_s)
                cycle_s = time.time() - cycle0
                cycle_row = {
                    "cycle": cycle_index, "generation": generation, "repeat": repeat,
                    "trajectory_horizon": cycle_horizon,
                    "root_groups": len(batch), "current_root_groups": current_count,
                    "reward_mean": (float(batch_best_rewards.mean())
                                    if batch_best_rewards.size else 0.0),
                    "terminal_reward_mean": (float(batch_rewards.mean())
                                             if batch_rewards.size else 0.0),
                    "objective_reward_mean": (
                        float(batch_objective_rewards.mean())
                        if batch_objective_rewards.size else 0.0),
                    "net_intervention_reward_mean": (
                        float(batch_objective_rewards.mean())
                        if batch_objective_rewards.size else 0.0),
                    "reward_max": (float(batch_best_rewards.max())
                                   if batch_best_rewards.size else 0.0),
                    "reward_positive_rate": (
                        float(np.mean(batch_best_rewards > 0))
                        if batch_best_rewards.size else 0.0),
                    "terminal_reward_positive_rate": (
                        float(np.mean(batch_rewards > 0))
                        if batch_rewards.size else 0.0),
                    "best_prefix_reward_mean": (
                        float(batch_best_rewards.mean())
                        if batch_best_rewards.size else 0.0),
                    "best_prefix_positive_rate": (
                        float(np.mean(batch_best_rewards > 0))
                        if batch_best_rewards.size else 0.0),
                    "best_step_mean": (float(batch_best_steps.mean())
                                       if batch_best_steps.size else 0.0),
                    "regressed_after_best_rate": (
                        float(np.mean(batch_regression > 0))
                        if batch_regression.size else 0.0),
                    "best_to_final_loss_mean": (
                        float(batch_regression.mean())
                        if batch_regression.size else 0.0),
                    "anchor_positive_rate": (
                        float(np.mean(anchor_best_rewards > 0))
                        if anchor_best_rewards.size else 0.0),
                    "on_policy_positive_rate": (
                        float(np.mean(pure_best_rewards > 0))
                        if pure_best_rewards.size else 0.0),
                    "reward_zero_rate": (
                        float(np.mean(batch_best_rewards == 0))
                        if batch_best_rewards.size else 0.0),
                    "normalized_reward_mean": (
                        float(normalized_best_rewards.mean())
                        if normalized_best_rewards.size else 0.0),
                    "normalized_reward_std": (
                        float(normalized_best_rewards.std())
                        if normalized_best_rewards.size else 0.0),
                    "normalized_terminal_reward_mean": (
                        float(normalized_rewards.mean())
                        if normalized_rewards.size else 0.0),
                    "zero_std_group_rate": zero_std_group_rate,
                    "horizon_complete_rate": horizon_complete_rate,
                    "policy_stop_rate": policy_stop_rate,
                    "empty_feasible_pool_rate": empty_feasible_pool_rate,
                    "post_m2_root_gate_bypass_rate": post_m2_root_gate_bypass_rate,
                    "post_m2_root_gate_bypass_attempts":
                        post_m2_root_gate_bypass_attempts,
                    "makespan_reduction_increment_sum": increment_sum,
                    "fixed_cohort_metrics": fixed_metrics,
                    "cumulative_best_reduction_sum": best_sum_after,
                    "cumulative_best_reduction_mean_per_graph":
                        float(best_gain_cycle.mean()),
                    "cumulative_improved_graph_rate": float(np.mean(best_gain_cycle > 0)),
                    "loss": float(last_epoch.get("loss", 0.0)),
                    "kl_m3": float(last_epoch.get("kl_ref_m3", 0.0)),
                    "kl_m2": float(last_epoch.get("kl_m2", 0.0)),
                    "collect_s": collect_s, "postprocess_s": postprocess_s,
                    "update_s": update_s, "metrics_s": metrics_s,
                    "validation_s": validation_s, "cycle_s": cycle_s,
                    "collect_profile_worker_cpu_s": prof,
                    "gpu_peak_memory_mb": float(monitor.get("gpu_peak_memory_mb", 0.0)),
                    "gpu_update_utilization_mean_pct":
                        monitor.get("gpu_utilization_update_mean_pct"),
                    "gpu_update_utilization_peak_pct":
                        monitor.get("gpu_utilization_update_peak_pct"),
                    "validation": eval_metrics,
                }
                cycle_history.append(cycle_row)
                print(
                    f"{log_prefix} cycle {cycle_index}: generation={generation} "
                    f"roots={len(batch)} current={current_count} H={cycle_horizon} "
                    f"reward={batch_best_rewards.mean() if batch_best_rewards.size else 0.0:+.3f} "
                    f"terminal={batch_rewards.mean() if batch_rewards.size else 0.0:+.3f} "
                    f"objective={batch_objective_rewards.mean() if batch_objective_rewards.size else 0.0:+.3f} "
                    f"net={batch_objective_rewards.mean() if batch_objective_rewards.size else 0.0:+.3f} "
                    f"positive={np.mean(batch_best_rewards > 0) if batch_best_rewards.size else 0.0:.3f} "
                    f"terminal_pos={np.mean(batch_rewards > 0) if batch_rewards.size else 0.0:.3f} "
                    f"best_gain={batch_best_rewards.mean() if batch_best_rewards.size else 0.0:+.3f} "
                    f"gain_increment={increment_sum:+.0f} "
                    f"active_pool_best_sum={best_sum_after:+.0f} "
                    f"fixed200_best_sum={fixed_metrics['sum_best_makespan_reduction_all_graphs']:+.0f} "
                    f"loss={float(last_epoch.get('loss', 0.0)):+.5f} "
                    f"fullH={horizon_complete_rate:.3f} "
                    f"empty={empty_feasible_pool_rate:.3f} "
                    f"root_gate_bypass={post_m2_root_gate_bypass_rate:.3f} "
                    f"collect={collect_s:.1f}s post={postprocess_s:.1f}s "
                    f"update={update_s:.1f}s metrics={metrics_s:.1f}s "
                    f"validation={validation_s:.1f}s total={cycle_s:.1f}s",
                    flush=True)
                timing_event("cycle", "END", cycle=cycle_index,
                             generation=generation, seconds=cycle_s,
                             collect_s=round(collect_s, 3),
                             postprocess_s=round(postprocess_s, 3),
                             gpu_update_s=round(update_s, 3),
                             metrics_s=round(metrics_s, 3),
                             validation_s=round(validation_s, 3))
                if eval_metrics is not None:
                    bestn_text = (f"bestN_norm="
                                  f"{eval_metrics['bestn_normalized_mean_pct']:+.3f}% "
                                  if "bestn_normalized_mean_pct" in eval_metrics else "")
                    print(f"{log_prefix} validation@{cycle_index}: "
                          f"greedy_norm={eval_metrics['greedy_normalized_mean_pct']:+.3f}% "
                          f"{bestn_text}"
                          f"improved={eval_metrics['greedy_improved_rate']:.3f} "
                          f"n={eval_metrics['n']} sec={eval_metrics['seconds']:.1f}",
                          flush=True)
                if tensorboard_writer is not None:
                    ch = np.asarray(diag_payload.get("m2_candidate_hops", ()), dtype=np.int64)
                    sh = np.asarray(diag_payload.get("m2_selected_hops", ()), dtype=np.int64)
                    ss = np.asarray(diag_payload.get("m2_selected_support_counts", ()),
                                    dtype=np.float64)
                    pt = np.asarray(diag_payload.get("m2_selected_prior_trust", ()),
                                    dtype=np.float64)
                    rv = np.asarray(diag_payload.get("m2_selected_root_values", ()),
                                    dtype=np.float64)
                    tp = np.asarray(diag_payload.get(
                        "m2_selected_trace_probabilities", ()), dtype=np.float64)
                    ae = np.asarray(diag_payload.get(
                        "m2_appearance_potential_entropies", ()), dtype=np.float64)
                    ac = np.asarray(diag_payload.get(
                        "m2_active_appearance_counts", ()), dtype=np.float64)
                    td = np.asarray(diag_payload.get(
                        "m2_trace_expected_depths", ()), dtype=np.float64)
                    df = np.asarray(diag_payload.get("m2_selected_differs_from_b5", ()),
                                    dtype=np.float64)
                    mpe = np.asarray(diag_payload.get("m2_root_policy_entropies", ()),
                                     dtype=np.float64)
                    mef = np.asarray(diag_payload.get(
                        "m2_root_policy_effective_counts", ()), dtype=np.float64)
                    mcc = np.asarray(diag_payload.get("m2_root_candidate_counts", ()),
                                     dtype=np.float64)
                    m2_step_records = [rec for group in groups
                                       for traj in group.get("trajs", ())
                                       if not traj.get("is_anchor", False)
                                       for rec in traj.get("steps", ())]
                    m2_future_best = np.asarray([
                        float(rec.get("m2_future_best_reward", 0.0))
                        for rec in m2_step_records], dtype=np.float64)
                    m2_future_terminal = np.asarray([
                        float(rec.get("m2_future_terminal_reward", 0.0))
                        for rec in m2_step_records], dtype=np.float64)
                    m2_future_net = np.asarray([
                        float(rec.get("m2_future_net_reward", 0.0))
                        for rec in m2_step_records], dtype=np.float64)
                    m2_training_adv = np.asarray([
                        float(rec.get("adv2", 0.0)) for rec in m2_step_records],
                        dtype=np.float64)
                    def _rate(values, predicate):
                        return float(np.mean(predicate(values))) if values.size else 0.0
                    cycle_scalars = {
                        "cycle/trajectory_horizon": cycle_horizon,
                        "cycle/makespan_reduction_increment_sum": increment_sum,
                        "cycle/makespan_reduction_increment_mean_all_graphs":
                            increment_sum / max(len(arena), 1),
                        "cycle/batch_rollout_reward_mean": (
                            float(batch_best_rewards.mean())
                            if batch_best_rewards.size else 0.0),
                        "cycle/batch_terminal_reward_mean": (
                            float(batch_rewards.mean()) if batch_rewards.size else 0.0),
                        "train/objective_reward_mean": (
                            float(batch_objective_rewards.mean())
                            if batch_objective_rewards.size else 0.0),
                        "train/net_intervention_reward_mean": (
                            float(batch_objective_rewards.mean())
                            if batch_objective_rewards.size else 0.0),
                        "train/net_regression_weight": float(
                            JG.C.T2L_NET_REGRESSION_WEIGHT),
                        "cycle/batch_rollout_reward_max": (
                            float(batch_best_rewards.max())
                            if batch_best_rewards.size else 0.0),
                        "cycle/batch_reward_positive_rate": (
                            float(np.mean(batch_best_rewards > 0))
                            if batch_best_rewards.size else 0.0),
                        "trajectory/terminal_positive_rate": (
                            float(np.mean(batch_rewards > 0)) if batch_rewards.size else 0.0),
                        "trajectory/best_prefix_positive_rate": (
                            float(np.mean(batch_best_rewards > 0))
                            if batch_best_rewards.size else 0.0),
                        "trajectory/best_prefix_gain_mean": (
                            float(batch_best_rewards.mean())
                            if batch_best_rewards.size else 0.0),
                        "trajectory/best_step_mean": (
                            float(batch_best_steps.mean())
                            if batch_best_steps.size else 0.0),
                        "trajectory/regressed_after_best_rate": (
                            float(np.mean(batch_regression > 0))
                            if batch_regression.size else 0.0),
                        "trajectory/best_to_final_loss_mean": (
                            float(batch_regression.mean())
                            if batch_regression.size else 0.0),
                        "trajectory/anchor_terminal_positive_rate": (
                            float(np.mean(anchor_rewards > 0))
                            if anchor_rewards.size else 0.0),
                        "trajectory/on_policy_terminal_positive_rate": (
                            float(np.mean(pure_rewards > 0))
                            if pure_rewards.size else 0.0),
                        "trajectory/anchor_best_within_horizon_positive_rate": (
                            float(np.mean(anchor_best_rewards > 0))
                            if anchor_best_rewards.size else 0.0),
                        "trajectory/on_policy_best_within_horizon_positive_rate": (
                            float(np.mean(pure_best_rewards > 0))
                            if pure_best_rewards.size else 0.0),
                        "cycle/batch_reward_zero_rate": (
                            float(np.mean(batch_best_rewards == 0))
                            if batch_best_rewards.size else 0.0),
                        "train/normalized_reward_mean": (
                            float(normalized_best_rewards.mean())
                            if normalized_best_rewards.size else 0.0),
                        "train/normalized_reward_std": (
                            float(normalized_best_rewards.std())
                            if normalized_best_rewards.size else 0.0),
                        "train/normalized_terminal_reward_mean": (
                            float(normalized_rewards.mean())
                            if normalized_rewards.size else 0.0),
                        "train/zero_std_group_rate": zero_std_group_rate,
                        "trajectory/horizon_complete_rate": horizon_complete_rate,
                        "trajectory/policy_stop_rate": policy_stop_rate,
                        "trajectory/empty_feasible_pool_rate": empty_feasible_pool_rate,
                        "trajectory/no_proposals_at_root_rate": (
                            float(diag_payload["terminal_depth_counts"].get(
                                "no_proposals@acted_0", 0)) /
                            max(float(diag_payload["trajectory_count"]), 1.0)),
                        "trajectory/post_m2_root_gate_bypass_rate":
                            post_m2_root_gate_bypass_rate,
                        "cycle/root_groups": len(batch),
                        "cycle/current_root_groups": current_count,
                        "cumulative/mean_best_makespan_reduction_per_graph":
                            float(best_gain_cycle.mean()),
                        "cumulative/sum_best_makespan_reduction_all_graphs": best_sum_after,
                        "cumulative/mean_best_makespan_reduction_pct": float(np.mean([
                            100.0 * gain / max(int(ag.s0_schedule.makespan), 1)
                            for gain, ag in zip(best_gain_cycle, arena)])),
                        "cumulative/improved_graph_rate": float(np.mean(best_gain_cycle > 0)),
                        "train/loss": float(last_epoch.get("loss", 0.0)),
                        "train/kl_m3": float(last_epoch.get("kl_ref_m3", 0.0)),
                        "train/kl_m2": float(last_epoch.get("kl_m2", 0.0)),
                        "train/grad_norm_m3": float(last_epoch.get("grad_norm_m3", 0.0)),
                        "train/grad_norm_m2": float(last_epoch.get("grad_norm_m2", 0.0)),
                        "train/clip_fraction_m3": float(last_epoch.get("clip_frac_m3", 0.0)),
                        "train/policy_entropy_m3": float(last_epoch.get("entropy_m3", 0.0)),
                        "m2/candidate_hop0_rate": _rate(ch, lambda x: x == 0),
                        "m2/candidate_hop1_rate": _rate(ch, lambda x: x == 1),
                        "m2/candidate_hop2_rate": _rate(ch, lambda x: x == 2),
                        "m2/candidate_hop3plus_rate": _rate(ch, lambda x: x >= 3),
                        "m2/selected_hop0_rate": _rate(sh, lambda x: x == 0),
                        "m2/selected_hop1_rate": _rate(sh, lambda x: x == 1),
                        "m2/selected_hop2_rate": _rate(sh, lambda x: x == 2),
                        "m2/selected_hop3plus_rate": _rate(sh, lambda x: x >= 3),
                        "m2/selected_unknown_hop_rate": _rate(sh, lambda x: x < 0),
                        "m2/selected_support_count_mean": (
                            float(ss.mean()) if ss.size else 0.0),
                        "m2/selected_multi_appearance_rate": _rate(ss, lambda x: x >= 2),
                        "m2/selected_differs_from_b5_top1_rate": (
                            float(df.mean()) if df.size else 0.0),
                        "m2/prior_trust_mean": float(pt.mean()) if pt.size else 1.0,
                        "m2/rl_root_value_abs_mean": (
                            float(np.abs(rv).mean()) if rv.size else 0.0),
                        "m2/unified_selected_node_probability_mean": (
                            float(tp.mean()) if tp.size else 0.0),
                        "m2/appearance_potential_entropy_mean": (
                            float(ae.mean()) if ae.size else 0.0),
                        "m2/active_appearance_count_mean": (
                            float(ac.mean()) if ac.size else 0.0),
                        "m2/trace_expected_hop_mean": (
                            float(td.mean()) if td.size else 0.0),
                        "m2/unique_initial_roots_mean": (
                            float(np.mean([g.get("unique_initial_roots", 0)
                                           for g in groups])) if groups else 0.0),
                        **{
                            f"diversity_v4/{key}_mean": float(np.mean([
                                g.get("diversity_v4", {}).get(key, 0.0) for g in groups]))
                            if groups else 0.0
                            for key in ("initial_root_set_rate", "first_action_unique_rate",
                                        "prefix2_duplicate_rate", "prefix3_duplicate_rate",
                                        "state_prefix3_duplicate_rate", "replay_cache_hit_rate")
                        },
                        "diversity/unique_trajectory_rate_mean": float(np.mean(
                            [g.get("unique_trajectory_rate", 0.0) for g in groups])) if groups else 0.0,
                        "diversity/pure_unique_trajectory_rate_mean": float(np.mean(
                            [g.get("pure_unique_trajectory_rate", 0.0) for g in groups])) if groups else 0.0,
                        "diversity/unique_terminal_state_rate_mean": float(np.mean(
                            [g.get("unique_terminal_state_rate", 0.0) for g in groups])) if groups else 0.0,
                        "m2/root_policy_entropy_mean": (
                            float(mpe.mean()) if mpe.size else 0.0),
                        "m2/root_policy_effective_count_mean": (
                            float(mef.mean()) if mef.size else 0.0),
                        "m2/root_candidate_count_mean": (
                            float(mcc.mean()) if mcc.size else 0.0),
                        "m2/root_policy_effective_fraction_mean": (
                            float(np.mean(mef / np.maximum(mcc, 1.0)))
                            if mef.size and mcc.size else 0.0),
                        "m2/future_best_reward_mean": (
                            float(m2_future_best.mean()) if m2_future_best.size else 0.0),
                        "m2/future_best_positive_rate": (
                            float(np.mean(m2_future_best > 0))
                            if m2_future_best.size else 0.0),
                        "m2/future_terminal_reward_mean": (
                            float(m2_future_terminal.mean())
                            if m2_future_terminal.size else 0.0),
                        "m2/future_net_intervention_reward_mean": (
                            float(m2_future_net.mean())
                            if m2_future_net.size else 0.0),
                        "m2/future_net_intervention_positive_rate": (
                            float(np.mean(m2_future_net > 0))
                            if m2_future_net.size else 0.0),
                        "m2/training_advantage_abs_mean": (
                            float(np.abs(m2_training_adv).mean())
                            if m2_training_adv.size else 0.0),
                        "actions/pair_candidate_rate": float(np.mean([
                            rec.get("pair_candidate_rate", 0.0)
                            for rec in decision_steps])) if decision_steps else 0.0,
                        "actions/pair_probability_mass_mean": float(np.mean([
                            rec.get("pair_probability_mass", 0.0)
                            for rec in decision_steps])) if decision_steps else 0.0,
                        "actions/on_policy_pair_selection_rate": float(np.mean([
                            rec.get("selected_is_pair", False)
                            for rec in pure_decision_steps])) if pure_decision_steps else 0.0,
                        "actions/anchor_pair_selection_rate": float(np.mean([
                            rec.get("selected_is_pair", False)
                            for rec in anchor_decision_steps])) if anchor_decision_steps else 0.0,
                        "actions/single_future_best_positive_rate": float(np.mean([
                            rec.get("m2_future_best_reward", 0.0) > 0.0
                            for rec in single_steps])) if single_steps else 0.0,
                        "actions/pair_future_best_positive_rate": float(np.mean([
                            rec.get("m2_future_best_reward", 0.0) > 0.0
                            for rec in pair_steps])) if pair_steps else 0.0,
                        "actions/single_future_best_gain_mean": float(np.mean([
                            rec.get("m2_future_best_reward", 0.0)
                            for rec in single_steps])) if single_steps else 0.0,
                        "actions/pair_future_best_gain_mean": float(np.mean([
                            rec.get("m2_future_best_reward", 0.0)
                            for rec in pair_steps])) if pair_steps else 0.0,
                        "runtime/cycle_collect_s": collect_s,
                        "runtime/cycle_update_s": update_s,
                        "runtime/cycle_postprocess_s": postprocess_s,
                        "runtime/cycle_metrics_s": metrics_s,
                        "runtime/cycle_validation_s": validation_s,
                        "runtime/cycle_total_s": cycle_s,
                        "runtime/preselection_replay_count":
                            diag_payload["preselection_replay_count"],
                        "runtime/selected_action_replay_count":
                            diag_payload["selected_action_replay_count"],
                        "progress/generation_sweep": generation,
                        "gpu/peak_memory_mb": float(monitor.get("gpu_peak_memory_mb", 0.0)),
                    }
                    for operator_name in ("ROUTE", "SEQ_SWAP", "SEQ_INSERT"):
                        cycle_scalars[f"operators/selected_{operator_name.lower()}_count"] = (
                            diag_payload["selected_operator_counts"].get(operator_name, 0))
                    # Emit stable zero-valued series too.  For example JSP/FSP
                    # must show zero ROUTE selections by construction rather
                    # than making the corresponding TensorBoard tag disappear.
                    canonical_action_families = (
                        "single::ROUTE", "single::SEQ_SWAP",
                        "single::SEQ_INSERT", "pair::ROUTE+ROUTE",
                        "pair::ROUTE+SEQ_SWAP", "pair::ROUTE+SEQ_INSERT",
                        "pair::SEQ_SWAP+SEQ_SWAP",
                        "pair::SEQ_SWAP+SEQ_INSERT",
                    )
                    action_family_counts = diag_payload[
                        "selected_action_family_counts"]
                    for family_name in canonical_action_families:
                        count = action_family_counts.get(family_name, 0)
                        family_tag = str(family_name).replace("::", "_").replace("+", "_")
                        safe_family = "".join(
                            char.lower() if char.isalnum() else "_"
                            for char in family_tag).strip("_") or "unknown"
                        cycle_scalars[f"actions/selected_{safe_family}_count"] = count
                    pair_actions = sum(
                        count for family_name, count in
                        action_family_counts.items()
                        if str(family_name).startswith("pair::"))
                    known_actions = sum(
                        action_family_counts.values())
                    cycle_scalars["actions/pair_selection_rate"] = (
                        pair_actions / max(known_actions, 1))
                    for reason, count in diag_payload["execution_failure_counts"].items():
                        safe_reason = "".join(
                            char.lower() if char.isalnum() else "_" for char in str(reason)
                        ).strip("_") or "unknown"
                        cycle_scalars[f"execution_failures/{safe_reason}_count"] = count
                    for tag, value in cycle_scalars.items():
                        tensorboard_writer.add_scalar(tag, float(value), cycle_index)
                    for key, value in prof.items():
                        tensorboard_writer.add_scalar(
                            f"runtime/collect_cpu_{key}_s", float(value), cycle_index)
                    for tag, key in (
                            ("gpu/update_utilization_mean_pct",
                             "gpu_utilization_update_mean_pct"),
                            ("gpu/update_utilization_peak_pct",
                             "gpu_utilization_update_peak_pct")):
                        if monitor.get(key) is not None:
                            tensorboard_writer.add_scalar(tag, float(monitor[key]), cycle_index)
                    if eval_metrics is not None:
                        for key, value in eval_metrics.items():
                            if key not in ("cycle", "seconds") and isinstance(value, (int, float)):
                                tensorboard_writer.add_scalar(
                                    f"validation/{key}", float(value), cycle_index)
                    tensorboard_writer.flush()
                print_progress()
                if diverse_training and tensorboard_writer is not None:
                    trace = getattr(jpol.m2, "probability_trace", None)
                    tensorboard_writer.add_scalar("m2/probability_trace_enabled",
                        int(trace is not None), cycle_index)
                    if trace is not None:
                        tensorboard_writer.add_scalar("m2/probability_trace_strength",
                            float(trace.strength.detach().sigmoid().cpu()), cycle_index)
                        tensorboard_writer.add_scalar(
                            "m2/probability_trace_effective_scale",
                            float(trace.strength.detach().sigmoid().cpu()) *
                            max(float(getattr(jpol.m2, "alpha_fraction", 0.0)), 0.1),
                            cycle_index)
                    for source in ("s0", "pool", "elite", "best"):
                        tensorboard_writer.add_scalar(f"state_pool/batch_{source}_fraction",
                            sum(k == source for k, _, _ in batch) / len(batch), cycle_index)
                    tensorboard_writer.add_scalar("state_pool/stored_states",
                        sum(len(ag.state_pool) for ag in arena), cycle_index)
                    tensorboard_writer.flush()

        promote0 = time.time()
        timing_event("frontier_promote", "START", cycle=cycle_index,
                     generation=generation, graphs=len(active))
        selected_scores, entropies, unique_counts = [], [], []
        for gi in active:
            ag, root = arena[gi], generation_roots[gi]
            retained = retain_four(banks[gi], elites)
            if not retained:
                # Defensive only: K>=2 always returns terminal trajectories.
                retained = [{"state": root, "score": 0.0, "memory_score": 0.0,
                             "n_steps": 1, "n_acted": 0, "terminal": "empty"}]
            chosen, probs, entropy = _sample_candidate(
                retained, random.Random(_seed(seed, generation, ag.iid, "promote")))
            ag.current = chosen["state"]
            all_elites = retain_four(
                retained + [{"state": s, "score": float((ag.s0_schedule.makespan if diverse_training else root.ms) - s.ms),
                             "memory_score": 0.0, "n_steps": 0,
                             "n_acted": 0, "terminal": "archive"}
                            for s in ag.elites], elites)
            ag.elites = [c["state"] for c in all_elites]
            best_candidate = min((c["state"] for c in retained), key=lambda s: s.ms)
            if best_candidate.ms < ag.best.ms:
                ag.best = best_candidate
            selected_scores.append(float(chosen["score"]))
            entropies.append(entropy)
            unique_counts.append(len(retained))
        phase["promote"] = time.time() - promote0
        timing_event("frontier_promote", "END", cycle=cycle_index,
                     generation=generation, seconds=phase["promote"])

        steps = np.asarray([ag.current.search_steps for ag in arena], dtype=np.float64)
        progress_fraction = float(np.mean(np.minimum(steps, float(target_steps))) /
                                  max(float(target_steps), 1.0))
        # The paper x-axis is optimizer cycles, not an extrapolation from how
        # many actions survived early STOP.  Keep its denominator fixed.
        acted = np.asarray([ag.current.acted_steps for ag in arena], dtype=np.float64)
        current_gain = np.asarray([int(ag.s0_schedule.makespan) - ag.current.ms
                                   for ag in arena], dtype=np.float64)
        best_gain = np.asarray([int(ag.s0_schedule.makespan) - ag.best.ms
                                for ag in arena], dtype=np.float64)
        cumulative_best_mean = float(best_gain.mean())
        cumulative_best_sum = float(best_gain.sum())
        fixed_cohort.update(arena)
        fixed_cohort.publish(fixed_writer, cycle_index, arena)
        cumulative_improved_rate = float(np.mean(best_gain > 0))
        rewards = np.asarray(rollout_rewards, dtype=np.float64)
        best_rewards = np.asarray(rollout_best_rewards, dtype=np.float64)
        monitors = [u.get("_monitor", {}) for u in update_rows
                    if u.get("_monitor")]
        row = {
            "generation": generation, "n_active": len(active),
            "cycle_start": generation_cycle_start, "cycle_end": cycle_index,
            "search_steps_min": float(steps.min()),
            "search_steps_mean": float(steps.mean()),
            "search_steps_max": float(steps.max()),
            "acted_steps_mean": float(acted.mean()),
            "current_gain_mean": float(current_gain.mean()),
            "best_gain_mean": cumulative_best_mean,
            "best_gain_max": float(best_gain.max()),
            "cumulative_best_reduction_mean_per_graph": cumulative_best_mean,
            "cumulative_best_reduction_sum": cumulative_best_sum,
            "cumulative_improved_graph_rate": cumulative_improved_rate,
            "selected_reward_mean": float(np.mean(selected_scores)),
            "rollout_reward_mean": (
                float(best_rewards.mean()) if best_rewards.size else 0.0),
            "rollout_reward_max": (
                float(best_rewards.max()) if best_rewards.size else 0.0),
            "rollout_terminal_reward_mean": (
                float(rewards.mean()) if rewards.size else 0.0),
            "rollout_reward_positive_rate": (
                float(np.mean(best_rewards > 0)) if best_rewards.size else 0.0),
            "rollout_reward_zero_rate": (
                float(np.mean(best_rewards == 0)) if best_rewards.size else 0.0),
            "rollout_terminal_positive_rate": (
                float(np.mean(rewards > 0)) if rewards.size else 0.0),
            "selection_entropy": float(np.mean(entropies)),
            "unique_candidates_mean": float(np.mean(unique_counts)),
            "anchors": anchor_counts, "phase_s": phase,
            "elapsed_s": time.time() - wall0,
            "updates": len(update_rows),
            "gpu_peak_memory_mb": max((float(m.get("gpu_peak_memory_mb", 0.0))
                                        for m in monitors), default=0.0),
            "gpu_utilization_pct_sample": float(np.mean([
                m["gpu_utilization_pct_sample"] for m in monitors
                if m.get("gpu_utilization_pct_sample") is not None])) if any(
                    m.get("gpu_utilization_pct_sample") is not None
                    for m in monitors) else None,
            "gpu_update_utilization_mean_pct": float(np.mean([
                m["gpu_utilization_update_mean_pct"] for m in monitors
                if m.get("gpu_utilization_update_mean_pct") is not None])) if any(
                    m.get("gpu_utilization_update_mean_pct") is not None
                    for m in monitors) else None,
            "gpu_update_utilization_peak_pct": max((
                float(m["gpu_utilization_update_peak_pct"]) for m in monitors
                if m.get("gpu_utilization_update_peak_pct") is not None), default=None),
            "bucketed_actor_forward": all(bool(m.get("bucketed_actor_forward"))
                                           for m in monitors) if monitors else False,
        }
        history.append(row)
        completed = int(np.sum(steps >= int(target_steps)))
        avg_gen = row["elapsed_s"] / max(len(history), 1)
        remaining_generations = max(0.0, (int(target_steps) - steps.min()) /
                                    max(float(steps.mean() - (history[-2]["search_steps_mean"]
                                                             if len(history) > 1 else 0.0)),
                                        1.0))
        eta_s = ((time.time() - wall0) / max(cycle_index - cycle_at_start, 1) *
                 max(estimated_total_cycles - cycle_index, 0) if diverse_training
                 else avg_gen * remaining_generations)
        print(f"{log_prefix} generation {generation}: completed={completed}/{len(arena)} "
              f"steps(min/mean/max)={steps.min():.0f}/{steps.mean():.1f}/{steps.max():.0f} "
              f"gain(current/best)={current_gain.mean():+.2f}/{best_gain.mean():+.2f} "
              f"cumulative_best_sum={cumulative_best_sum:+.0f} "
              f"reward(rollout/selected)={row['rollout_reward_mean']:+.3f}/"
              f"{np.mean(selected_scores):+.3f} pos={row['rollout_reward_positive_rate']:.3f} "
              f"unique={np.mean(unique_counts):.2f} sec={sum(phase.values()):.1f} "
              f"ETA~{eta_s / 3600.0:.1f}h phase={phase}", flush=True)
        if tensorboard_writer is not None:
            tb_step = generation + 1
            scalars = {
                "generation/search_steps_min": steps.min(),
                "generation/search_steps_mean": steps.mean(),
                "generation/acted_steps_mean": acted.mean(),
                "generation/current_gain_mean": current_gain.mean(),
                "generation/best_gain_mean": best_gain.mean(),
                "generation/selected_reward_mean": np.mean(selected_scores),
                "generation/selection_entropy": np.mean(entropies),
                "generation/unique_candidates_mean": np.mean(unique_counts),
                "generation/collect_s": phase["collect"],
                "generation/update_s": phase["update"],
                "generation/completed_graphs": completed,
            }
            for tag, value in scalars.items():
                tensorboard_writer.add_scalar(tag, float(value), tb_step)
            tensorboard_writer.flush()
        if checkpoint_cb is not None:
            checkpoint0 = time.time()
            timing_event("checkpoint", "START", cycle=cycle_index,
                         generation=generation)
            checkpoint_cb(arena_state_dict(arena, generation, history, cycle_history,
                                          fixed_cohort.state_dict()),
                          generation)
            phase["checkpoint"] = time.time() - checkpoint0
            timing_event("checkpoint", "END", cycle=cycle_index,
                         generation=generation, seconds=phase["checkpoint"])

    target_complete = all(
        ag.current.search_steps >= int(target_steps) for ag in arena)
    search_budget_complete = bool(
        history and int(history[-1]["generation"]) + 1 >= planned_generations)
    budget_exhausted = bool(
        planned_generations < nominal_generations and search_budget_complete)
    complete = bool(search_budget_complete)
    return {"arena": arena, "history": history, "cycle_history": cycle_history,
            "fixed_cohort": fixed_cohort.state_dict(),
            "initial_validation": initial_validation,
            "complete": complete,
            "target_complete": bool(target_complete),
            "search_budget_complete": search_budget_complete,
            "budget_exhausted": bool(budget_exhausted),
            "requested_cycle_cap": requested_cycle_cap,
            "effective_cycle_limit": estimated_total_cycles,
            "generation": (history[-1]["generation"] if history else -1),
            "target_steps": int(target_steps)}


def write_summary(result, path):
    path = Path(path)
    rows = [{"iid": ag.iid, "src": ag.src,
             "s0_ms": int(ag.s0_schedule.makespan),
             "current_ms": int(ag.current.ms), "best_ms": int(ag.best.ms),
             "search_steps": int(ag.current.search_steps),
             "acted_steps": int(ag.current.acted_steps),
             "best_gain": int(ag.s0_schedule.makespan) - int(ag.best.ms)}
            for ag in result["arena"]]
    payload = {"fixed_cohort": result.get("fixed_cohort"),
               "complete": result["complete"],
               "target_complete": result.get("target_complete", False),
               "search_budget_complete": result.get("search_budget_complete", False),
               "budget_exhausted": result.get("budget_exhausted", False),
               "requested_cycle_cap": result.get("requested_cycle_cap"),
               "effective_cycle_limit": result.get("effective_cycle_limit"),
               "generation": result["generation"],
               "target_steps": result["target_steps"], "graphs": rows,
               "initial_validation": result.get("initial_validation"),
               "history": result["history"],
               "cycle_history": result.get("cycle_history", [])}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
