"""Bounded sibling diversity: sampling, diagnostics, and pure replay memoization.

No rejection sampling, reward bonus, beam expansion, or trajectory removal.
First-action rotations have a separate RNG namespace from M2/pool generation.
A stratum uses the global sibling index, not worker-local arrival order.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
from collections import OrderedDict

import torch


def canonical_root_set(roots):
    """For coverage diagnostics ONLY. Ordered M2 likelihood records stay intact."""
    return tuple(sorted(set(map(str, roots or ()))))


def policy_context_fingerprint(identity, tensors):
    """Hash ordered actions and every actor-visible input, including Memory.

    This is not merely a schedule hash. Different pool ordering, context, masks,
    or logits must not accidentally share a sampling rotation.
    """
    h = hashlib.sha256()
    h.update(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True).encode())
    for name, value in sorted(tensors.items()):
        h.update(name.encode() + b"\0")
        if value is None:
            h.update(b"NONE")
            continue
        t = value.detach().cpu().contiguous()
        h.update(str((str(t.dtype), tuple(t.shape))).encode())
        # Byte view also supports bfloat16 without a NumPy bfloat16 dependency.
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def rotated_action_sample(probs, stratum, context_key):
    """Inverse-CDF draw with q=(U_context + sibling_index/N) mod 1.

    Each sibling retains the categorical marginal. A matching-context subset
    uses its positions on the SAME global grid, not a new worker-local grid.
    Subsets need not exhaust the grid, so different actions are not guaranteed.
    The exact mixture log-prob is still computed by the collector as before.
    """
    common_seed, index, count = map(int, stratum)
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid sibling stratum")
    p = probs.detach().cpu().double().reshape(-1)
    if not p.numel() or not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
        raise ValueError("action probabilities must be finite and nonnegative")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("empty probability mass")
    salt = hashlib.sha256(f"m3-first-v4::{common_seed}::{context_key}".encode()).digest()
    offset = random.Random(int.from_bytes(salt[:16], "big")).random()
    q = (offset + index / count) % 1.0
    cdf = (p / total).cumsum(0)
    cdf[-1] = 1.0
    # right=True skips zero-probability entries even in the q==0 edge case.
    action = int(torch.searchsorted(cdf, torch.tensor(q, dtype=cdf.dtype),
                                   right=True).item())
    return min(action, p.numel() - 1), q


class BoundedReplayCache:
    """Graph/shard-local successful Frozen-Local replay cache.

    Both count and serialized-byte limits are enforced. Only successes are
    memoized; timeouts, exceptions and other failed executions are never cached.
    Cached bytes are decoded for each hit, so branch state remains isolated.
    A graph-scoped lifetime prevents reuse across changing problem definitions.
    """
    def __init__(self, max_entries=128, max_bytes=16 * 1024 * 1024):
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.lookups = self.executions = 0

    def execute(self, execute_fn, executor, problem, schedule, edits, base_ms, base_hash):
        self.lookups += 1
        # Full serialized state + edits, not just operation IDs or action labels.
        # Object identities scope executor configuration/problem to this job.
        key = (id(executor), id(problem), int(base_ms), str(base_hash),
               hashlib.sha256(pickle.dumps((schedule, tuple(edits)), protocol=5)).digest())
        if self.max_entries and self.max_bytes and key in self.entries:
            self.hits += 1
            payload = self.entries.pop(key)
            self.entries[key] = payload
            result, reason = pickle.loads(payload)
            return result, reason, True
        self.executions += 1
        result, reason = execute_fn(executor, problem, schedule, edits, base_ms, base_hash)
        if result is not None and self.max_entries and self.max_bytes:
            payload = pickle.dumps((result, reason), protocol=5)
            size = len(payload)
            if size <= self.max_bytes:
                while self.entries and (len(self.entries) >= self.max_entries or
                                        self.bytes + size > self.max_bytes):
                    _, old = self.entries.popitem(last=False)
                    self.bytes -= len(old)
                self.entries[key] = payload
                self.bytes += size
        return result, reason, False


def sibling_diversity(trajs):
    """Pure-policy diagnostics. Empty or short paths are not fake unique paths."""
    pure = [t for t in trajs if not t.get("is_anchor", False)]
    active = [t for t in pure if t.get("steps")]
    roots = [canonical_root_set(t.get("initial_selected_roots")) for t in active]
    roots = [r for r in roots if r]
    first = [(t["steps"][0].get("state_hash"),
              str(t["steps"][0].get("action_signature", ""))) for t in active]
    out = {
        "pure_trajectory_count": len(pure),
        "pure_first_decision_count": len(active),
        "initial_root_set_count": len(set(roots)),
        "initial_root_set_rate": len(set(roots)) / max(len(roots), 1),
        "first_action_unique_count": len(set(first)),
        "first_action_unique_rate": len(set(first)) / max(len(first), 1),
        "m3_first_stratified_count": sum(
            t["steps"][0].get("m3_sampling_mode") == "rotated_first_step"
            for t in active),
    }
    for depth in (2, 3):
        paths = [tuple((r.get("state_hash"), str(r.get("action_signature", "")))
                       for r in t["steps"][:depth])
                 for t in pure if len(t.get("steps", ())) >= depth]
        # Include resulting hashes, not merely the common initial state.
        states = [tuple(r.get("successor_state_hash") for r in t["steps"][:depth])
                  for t in pure if len(t.get("steps", ())) >= depth and
                  all(r.get("successor_state_hash") for r in t["steps"][:depth])]
        out[f"prefix{depth}_eligible_count"] = len(paths)
        out[f"prefix{depth}_duplicate_rate"] = (
            1.0 - len(set(paths)) / len(paths) if paths else 0.0)
        out[f"state_prefix{depth}_eligible_count"] = len(states)
        out[f"state_prefix{depth}_duplicate_rate"] = (
            1.0 - len(set(states)) / len(states) if states else 0.0)
    lookups = sum(int(t.get("replay_cache_lookups", 0)) for t in trajs)
    hits = sum(int(t.get("replay_cache_hits", 0)) for t in trajs)
    out.update(replay_cache_lookups=lookups, replay_cache_hits=hits,
               replay_cache_hit_rate=hits / max(lookups, 1),
               real_execution_calls=sum(int(t.get("real_execution_calls", 0))
                                        for t in trajs))
    return out
