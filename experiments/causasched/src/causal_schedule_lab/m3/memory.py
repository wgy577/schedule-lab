"""Canonical M3 memory: replay-time collector store (r2) + progressive causal-time
state-gated retrieval (R4, canonical).

VERBATIM extraction (no semantic change):
  - MemoryStore / compute_mem_features                     <- r2
  - _z_stats / _z / _state_dist / ProgressiveMemory /
    compute_prog_mem_features                              <- R4
  - assert_causal_lookahead_free / assert_unseen_state_zero <- R4
Memory is *evidence only*: never reward / ground truth / causal truth / legality
/ authority.  ProgressiveMemory is the canonical retrieval contract (causal-time
written_at_step < current_step, sibling-clean, same-instance future impossible).
"""

from __future__ import annotations

import numpy as np
import torch

from .config import MEM_FEAT_DIM, MEM_TAU_STD, MEM_TOP_N_STATES, STATE_FEAT_DIM


# ---------------------------------------------------------------------------
# Memory store (evidence-only, replay-time).  [r2] (write contract §8)
# ---------------------------------------------------------------------------
class MemoryStore:
    """Historical Frozen-Local outcome records keyed by full provenance triple.

    Retrieval is leave-one-state-out (excludes the query state), same-instance only,
    aggregated over structural buckets (type, role, src->tgt).  Memory is a *feature*
    (retrieval evidence), never a reward / ground truth / legality / causal truth.
    """

    def __init__(self):
        self.records = []                 # list of record dicts
        self.triple_index = {}            # (iid, state_hash, sig) -> record  (uniqueness)
        self.key_index = {}               # bucket key -> list[record]
        self.bare_sig_index = {}          # sig -> list[iid]  (collision diagnostic)

    def add(self, rec):
        key = (rec["instance_id"], rec["state_hash"], rec["proposal_signature"])
        assert key not in self.triple_index, f"duplicate provenance triple {key}"
        self.triple_index[key] = rec
        self.records.append(rec)
        for k in (rec["fine_key"], rec["coarse_key"]):
            self.key_index.setdefault(k, []).append(rec)
        self.bare_sig_index.setdefault(rec["proposal_signature"], []).append(rec["instance_id"])

    def bare_sig_collisions(self):
        return {s: iids for s, iids in self.bare_sig_index.items() if len(set(iids)) > 1}

    def features(self, iid, state_hash, queries):
        """queries: list of dicts {type, role, src, tgt}.  -> [N, 6] numpy array.

        fine bucket = (type, role, src, tgt) for single, (type, role) for pair;
        coarse bucket = (type, role).  Only feasible records (true_U not None)
        contribute to success/mean/max; support counts feasible matching records.
        All zeros when no evidence (also yields a natural 'without memory' ablation).
        """
        out = np.zeros((len(queries), MEM_FEAT_DIM), dtype=np.float32)
        for n, q in enumerate(queries):
            fine_key = (q["type"], q["role"], q["src"], q["tgt"]) if q["type"] == "single" \
                else (q["type"], q["role"])
            coarse_key = (q["type"], q["role"])
            fine = [r for r in self.key_index.get(fine_key, ())
                    if r["instance_id"] == iid and r["state_hash"] != state_hash
                    and r["true_U"] is not None]
            coarse = [r for r in self.key_index.get(coarse_key, ())
                      if r["instance_id"] == iid and r["state_hash"] != state_hash
                      and r["true_U"] is not None]
            if fine:
                U = np.array([r["true_U"] for r in fine], dtype=np.float32)
                out[n, 0] = len(fine)
                out[n, 1] = float((U > 0).mean())
                out[n, 2] = float(U.mean())
                out[n, 3] = float(U.max())
            if coarse:
                Uc = np.array([r["true_U"] for r in coarse], dtype=np.float32)
                out[n, 4] = len(coarse)
                out[n, 5] = float((Uc > 0).mean())
        return out


def compute_mem_features(store, state_examples):
    """Attach r2 memory features to every replay example (leave-one-state-out)."""
    for ex in state_examples:
        queries = [{"type": t, "role": r_, "src": s_, "tgt": g_}
                   for t, r_, s_, g_ in zip(ex["type"], ex["role"], ex["src"], ex["tgt"])]
        m = store.features(ex["iid"], ex["state_hash"], queries)
        ex["mem_feats"] = torch.tensor(m, dtype=torch.float32)


# ---------------------------------------------------------------------------
# state similarity helpers (policy-observable only; never true_U)  [R4]
# ---------------------------------------------------------------------------
def _z_stats(state_feats):
    """mean/std over the TRAIN replay state_feat matrix [Nstates,7]."""
    a = np.asarray(state_feats, dtype=np.float64)
    mu = a.mean(axis=0)
    sd = a.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def _z(vec, mu, sd):
    return (np.asarray(vec, dtype=np.float64) - mu) / sd


def _state_dist(a, b, mu, sd):
    """Euclidean in standardized state space."""
    return float(np.linalg.norm(_z(a, mu, sd) - _z(b, mu, sd)))


# ---------------------------------------------------------------------------
# memory reliability gate (canonical R8 §23-§24)  [policy-observable only]
# ---------------------------------------------------------------------------
# g_mem ∈ [0,1] = coverage × similarity of the retrieval that produced an
# embedding:  coverage = n_accepted_states / MEM_TOP_N_STATES,  similarity =
# 1/(1+mean_z_dist).  Weak / absent retrieval -> g_mem → 0 (no historical-average
# fallback; ProgressiveMemory still returns zeros when nothing passes the gate).
def memory_reliability_gate(n_neighbors, mean_dist, top_n=MEM_TOP_N_STATES):
    if n_neighbors <= 0:
        return 0.0
    d = max(float(mean_dist), 0.0)
    coverage = min(float(n_neighbors) / max(top_n, 1), 1.0)
    similarity = 1.0 / (1.0 + d)
    return float(np.clip(coverage * similarity, 0.0, 1.0))


def _gate_from_diag(diag):
    """g_mem from a `features(return_diag=True)` result dict (R8 §24)."""
    if not diag:
        return 0.0
    n = int(diag.get("n_neighbors", 0))
    d = float(diag.get("mean_dist", 0.0))
    return memory_reliability_gate(n, d)


# ---------------------------------------------------------------------------
# Progressive Memory (canonical: causal-time + state-gated retrieval)  [R4]
# ---------------------------------------------------------------------------
# Features per (state, proposal): 6 dims over the records that pass BOTH the
# causal-time filter AND the state-similarity gate, aggregated by bucket:
#   [fine_support, fine_success_rate, fine_mean_gain,
#    coarse_support, coarse_success_rate, coarse_mean_gain]
# fine = (type, role, src, tgt) for single else (type, role);
# coarse = (type, role).  Gains are similarity-weighted (w = 1/(1+d)).
# zero candidates -> all-zero vector (never fall back to whole instance).
class ProgressiveMemory:
    """Retrieval evidence WITH explicit causal-time semantics.

    Written contract (directive Section 20):
      (instance_id, episode_id, state_hash, state_feat, proposal_signature,
       proposal_type, role, src, tgt, true_U, outcome, trajectory_step,
       written_at_step, successor_state_hash)

    Access rule for a query at (episode e, step t):
      - cross-episode: records of episodes ENTIRELY BEFORE e (all its states),
        available regardless of step (the episode is complete).
      - current-episode: only EPHEMERAL executed-transition records with
        written_at_step < t (the agent's own past actions).
      - FORBIDDEN: any record with episode > e, and any current-episode record
        with written_at_step >= t (future).  Same-instance future is therefore
        structurally impossible here (one episode per instance in TRAIN).
    State gate: candidates are deduped by state_hash; only states within
    MEM_TAU_STD (z-space) survive; at most MEM_TOP_N_STATES nearest pass.
    """

    def __init__(self, state_feats=None):
        self.cross_episodes = {}        # episode_id -> list[record]
        self.executed = {}              # instance_id -> {step: record}
        self.next_episode = 0
        self.z_mean = None
        self.z_sd = None
        if state_feats is not None:
            self.set_state_stats(state_feats)
        self._diag = {"neighbors_per_query": [], "zero_queries": 0, "n_queries": 0,
                      "rejected_same_instance_dump_records": 0, "rejected_total_records": 0}

    # -- stats --------------------------------------------------------------
    def set_state_stats(self, state_feats):
        self.z_mean, self.z_sd = _z_stats(state_feats)

    # -- writes -------------------------------------------------------------
    def start_episode(self):
        eid = self.next_episode
        self.cross_episodes[eid] = []
        return eid

    def add_episode_record(self, rec, episode_id):
        """Full-evaluation record of a state inside the given episode."""
        assert rec["episode_id"] == episode_id
        self.cross_episodes.setdefault(episode_id, []).append(rec)

    def add_executed(self, iid, step, rec):
        """Current-episode executed-transition record (written AFTER execution)."""
        assert rec["written_at_step"] == step
        self.executed.setdefault(iid, {})[step] = rec

    def complete_episode(self):
        self.next_episode += 1

    # -- availability --------------------------------------------------------
    def _candidates(self, episode_id, step, iid):
        out = []
        # Episode ids are causal timestamps, not a promise that the mapping is
        # dense.  Validation deliberately uses a large, future-only id; walking
        # every absent integer before it made each query perform ~1M empty dict
        # lookups.  Iterating the actually stored earlier episodes preserves the
        # exact old numeric episode order and therefore the retrieval/tie order.
        earlier = sorted(eid for eid in self.cross_episodes if eid < episode_id)
        for eid in earlier:
            out.extend(self.cross_episodes[eid])
        exe = self.executed.get(iid, {})
        for s, rec in exe.items():
            if s < step:
                out.append(rec)
        return out

    # -- retrieval ----------------------------------------------------------
    def _accepted_states(self, candidates, state_feat, top_n=None, tau=None):
        """Dedup by state_hash; keep <= top_n nearest within tau (z-space)."""
        if not candidates:
            return []
        top_n = MEM_TOP_N_STATES if top_n is None else top_n
        tau = MEM_TAU_STD if tau is None else tau
        best_state = {}
        for rec in candidates:
            h = rec["state_hash"]
            if h not in best_state or rec["state_feat"] is not None:
                if h not in best_state:
                    best_state[h] = rec["state_feat"]
        rows = []
        for h, sf in best_state.items():
            if sf is None:
                continue
            d = _state_dist(state_feat, sf, self.z_mean, self.z_sd)
            rows.append((d, h, sf))
        rows.sort(key=lambda x: x[0])
        acc = []
        for d, h, sf in rows:
            if d > tau:
                break
            acc.append((d, h, sf))
            if len(acc) >= top_n:
                break
        return acc

    def features(self, iid, episode_id, step, state_feat, queries, return_diag=False):
        """queries: list[{type, role, src, tgt}] -> [N, 6]."""
        cands = self._candidates(episode_id, step, iid)
        if not cands:
            if return_diag:
                z = np.zeros((len(queries), MEM_FEAT_DIM), dtype=np.float32)
                self._diag["n_queries"] += len(queries)
                self._diag["zero_queries"] += len(queries)
                self._diag["neighbors_per_query"].extend([0] * len(queries))
                return z, {"n_neighbors": 0, "mean_dist": 0.0, "zero": True}
            return np.zeros((len(queries), MEM_FEAT_DIM), dtype=np.float32)
        # causal-time safety (cheap, deterministic)
        for rec in cands:
            if rec["episode_id"] == episode_id and rec["written_at_step"] >= step:
                raise AssertionError("LOOKAHEAD: current-episode future record leaked")
            assert rec["episode_id"] <= episode_id, "future episode leaked"
        acc = self._accepted_states(cands, state_feat)
        if not acc:
            if return_diag:
                self._diag["n_queries"] += len(queries)
                self._diag["zero_queries"] += len(queries)
                self._diag["neighbors_per_query"].extend([0] * len(queries))
                self._diag["rejected_total_records"] += len(cands)
                if any(rec["episode_id"] == episode_id for rec in cands):
                    self._diag["rejected_same_instance_dump_records"] += len(cands)
                return np.zeros((len(queries), MEM_FEAT_DIM), dtype=np.float32), \
                    {"n_neighbors": 0, "mean_dist": 0.0, "zero": True, "rejected": len(cands)}
        acc_hash = {h for _, h, _ in acc}
        by_state = {}
        for rec in cands:
            if rec["state_hash"] in acc_hash:
                by_state.setdefault(rec["state_hash"], []).append(rec)
        if return_diag:
            self._diag["n_queries"] += len(queries)
            self._diag["neighbors_per_query"].extend([len(by_state)] * len(queries))
        dmap = {h: d for d, h, _ in acc}
        wmap = {h: 1.0 / (1.0 + d) for d, h, _ in acc}
        out = np.zeros((len(queries), MEM_FEAT_DIM), dtype=np.float32)
        for n, q in enumerate(queries):
            fine_key = (q["type"], q["role"], q["src"], q["tgt"]) if q["type"] == "single" \
                else (q["type"], q["role"])
            coarse_key = (q["type"], q["role"])
            fine, coarse = [], []
            for h, recs in by_state.items():
                w = wmap[h]
                for rec in recs:
                    if rec["true_U"] is None:
                        continue
                    if rec.get("fine_key") == fine_key:
                        fine.append((w, rec["true_U"]))
                    if rec.get("coarse_key") == coarse_key:
                        coarse.append((w, rec["true_U"]))
            for lab, pool in (("fine", fine), ("coarse", coarse)):
                if not pool:
                    continue
                st = 0 if lab == "fine" else 3
                out[n, st] = len(pool)
                wsum = sum(w for w, _ in pool)
                out[n, st + 1] = sum(w * float(u > 0) for w, u in pool) / wsum
                out[n, st + 2] = sum(w * float(u) for w, u in pool) / wsum
        if return_diag:
            return out, {"n_neighbors": len(acc), "mean_dist": float(np.mean([d for d, _, _ in acc]))
                         if acc else 0.0, "zero": False,
                         "accepted_states": [h for _, h, _ in acc]}
        return out

    def retrieval_gate(self, iid, episode_id, step, state_feat):
        """R8 §24: g_mem ∈ [0,1] from policy-observable retrieval quality at runtime
        (same code path as the train-time diag: candidate set -> state gate -> gate).
        Weak / absent retrieval -> 0.0; NO historical-average fallback."""
        cands = self._candidates(episode_id, step, iid)
        acc = self._accepted_states(cands, state_feat)
        if not acc:
            return 0.0
        n = len(acc)
        d = float(np.mean([d_ for d_, _, _ in acc]))
        return memory_reliability_gate(n, d)

    def proposal_mem_evidence(self, iid, episode_id, step, state_feat, signature,
                              qtype, role, src, tgt):
        """R18 §8-9 Proposal-level Memory evidence (evidence ONLY, never authority).

        Priority key (§9): (instance_id, state_hash, proposal_signature) -- EXACT
        proposal-signature records at similar states first; if no exact proposal
        memory exists, fall back to explicitly-provenance fine-key (structurally
        similar) evidence at the SAME similar states.  Records are weighted by
        1/(1+state_dist) inside the same causal-time-safe candidate set as
        `features`; current/future-episode records raise the LOOKAHEAD guard.

        Never vetoes a G_prop>0 Proposal (§8).  Never reads future true_U (§3-5).
        Returns dict: support, success, mean_gain, n_exact, n_fine, mean_w, exact.
        """
        cands = self._candidates(episode_id, step, iid)
        for rec in cands:
            if rec["episode_id"] == episode_id and rec["written_at_step"] >= step:
                raise AssertionError("LOOKAHEAD: current-episode future record leaked")
            assert rec["episode_id"] <= episode_id, "future episode leaked"
        acc = self._accepted_states(cands, state_feat)
        if not acc:
            return {"support": 0.0, "success": 0.0, "mean_gain": 0.0,
                    "n_exact": 0, "n_fine": 0, "mean_w": 0.0, "exact": False}
        acc_hash = {h for _, h, _ in acc}
        by_state = {}
        for rec in cands:
            if rec["state_hash"] in acc_hash:
                by_state.setdefault(rec["state_hash"], []).append(rec)
        wmap = {h: 1.0 / (1.0 + d) for d, h, _ in acc}
        fine_key = (qtype, role, src, tgt) if qtype == "single" else (qtype, role)
        exact_pool, fine_pool = [], []
        for h, recs in by_state.items():
            w = wmap[h]
            for rec in recs:
                if rec.get("true_U") is None:
                    continue
                if signature is not None and rec.get("proposal_signature") == signature:
                    exact_pool.append((w, float(rec["true_U"])))
                if rec.get("fine_key") == fine_key:
                    fine_pool.append((w, float(rec["true_U"])))

        def _agg(pool):
            if not pool:
                return (0, 0.0, 0.0, 0.0)
            ws = sum(w for w, _ in pool)
            return (len(pool), ws,
                    sum(w * float(u > 0) for w, u in pool) / ws,
                    sum(w * float(u) for w, u in pool) / ws)

        n_exact, ws_exc, succ_exc, gain_exc = _agg(exact_pool)
        n_fine, ws_fin, succ_fin, gain_fin = _agg(fine_pool)
        if n_exact > 0:
            return {"support": float(ws_exc), "success": float(succ_exc),
                    "mean_gain": float(gain_exc), "n_exact": n_exact,
                    "n_fine": n_fine, "mean_w": float(ws_exc / n_exact), "exact": True}
        return {"support": float(ws_fin), "success": float(succ_fin),
                "mean_gain": float(gain_fin), "n_exact": 0, "n_fine": n_fine,
                "mean_w": float(ws_fin / n_fine) if n_fine else 0.0, "exact": False}

    def diag_summary(self):
        nq = max(self._diag["n_queries"], 1)
        nb = np.array(self._diag["neighbors_per_query"], dtype=np.float64) if \
            self._diag["neighbors_per_query"] else np.zeros(1)
        return {
            "n_queries": self._diag["n_queries"],
            "zero_ratio": self._diag["zero_queries"] / nq,
            "mean_neighbors_per_query": float(nb.mean()),
            "median_neighbors_per_query": float(np.median(nb)) if len(nb) else 0.0,
            "p90_neighbors_per_query": float(np.percentile(nb, 90)) if len(nb) else 0.0,
            "rejected_same_instance_dump_records": self._diag["rejected_same_instance_dump_records"],
            "rejected_total_records": self._diag["rejected_total_records"],
        }


def compute_prog_mem_features(examples, progmem):
    """Attach corrected progressive features to every replay example.
    Collects neighbor-coverage diagnostics while computing; attaches the R8 §24
    memory reliability gate (g_mem) per state."""
    for ex in examples:
        N = len(ex["metas"])
        queries = [{"type": ex["type"][k], "role": ex["role"][k],
                    "src": ex["src"][k], "tgt": ex["tgt"][k]} for k in range(N)]
        m, diag = progmem.features(ex["iid"], ex["ep_id"], ex["tstep"],
                                   ex["state_feat"].tolist(), queries, return_diag=True)
        ex["prog_mem_feats"] = torch.tensor(m, dtype=torch.float32)
        ex["mem_gate"] = _gate_from_diag(diag)


# ---------------------------------------------------------------------------
# memory-causal regression assertions  [R4]
# ---------------------------------------------------------------------------
def assert_causal_lookahead_free(state_examples, progmem, store):
    """(a) every training state's features are step-causal by construction, and
    (b) an explicit S0 probe never retrieves same-instance (or future-episode)
    records.  Fails loudly on lookahead (directive Section 17)."""
    for ex in state_examples:
        N = len(ex["metas"])
        queries = [{"type": ex["type"][k], "role": ex["role"][k],
                    "src": ex["src"][k], "tgt": ex["tgt"][k]} for k in range(N)]
        cands = progmem._candidates(ex["ep_id"], ex["tstep"], ex["iid"])
        for rec in cands:
            assert rec["episode_id"] <= ex["ep_id"], "future episode leaked"
            if rec["episode_id"] == ex["ep_id"]:
                assert rec["written_at_step"] < ex["tstep"], \
                    f"LOOKAHEAD leak: {rec['written_at_step']} >= {ex['tstep']}"
        # S0 probe: never retrieve current-episode (same-instance) records;
        # the FIRST episode's S0 has no history at all -> memory must be zero.
        if ex.get("s0", False):
            m0, d0 = progmem.features(ex["iid"], ex["ep_id"], 0,
                                      ex["state_feat"].tolist(), queries, return_diag=True)
            for rec in cands:
                if rec["episode_id"] == ex["ep_id"]:
                    raise AssertionError(f"S0 lookahead: {ex['iid']} retrieved current-episode record")
            if ex["ep_id"] == 0:
                assert np.all(m0 == 0), f"S0 of first episode must have zero memory: {ex['iid']}"
            progmem._diag["s0_probe_states"] = progmem._diag.get("s0_probe_states", 0) + 1
    return {"causal_assertion": "PASS", "n_states_checked": len(state_examples)}


def assert_unseen_state_zero(state_examples, progmem, iid=None):
    """Directive Section 18 regression: an unseen (far) state must retrieve
    zero evidence (no state-gate violation, no bucket dump)."""
    ex0 = state_examples[0]
    iid = iid or ex0["iid"]
    N = len(ex0["metas"])
    queries = [{"type": ex0["type"][k], "role": ex0["role"][k],
                "src": ex0["src"][k], "tgt": ex0["tgt"][k]} for k in range(N)]
    far = np.zeros(STATE_FEAT_DIM, dtype=np.float64)
    far[0] = 50.0                              # implausibly good makespan ratio
    m, d = progmem.features(iid, ex0["ep_id"], 0, far.tolist(), queries, return_diag=True)
    assert np.all(m == 0), "unseen-state regression: retrieved non-zero evidence"
    return {"unseen_state_zero": "PASS", "checked_queries": N,
            "rejected": d.get("rejected", 0)}


def replay_primary_key_uniqueness(store, ep_id_of):
    """Directive Section 20: (instance, episode, state_hash, sig) unique."""
    seen = set()
    n = 0
    for rec in store.records:
        key = (rec["instance_id"], ep_id_of[rec["instance_id"]], rec["state_hash"],
               rec["proposal_signature"])
        if key in seen:
            raise AssertionError(f"duplicate primary key {key}")
        seen.add(key)
        n += 1
    return {"n_records": n, "n_unique_primary_keys": len(seen)}
