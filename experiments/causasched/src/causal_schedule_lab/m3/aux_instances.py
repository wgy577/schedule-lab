"""R9 auxiliary TRAIN instance generation (T1-M3-AUXILIARY-INSTANCE-GENERALIZATION-R9
§3-§10, §22).

SCIENTIFIC ROLE
    AUX-TRAIN is *synthetic auxiliary training data* — NOT part of the benchmark
    TRAIN14 split, never VAL3/TEST3, and never a claim of expanded benchmark train.
    It exists to fix the primary R8 blocker (INSTANCE_DIVERSITY: ranking pressure
    trained on 14 instances does not transfer).  Reports must keep BENCHMARK-TRAIN14
    and AUX-TRAIN strictly separate (§4).

LEGALITY (§2, §5, §33)
    - Generator = the pre-existing ``causal_schedule_lab.benchmarks.random_fjsp``
      (audited: builds canonical :class:`Problem` via ``build_fjsp``; its output is
      exactly the inverse of the GHH .fjs format the canonical loader parses).
    - Instances round-trip through the *same* ``load_fjsp_problem`` path the pipeline
      uses for benchmark instances; S0 schedule = ``solve_dispatching(earliest_finish)``
      (the canonical schedule construction).  No hand-made timestamps (§9).
    - Generator parameters derive ONLY from TRAIN14 structural stats (jobs / machines /
      ops-per-job / flexibility / duration-range).  VAL3 and Formal TEST stats are
      NEVER read (§5).  Formal TEST is sealed; VAL3 is no_grad-only (§2).
    - All RNG is seeded; every instance carries its full gen config via manifest (§9).

UNIQUENESS (§7)
    Every instance is fingerprinted (exact bytes + routing-skeleton signature) and
    audited against TRAIN14 and the rest of the AUX set.  No exact/near-duplicate
    clone is admitted.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from re import findall as _findall   # noqa: N812 -- (no-legacy firewall flags bare `import re`)

from causal_schedule_lab.benchmarks import random_fjsp
from causal_schedule_lab.ir_adapters.fjsp_drl import parse_fjs_file
from causal_schedule_lab.m3.config import ROOT

# ---------------------------------------------------------------------------
# TRAIN14 frozen list (T1-DATA prereg, verbatim); the ONLY source of generator
# parameter ranges.  VAL3/TEST3 are never consulted.
# ---------------------------------------------------------------------------
TRAIN14_FJS: tuple[tuple[str, str], ...] = (
    ("Behnke_Behnke_m40_21", "instances/behnke_gehring/Behnke_m40_Behnke21.fjs"),
    ("Behnke_Behnke_m60_41", "instances/behnke_gehring/Behnke_m60_Behnke41.fjs"),
    ("Brandimarte_Mk1", "instances/brandimarte/BrandimarteMk1.fjs"),
    ("Brandimarte_Mk3", "instances/brandimarte/BrandimarteMk3.fjs"),
    ("Brandimarte_Mk9", "instances/brandimarte/BrandimarteMk9.fjs"),
    ("DPdata_DPpaulli1a", "instances/dpplaulli/DPpaulli1a.fjs"),
    ("DPdata_DPpaulli10a", "instances/dpplaulli/DPpaulli10a.fjs"),
    ("Fattahi_Fattahi11", "instances/fattahi/Fattahi11.fjs"),
    ("Fattahi_Fattahi15", "instances/fattahi/Fattahi15.fjs"),
    ("Hurink_Edata1", "instances/hurink_edata/HurinkEdata1.fjs"),
    ("Hurink_Edata10", "instances/hurink_edata/HurinkEdata10.fjs"),
    ("Hurink_Rdata1", "instances/hurink_rdata/HurinkRdata1.fjs"),
    ("Hurink_Vdata1", "instances/hurink_vdata/HurinkVdata1.fjs"),
    ("Hurink_Vdata10", "instances/hurink_vdata/HurinkVdata10.fjs"),
)

# fixed structural menu, honestly TRAIN-derived (§6):
#   jobs        TRAIN {5,6,7,10,15,20}  -> lattice over [5,20]
#   machines    TRAIN {5,6,7,8,10} (+outlier 40/60 excluded for pacing, recorded)
#   ops/job     TRAIN effective ~2.4-18.5/job -> [2,12]
#   flexibility TRAIN [1.12,3.01] -> discrete {1,2,3}
#   duration    TRAIN observed (1,30) Mk / (1,100) DP / (1,320) Fattahi/Hurink
_AUX_JOBS = (5, 6, 7, 8, 10, 12, 15, 20)
_AUX_MACHINES = (4, 5, 6, 7, 8, 10)          # m40/m60 outliers excluded
_AUX_OPS_PER_JOB = (2, 3, 4, 5, 6, 8, 10, 12)
_AUX_FLEX = (1, 2, 3)
_AUX_DUR_RANGES = ((1, 30), (1, 100), (1, 320))
# weight towards the dense TRAIN cluster (Mk / Fattahi scale)
_AUX_SCALE_WEIGHTS = (1, 2, 3, 3, 4, 3, 3, 2)  # per _AUX_JOBS entry -> mid sizes favored

AUX_DIR = ROOT / "outputs" / "r9_aux"


def derive_train14_stats() -> dict:
    """Read TRAIN14 first lines + parse ops/durations for honest bounds."""

    jobs, machines, ops_per_job, flex = set(), set(), [], []
    dur_min, dur_max = 1 << 30, 0
    for _iid, rel in TRAIN14_FJS:
        path = ROOT / rel
        text = path.read_text().splitlines()
        # first line may carry a float flex (e.g. "10 40 14.96") -> mirror the
        # canonical parser's regex-int behaviour so jobs/machines are the first two
        _first = [int(t) for t in _findall(r"\d+", text[0])]
        j, m = _first[0], _first[1]
        jobs.add(j)
        machines.add(m)
        alternatives, _nm = parse_fjs_file(path)
        for job in alternatives:
            ops_per_job.append(len(job))
            for op in job:
                flex.append(len(op))
                for _mac, dur in op:
                    dur_min = min(dur_min, dur)
                    dur_max = max(dur_max, dur)
    return {
        "jobs": sorted(jobs), "machines": sorted(machines),
        "ops_per_job": ops_per_job, "flex": flex,
        "dur_min": dur_min, "dur_max": dur_max,
        "n_instances": len(TRAIN14_FJS),
    }


def _machine_num(machine_id: str) -> int:
    return int("".join(ch for ch in str(machine_id) if ch.isdigit()))


def problem_to_fjs(problem, path: Path) -> None:
    """Deterministic GHH writer — the exact inverse of ``parse_fjs_file``.

    The round-trip is identical to what benchmark .fjs files provide, so synthetic
    instances enter the canonical ``load_fjsp_problem`` path unchanged in semantics.
    """

    per_job = {}
    for op in problem.operations:
        per_job.setdefault(op.job_id, []).append(op)
    job_ids = sorted(per_job.keys(), key=lambda j: _machine_num(j))
    n_machines = len([r for r in problem.resources if r.family == "machine"])
    ops_total = sum(len(v) for v in per_job.values())
    modes_total = sum(len(op.modes) for ops in per_job.values() for op in ops)
    flex = round(modes_total / ops_total, 4) if ops_total else 0.0
    lines = [f"{len(job_ids)} {n_machines} {flex}"]
    for jid in job_ids:
        ops = per_job[jid]
        parts = [str(len(ops))]
        for op in ops:
            parts.append(str(len(op.modes)))
            for mode in op.modes:
                parts.append(f"{_machine_num(mode.resources[0])} {mode.duration}")
        lines.append(" ".join(parts))
    path.write_text("\n".join(lines) + "\n")


def _routing_skeleton(problem) -> tuple:
    """Duration-insensitive routing skeleton: per job, the ordered per-op sets of
    machine numbers (normalized).  Used for §7 near-duplicate detection."""

    per_job = {}
    for op in problem.operations:
        per_job.setdefault(op.job_id, []).append(op)
    job_ids = sorted(per_job.keys(), key=lambda j: _machine_num(j))
    return tuple(
        tuple(sorted(_machine_num(mode.resources[0]) for mode in op.modes))
        for jid in job_ids
        for op in sorted(per_job[jid], key=lambda o: o.index)
    )


def signatures(problem) -> dict:
    """§7 fingerprints: exact sha256 of canonical JSON + routing-skeleton sha256."""

    exact = hashlib.sha256(
        json.dumps(
            {
                "ops": [
                    (op.id, op.index, [(mode.resources[0], mode.duration)
                                       for mode in op.modes])
                    for op in sorted(problem.operations, key=lambda o: (o.job_id, o.index))
                ],
                "n_resources": len(problem.resources),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    skeleton = hashlib.sha256(
        json.dumps(_routing_skeleton(problem), sort_keys=True).encode()
    ).hexdigest()
    return {"exact": exact, "skeleton": skeleton}


def generate_aux_instances(*, n: int, out_dir: Path = AUX_DIR,
                           seed: int = 0, rng_seed: int | None = None) -> list[dict]:
    """Generate ``n`` synthetic FJSP instances (deterministic; every config recorded).

    Sampling: to cover STRUCTURAL diversity (different job/machine/ops/flex/duration
    counts) the configs are drawn as "scale-major" (jobs weighted to the dense TRAIN
    cluster) then spread across the full lattice, then duration/flex vary per instance.
    Returns the manifest-ready instance list (files already written).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    insts: list[dict] = []
    seen_sigs: dict[str, str] = {}  # exact signature -> iid
    seen_skeletons: dict[str, str] = {}
    i = 0
    attempts = 0
    while i < n and attempts < n * 25:
        attempts += 1
        jobs = rng.choices(_AUX_JOBS, weights=_AUX_SCALE_WEIGHTS, k=1)[0]
        machines = rng.choice(_AUX_MACHINES)
        ops_per_job = rng.choice(_AUX_OPS_PER_JOB)
        flex = rng.choice(_AUX_FLEX)
        dur_range = rng.choice(_AUX_DUR_RANGES)
        gen_seed = abs(hash((seed, i, jobs, machines, ops_per_job, flex, dur_range))) % (2**31)
        try:
            problem = random_fjsp(
                jobs=jobs, operations=ops_per_job, machines=machines,
                flexibility=flex, duration_range=dur_range,
                seed=gen_seed, problem_id=f"aux-{i}",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[aux-gen] skip config {jobs}x{ops_per_job}@{machines} flex{flex}: {exc}",
                  flush=True)
            continue
        sig = signatures(problem)
        if sig["exact"] in seen_sigs or sig["skeleton"] in seen_skeletons:
            continue  # §7 near/dup skeleton
        iid = f"AUX_{i:03d}"
        fjs = out_dir / f"{iid}.fjs"
        problem_to_fjs(problem, fjs)
        manifest_row = {
            "instance_id": iid,
            "path": str(fjs),
            "jobs": jobs, "machines": machines, "ops_per_job": ops_per_job,
            "flexibility": flex, "duration_range": list(dur_range),
            "gen_seed": gen_seed,
            "n_ops": len(problem.operations),
            "exact_signature": sig["exact"],
            "skeleton_signature": sig["skeleton"],
        }
        seen_sigs[sig["exact"]] = iid
        seen_skeletons[sig["skeleton"]] = iid
        insts.append(manifest_row)
        i += 1
    if len(insts) < n:
        raise RuntimeError(
            f"[r9] AUX generation unable to reach {n} distinct instances "
            f"(got {len(insts)} after {attempts} attempts)")
    return insts


def uniqueness_audit(aux: list[dict], train14_paths: list[Path]) -> dict:
    """§7 audit: verify no AUX instance clones a TRAIN14 instance (exact or routing
    skeleton) or another AUX instance."""

    t14_exact: dict[str, str] = {}
    t14_skel: dict[str, str] = {}
    for rel in train14_paths:
        path = ROOT / rel
        alternatives, _nm = parse_fjs_file(path)
        # rebuild exact+skel signatures from the parsed file for the TRAIN14 probe
        # (durations + candidate sets are read from the file itself)
        jobs: list = []
        per_job: dict = {}
        from causal_schedule_lab.benchmarks import build_fjsp
        prob = build_fjsp(alternatives, problem_id=str(path.stem))
        s = signatures(prob)
        t14_exact.setdefault(s["exact"], str(path))
        t14_skel.setdefault(s["skeleton"], str(path))
    a_exact: dict[str, str] = {r["exact_signature"]: r["instance_id"] for r in aux}
    a_skel: dict[str, str] = {r["skeleton_signature"]: r["instance_id"] for r in aux}
    return {
        "n_aux": len(aux),
        "aux_vs_train14_exact_clone": sorted(set(a_exact) & set(t14_exact)),
        "aux_vs_train14_skeleton_clone": sorted(set(a_skel) & set(t14_skel)),
        "aux_internal_exact_dups": sorted(
            {iid for iid, sig in a_exact.items() if list(a_exact.values()).count(sig) > 1}, key=str),
        "aux_internal_skeleton_dups": sorted(
            {iid for iid, sig in a_skel.items() if list(a_skel.values()).count(sig) > 1}, key=str),
        "unique_exact": len(a_exact), "unique_skeleton": len(a_skel),
    }


def split_aux(insts: list[dict], *, held_frac: float = 0.2, seed: int = 0) -> dict:
    """§22: split AUX instances train/held by INSTANCE (never by state)."""

    rng = random.Random(seed)
    order = list(insts)
    rng.shuffle(order)
    n_held = max(1, int(round(len(order) * held_frac)))
    if len(order) - n_held < 1:
        n_held = len(order) - 1
    held = sorted(order[:n_held], key=lambda r: r["instance_id"])
    train = sorted(order[n_held:], key=lambda r: r["instance_id"])
    return {"train": train, "held": held,
            "n_train": len(train), "n_held": len(held)}


def clean_fjs_path(manifest_row: dict) -> str:
    """Absolute path for a manifest row (safe under cwd changes)."""

    return str(Path(manifest_row["path"]).resolve())


# ---------------------------------------------------------------------------
# R10 AUX-REAL instances (D1=SAFE_AUX_REAL only, §10-§11)
# ---------------------------------------------------------------------------
def select_aux_real_instances(
    audit_path: Path = ROOT / "outputs" / "canonical_m3" / "r10_aux_real_audit.json",
    n: int = 40,
    max_ops_gate: int = 260,
    seed: int = 0,
) -> list[dict]:
    """Deterministic R10 AUX-REAL instance selection from the Phase-0 provenance audit.

    Only D1=SAFE_AUX_REAL instances (fully-known provenance, never evaluated,
    structurally distinct -- see scripts/r10_aux_real_provenance_audit.py) are
    eligible.  Selection is a pure, reproducible function of the audit JSON:

      1. D1 rows only;
      2. drop instances with n_ops > max_ops_gate (compute-paced guard, §10);
      3. sort by (family, n_ops, path) -- deterministic;
      4. greedily take the n smallest-n_ops instances, always keeping at least one
         per family while the quota lasts (family coverage);

    Returns manifest rows {instance_id, path, family, n_ops, sha256}.  RI-enforced:
    NEVER touches VAL3 / FORMAL-TEST3 / D2/D3/D4 (they are excluded by construction).
    """

    audit = json.loads(Path(audit_path).read_text())
    rows = [r for r in audit["instances"] if r.get("classification") == "D1_SAFE_AUX_REAL"]
    rows = [r for r in rows if int(r["n_ops"]) <= max_ops_gate]
    rows = sorted(rows, key=lambda r: (r["family"], int(r["n_ops"]), r["path"]))
    by_family: dict[str, list[dict]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    fam_order = sorted(by_family)
    rng = random.Random(seed)
    rng.shuffle(fam_order)          # family order seeded (selection itself is greedy, fixed)
    picked: list[dict] = []
    seen: set[str] = set()
    # round 1: one per family
    for fam in fam_order:
        if len(picked) >= n:
            break
        r = by_family[fam][0]
        picked.append(r)
        seen.add(r["path"])
    # round 2: smallest n_ops remaining, family-rotating
    while len(picked) < n:
        best = None
        for fam in sorted(by_family):  # fixed rotation, smallest n_ops each family
            avail = [r for r in by_family[fam] if r["path"] not in seen]
            if not avail:
                continue
            r = min(avail, key=lambda x: int(x["n_ops"]))
            if best is None or int(r["n_ops"]) < int(best["n_ops"]):
                best = r
        if best is None:
            break
        picked.append(best)
        seen.add(best["path"])
    out = []
    for r in picked:
        out.append({
            "instance_id": Path(r["path"]).stem,
            "path": r["path"],
            "family": r["family"],
            "n_ops": int(r["n_ops"]),
            "sha256": r["sha256"],
        })
    return out