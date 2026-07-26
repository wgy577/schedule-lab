---
name: scheduling-code-semantics
description: Inspect unfamiliar scheduling, optimization, RL, solver, simulator, or benchmark repositories and extract evidence-grounded problem semantics, objectives, constraints, decisions, environments, validators, and unknowns. Use for JSP, FSP, HFSP, FJSP and variants when Claude Code must navigate code before summarizing it, trace important behavior across files, or verify an existing semantic extraction without editing the repository.
---

# Scheduling Code Semantics

Use a navigation-first, evidence-led workflow. Treat code reachable from actual entrypoints as
stronger evidence than filenames, comments, papers, training logs, or model interpretation.

## Read boundary

- Use only `Glob`, `Grep`, and `Read`.
- If the caller declares Harness-controlled rereading, do not call tools directly. Return the
  requested `read_requests` fields and let the Harness approve, read and inject evidence.
- Never use Bash, Edit, Write, network access, package installation, training, or solver execution.
- Never read `.env`, credentials, checkpoints, model weights, generated results, or large logs.
- Stay inside the repository root.
- Respect the caller's file/read budget. Stop and report `unknown` when the budget is exhausted.

## Workflow

1. Build a compact repository map before making scheduling claims.
   - Identify entrypoints, configuration, parsers, environment/simulator, model, solver,
     training, evaluation, tests, validators, and generated artifacts.
   - Exclude logs, weights, cached data, vendored dependencies, result folders and binary files.
2. Locate semantic anchors.
   - Problem family and instance schema.
   - Objective/reward and terminal metric.
   - Action/decision definition and feasibility mask.
   - State transition or schedule construction.
   - Machine/resource eligibility and capacity.
   - Precedence, release, setup, transport, blocking, no-wait, batching, maintenance or
     domain constraints.
   - Validator/Oracle and test path.
3. Trace each important anchor across at least one caller/callee edge when the behavior depends
   on another function or file. Prefer symbol-targeted reads over whole-file reads.
4. Record an evidence ledger while reading:
   `claim -> path -> line/symbol -> direct|derived -> confidence`.
5. Separate facts into:
   - executable hard constraint;
   - objective or secondary mechanism;
   - controllable decision;
   - conventional construction policy;
   - training-only behavior;
   - evaluation/baseline-only content;
   - paper/comment claim not confirmed by code;
   - unknown or contradiction.
6. Check reachability. A constraint-like function that is never called by the active path is not
   an implemented runtime constraint.
7. Return exactly the schema requested by the caller. Preserve every fixed enum, batch ID and
   field name verbatim. Do not add convenience fields or suffixes.

## Analysis depth routing

Do not read every file line by line. Route each target deliberately:

- **Orientation**: map entrypoints, modules, imports and configuration with short targeted reads.
- **Focused trace**: for a feasibility, objective, decision or Oracle anchor, trace the complete
  active path and record each data transformation.
- **Function micro-analysis**: only for dense core functions, document explicit and implicit
  inputs, assumptions, output/effects, state changes, invariants, callers and callees.
- **Pattern sweep**: after proving one enforcement site, search for semantically related variants
  elsewhere. Start with the exact symbol/pattern and generalize one element per search.

At the end of each focused trace, list the minimum files that are essential to reproduce the
conclusion. Do not label a broad inventory as essential.

## Semantic claim gates

Before accepting an important claim, check:

1. **Evidence**: direct implementation evidence exists.
2. **Reachability**: the active entrypoint can reach the enforcing code.
3. **Applicability**: the condition applies to the relevant instance/candidate, not only training,
   logging, benchmark comparison or dead code.
4. **Control**: identify whether the item is fixed input, deterministic policy, direct decision,
   or verified indirect decision.
5. **Objective relevance**: identify the primary or secondary objective mechanism affected;
   do not equate any time-related variable with optimization leverage.
6. **Oracle coverage**: state whether generation, repair or validation mechanically checks it.

If a required gate fails, downgrade the claim to `unknown`, `partial`, `conflict`, conventional
policy, or experimental-only content. Hard constraints remain mandatory even when objective
leverage is low.

## Correction ledger

Maintain compact corrections during the run:

`earlier claim -> contradicting evidence -> corrected claim -> affected conclusions`.

Never reshape later evidence to preserve an earlier classification. Periodically anchor confirmed
entities, decisions, constraints, objectives and open questions so long call chains do not erase
context.

## Evidence rules

- Cite exact repository-relative paths and the narrowest useful line or symbol.
- Do not use the Skill itself, navigation map, filename or model inference as project evidence.
- Mark derived claims explicitly and include the chain of direct evidence.
- Do not promote a benchmark, comparison algorithm, hardware setting or training hyperparameter
  into the scheduling problem definition.
- Do not treat a conventional earliest-feasible insertion rule as a new decision variable unless
  the code actually exposes that choice.
- Keep hard feasibility and optimization importance separate: a hard constraint may have low
  leverage but must still be validated.

## Bounded reread rule

Request another read only when all are true:

- the unresolved claim affects feasibility, objective, decision space or Oracle validity;
- the target path or symbol is linked by import, call, configuration or data flow;
- the expected evidence is stated before reading;
- the same target has not already been read;
- the remaining budget allows it.

Otherwise classify the item as `unknown`, `partial` or `conflict`.

## Review gate

Before returning, apply the checklist in
[references/review-checklist.md](references/review-checklist.md). Report unresolved conflicts;
never silently choose the more convenient interpretation.
