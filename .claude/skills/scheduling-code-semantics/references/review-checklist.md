# Scheduling repository semantic review checklist

## Navigation

- Active entrypoint and configuration path identified.
- Generated artifacts, logs, weights and historical results excluded.
- Environment, model, solver, training and evaluation roles kept separate.

## Scheduling meaning

- Problem family is supported by instance schema and executable behavior.
- Job/operation/machine/resource entities and indexing conventions are explicit.
- Decisions distinguish direct action from deterministic placement after the action.
- Objective distinguishes terminal metric, reward shaping and experimental reporting.
- Secondary mechanisms that influence the main objective are recorded separately.

## Constraint proof

For every claimed hard constraint:

- identify the enforcing condition or state transition;
- verify it is reachable from the active entrypoint;
- identify whether it applies during generation, repair or final validation;
- distinguish fixed instance data from a controllable variable;
- preserve machine-specific processing time, eligibility and route differences.
- after proving one enforcement point, search exact-to-general for alternate enforcement paths;
- change one search abstraction at a time and classify new matches before widening further.

## Function micro-analysis

For each dense core function selected for deep analysis:

- explicit parameters and implicit state/environment inputs identified;
- assumptions and preconditions recorded;
- outputs, state changes, side effects and postconditions recorded;
- callers and callees traced through the relevant active path;
- at least one invariant or state relationship identified;
- earlier claims corrected if this function contradicts them.

## Oracle proof

- Identify the code that rejects or repairs infeasible candidates.
- Do not call evaluation-only comparison code an Oracle.
- State which constraints are mechanically checked and which remain assumed.

## Evidence integrity

- Every accepted claim has repository-relative path plus line or symbol.
- Derived claims cite all direct links in their reasoning chain.
- Paper/comment claims are labelled separately from code facts.
- Contradictions and missing evidence remain explicit.
- No secret, credential, environment file or model output is quoted.

## Output integrity

- Required schema validates.
- Fixed identifiers and enums are copied exactly.
- No extra fields, Markdown fences or narrative outside the requested object.
- Read count and unresolved items are reported when the schema provides fields for them.
- Minimum essential file set is distinguished from the broader navigation inventory.
- Important claims pass evidence, reachability, applicability, control, objective relevance and
  Oracle coverage gates.
