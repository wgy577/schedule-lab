"""Canonical appearance-taxonomy source of truth (single authority).

This module is the ONE place that decides which appearance ids are *actionable*
in the current version and which are permanently deprecated.  Every actionable
runtime path (M2 proposal builder, root/candidate construction, M3 action space,
readiness coverage audit) must consult :data:`ACTIVE_APPEARANCE_IDS` /
:func:`is_actionable_appearance` rather than hardcoding its own list.

Taxonomy (frozen by 新表象.md §2 / detector ``symptom-pruning-detector-v3``):

* **ACTIVE** = ``{A1, A2, A3, A4, A6}``  -- the only ids that may become a
  candidate / root / proposal / M3 action.  A6 is auxiliary-only at *emission*
  time (``emit_standalone_A6=False`` in the detector) but is an actionable id
  when it appears, so it stays in the active set.
* **DEPRECATED** = ``{A5, A7, A8, A9, A10}`` -- retained in vocab/catalog for
  *legacy readability only*.  They must NEVER enter an actionable path, no
  matter which (possibly stale) artifact is loaded.

Design invariant (threat model): the live v3 detector never *emits* a
deprecated block, so a deprecated appearance id can only reach an actionable
path by loading a stale pre-v3 artifact (``appearance-pruning-2.0`` /
``detector-v2``, which contained A5/A8 blocks).  The fail-closed gate here makes
such an id non-actionable regardless of the loaded fixture -- swapping the
fixture is data cleanup, not the fix; this module is the fix.

The full embedding vocab ``A1..A10`` still exists in the model/data layers
(``sg_sct_data_v1.APPEARANCE_RULES`` etc.) as a *schema-level* reservation and
is intentionally NOT changed -- reserving a slot in an embedding table is not an
actionable behaviour.
"""

from __future__ import annotations

from typing import Optional

# --- The canonical sets ---------------------------------------------------

#: The only appearance ids permitted on an actionable path.
ACTIVE_APPEARANCE_IDS: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A6")

#: Permanently deprecated; legacy-readable, never actionable.
DEPRECATED_APPEARANCE_IDS: tuple[str, ...] = ("A5", "A7", "A8", "A9", "A10")

#: Full historical vocab (active + deprecated), for schema/embedding reservation
#: and audit only.  Ordered A1..A10.
ALL_APPEARANCE_IDS: tuple[str, ...] = tuple(f"A{i}" for i in range(1, 11))

_ACTIVE_SET = frozenset(ACTIVE_APPEARANCE_IDS)
_DEPRECATED_SET = frozenset(DEPRECATED_APPEARANCE_IDS)

# Internal consistency guard: active and deprecated must partition the known
# vocab with no overlap and no id outside A1..A10.
assert _ACTIVE_SET.isdisjoint(_DEPRECATED_SET), "active/deprecated overlap"
assert _ACTIVE_SET | _DEPRECATED_SET == frozenset(ALL_APPEARANCE_IDS), (
    "active + deprecated must partition A1..A10"
)


# --- Rule extraction ------------------------------------------------------

def appearance_id_of(block_id: str) -> str:
    """Extract the appearance rule id from a block id.

    Block ids are formatted ``f"{rule}:{ordinal:04d}"`` (see
    ``symptom_pruning._emit_block``), e.g. ``"A4:0001" -> "A4"``.  A bare rule id
    (``"A4"``) or an unrecognised string is returned as-is / heads-only, so the
    caller can decide how to treat an out-of-vocab id.
    """
    head = str(block_id).split(":", 1)[0].strip()
    return head


# --- The gate -------------------------------------------------------------

def is_actionable_appearance(block_or_rule_id: str) -> bool:
    """True iff the appearance id may enter an actionable path.

    Accepts either a block id (``"A4:0001"``) or a bare rule id (``"A4"``).
    Fail-closed: anything not explicitly in :data:`ACTIVE_APPEARANCE_IDS`
    (deprecated ids, ``unknown``, malformed, empty) is non-actionable.
    """
    return appearance_id_of(block_or_rule_id) in _ACTIVE_SET


def is_deprecated_appearance(block_or_rule_id: str) -> bool:
    """True iff the appearance id is an explicitly deprecated (legacy) id."""
    return appearance_id_of(block_or_rule_id) in _DEPRECATED_SET


def classify_appearance(block_or_rule_id: str) -> str:
    """Return ``"active"`` / ``"deprecated"`` / ``"unknown"`` for one id.

    ``"unknown"`` is a diagnostic bucket only (out-of-vocab id, e.g. ``"other"``
    or an empty/malformed head).  It is NOT actionable -- only ids explicitly in
    the active set are.
    """
    rule = appearance_id_of(block_or_rule_id)
    if rule in _ACTIVE_SET:
        return "active"
    if rule in _DEPRECATED_SET:
        return "deprecated"
    return "unknown"


def partition_block_ids(
    block_ids,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split an iterable of block ids into ``(actionable, filtered)``.

    ``actionable`` preserves input order and contains only ids whose rule is in
    the active set; ``filtered`` holds every id that was gated out (deprecated
    or unknown), for provenance/audit surfacing.
    """
    actionable: list[str] = []
    filtered: list[str] = []
    for bid in block_ids:
        (actionable if is_actionable_appearance(bid) else filtered).append(str(bid))
    return tuple(actionable), tuple(filtered)


def filtered_reason(block_or_rule_id: str) -> Optional[str]:
    """A short reason string if the id is non-actionable, else ``None``."""
    kind = classify_appearance(block_or_rule_id)
    if kind == "active":
        return None
    if kind == "deprecated":
        return "deprecated_appearance_not_actionable"
    return "unknown_appearance_not_actionable"


__all__ = [
    "ACTIVE_APPEARANCE_IDS",
    "DEPRECATED_APPEARANCE_IDS",
    "ALL_APPEARANCE_IDS",
    "appearance_id_of",
    "is_actionable_appearance",
    "is_deprecated_appearance",
    "classify_appearance",
    "partition_block_ids",
    "filtered_reason",
]
