"""SQLite-backed versioned property graph and evidence store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field


ReviewStatus = Literal[
    "proposed",
    "canonicalized",
    "source_verified",
    "conflict_checked",
    "active",
    "rejected",
    "superseded",
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GraphNode(FrozenModel):
    canonical_id: str
    node_type: str
    name: str
    summary: str
    scope: str = "global"
    version: int = Field(default=1, ge=1)
    review_status: ReviewStatus = "proposed"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source_ids: tuple[str, ...] = ()
    valid_from: str | None = None
    valid_to: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(FrozenModel):
    edge_id: str
    source_id: str
    relation: str
    target_id: str
    scope: str = "global"
    version: int = Field(default=1, ge=1)
    review_status: ReviewStatus = "proposed"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source_ids: tuple[str, ...] = ()
    valid_from: str | None = None
    valid_to: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceChunk(FrozenModel):
    chunk_id: str
    layer: int = Field(ge=0, le=4)
    title: str
    content: str
    source_kind: str
    source_uri: str | None = None
    scope: str = "global"
    review_status: ReviewStatus = "proposed"
    node_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class InterventionRecord(FrozenModel):
    experiment_id: str
    problem_family: str
    instance_hash: str
    incumbent_hash: str
    state_hash: str
    operator_id: str
    factor_id: str
    mechanism_id: str
    objective_name: str = "makespan"
    mechanism_delta: float
    objective_gain: float
    valid: bool
    validation_fidelity: Literal["static", "light", "full"]
    runtime_seconds: float = Field(ge=0.0)
    token_cost: int = Field(default=0, ge=0)
    failure_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryStats(FrozenModel):
    nodes: int
    edges: int
    aliases: int
    evidence_chunks: int
    experiments: int
    active_nodes: int
    active_edges: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


class SQLiteGraphStore:
    """Authoritative graph plus immutable evidence and intervention history."""

    SCHEMA_VERSION = 2

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_versions (
                    schema_version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    canonical_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    review_status TEXT NOT NULL,
                    confidence REAL,
                    source_ids_json TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    metadata_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS edges (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES nodes(canonical_id),
                    relation TEXT NOT NULL,
                    target_id TEXT NOT NULL REFERENCES nodes(canonical_id),
                    scope TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    review_status TEXT NOT NULL,
                    confidence REAL,
                    source_ids_json TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    metadata_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS edges_source_idx
                    ON edges(source_id, relation, review_status);
                CREATE INDEX IF NOT EXISTS edges_target_idx
                    ON edges(target_id, relation, review_status);

                CREATE TABLE IF NOT EXISTS aliases (
                    alias_normalized TEXT NOT NULL,
                    canonical_id TEXT NOT NULL REFERENCES nodes(canonical_id),
                    scope TEXT NOT NULL,
                    alias_original TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(alias_normalized, canonical_id, scope)
                );

                CREATE TABLE IF NOT EXISTS evidence_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    layer INTEGER NOT NULL CHECK(layer BETWEEN 0 AND 4),
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_uri TEXT,
                    scope TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    node_ids_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
                    chunk_id UNINDEXED,
                    title,
                    content,
                    scope,
                    node_ids,
                    tokenize='unicode61'
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    problem_family TEXT NOT NULL,
                    instance_hash TEXT NOT NULL,
                    incumbent_hash TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    factor_id TEXT NOT NULL,
                    mechanism_id TEXT NOT NULL,
                    objective_name TEXT NOT NULL,
                    mechanism_delta REAL NOT NULL,
                    objective_gain REAL NOT NULL,
                    valid INTEGER NOT NULL,
                    validation_fidelity TEXT NOT NULL,
                    runtime_seconds REAL NOT NULL,
                    token_cost INTEGER NOT NULL,
                    failure_reason TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS experiments_scope_idx
                    ON experiments(
                        problem_family, instance_hash, mechanism_id, factor_id
                    );
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO memory_versions(schema_version, applied_at)
                VALUES (?, ?)
                """,
                (self.SCHEMA_VERSION, _now()),
            )

    def upsert_node(self, node: GraphNode) -> bool:
        payload = node.model_dump(mode="json")
        content_hash = _hash(payload)
        now = _now()
        with self.connection() as connection:
            previous = connection.execute(
                "SELECT content_hash FROM nodes WHERE canonical_id = ?",
                (node.canonical_id,),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO nodes(
                    canonical_id, node_type, name, summary, scope, version,
                    review_status, confidence, source_ids_json, valid_from,
                    valid_to, metadata_json, content_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_id) DO UPDATE SET
                    node_type=excluded.node_type,
                    name=excluded.name,
                    summary=excluded.summary,
                    scope=excluded.scope,
                    version=excluded.version,
                    review_status=excluded.review_status,
                    confidence=excluded.confidence,
                    source_ids_json=excluded.source_ids_json,
                    valid_from=excluded.valid_from,
                    valid_to=excluded.valid_to,
                    metadata_json=excluded.metadata_json,
                    content_hash=excluded.content_hash,
                    updated_at=excluded.updated_at
                """,
                (
                    node.canonical_id,
                    node.node_type,
                    node.name,
                    node.summary,
                    node.scope,
                    node.version,
                    node.review_status,
                    node.confidence,
                    _json(node.source_ids),
                    node.valid_from,
                    node.valid_to,
                    _json(node.metadata),
                    content_hash,
                    now,
                    now,
                ),
            )
        return previous is None or previous["content_hash"] != content_hash

    def upsert_edge(self, edge: GraphEdge) -> bool:
        payload = edge.model_dump(mode="json")
        content_hash = _hash(payload)
        now = _now()
        with self.connection() as connection:
            previous = connection.execute(
                "SELECT content_hash FROM edges WHERE edge_id = ?",
                (edge.edge_id,),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO edges(
                    edge_id, source_id, relation, target_id, scope, version,
                    review_status, confidence, source_ids_json, valid_from,
                    valid_to, metadata_json, content_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(edge_id) DO UPDATE SET
                    source_id=excluded.source_id,
                    relation=excluded.relation,
                    target_id=excluded.target_id,
                    scope=excluded.scope,
                    version=excluded.version,
                    review_status=excluded.review_status,
                    confidence=excluded.confidence,
                    source_ids_json=excluded.source_ids_json,
                    valid_from=excluded.valid_from,
                    valid_to=excluded.valid_to,
                    metadata_json=excluded.metadata_json,
                    content_hash=excluded.content_hash,
                    updated_at=excluded.updated_at
                """,
                (
                    edge.edge_id,
                    edge.source_id,
                    edge.relation,
                    edge.target_id,
                    edge.scope,
                    edge.version,
                    edge.review_status,
                    edge.confidence,
                    _json(edge.source_ids),
                    edge.valid_from,
                    edge.valid_to,
                    _json(edge.metadata),
                    content_hash,
                    now,
                    now,
                ),
            )
        return previous is None or previous["content_hash"] != content_hash

    def add_alias(self, alias: str, canonical_id: str, *, scope: str = "global") -> None:
        normalized = " ".join(alias.lower().split())
        with self.connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO aliases(
                    alias_normalized, canonical_id, scope, alias_original, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (normalized, canonical_id, scope, alias, _now()),
            )

    def resolve_alias(self, alias: str, *, scope: str | None = None) -> tuple[str, ...]:
        normalized = " ".join(alias.lower().split())
        with self.connection() as connection:
            if scope is None:
                rows = connection.execute(
                    """
                    SELECT canonical_id FROM aliases
                    WHERE alias_normalized = ?
                    ORDER BY canonical_id
                    """,
                    (normalized,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT canonical_id FROM aliases
                    WHERE alias_normalized = ? AND scope IN (?, 'global')
                    ORDER BY scope DESC, canonical_id
                    """,
                    (normalized, scope),
                ).fetchall()
        return tuple(dict.fromkeys(row["canonical_id"] for row in rows))

    def add_evidence(self, chunk: EvidenceChunk) -> bool:
        payload = chunk.model_dump(mode="json")
        content_hash = _hash(payload)
        with self.connection() as connection:
            previous = connection.execute(
                "SELECT content_hash FROM evidence_chunks WHERE chunk_id = ?",
                (chunk.chunk_id,),
            ).fetchone()
            if previous is not None and previous["content_hash"] == content_hash:
                return False
            if previous is not None:
                connection.execute(
                    "DELETE FROM evidence_fts WHERE chunk_id = ?",
                    (chunk.chunk_id,),
                )
            connection.execute(
                """
                INSERT INTO evidence_chunks(
                    chunk_id, layer, title, content, source_kind, source_uri,
                    scope, review_status, node_ids_json, metadata_json,
                    content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    layer=excluded.layer,
                    title=excluded.title,
                    content=excluded.content,
                    source_kind=excluded.source_kind,
                    source_uri=excluded.source_uri,
                    scope=excluded.scope,
                    review_status=excluded.review_status,
                    node_ids_json=excluded.node_ids_json,
                    metadata_json=excluded.metadata_json,
                    content_hash=excluded.content_hash
                """,
                (
                    chunk.chunk_id,
                    chunk.layer,
                    chunk.title,
                    chunk.content,
                    chunk.source_kind,
                    chunk.source_uri,
                    chunk.scope,
                    chunk.review_status,
                    _json(chunk.node_ids),
                    _json(chunk.metadata),
                    content_hash,
                    _now(),
                ),
            )
            connection.execute(
                """
                INSERT INTO evidence_fts(chunk_id, title, content, scope, node_ids)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.title,
                    chunk.content,
                    chunk.scope,
                    " ".join(chunk.node_ids),
                ),
            )
        return True

    def neighbors(
        self,
        canonical_ids: tuple[str, ...],
        *,
        depth: int = 1,
        active_only: bool = True,
    ) -> tuple[str, ...]:
        visited = set(canonical_ids)
        frontier = set(canonical_ids)
        with self.connection() as connection:
            for _ in range(max(0, depth)):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                status_clause = "AND review_status = 'active'" if active_only else ""
                rows = connection.execute(
                    f"""
                    SELECT source_id, target_id FROM edges
                    WHERE (source_id IN ({placeholders})
                       OR target_id IN ({placeholders}))
                    {status_clause}
                    """,
                    tuple(frontier) + tuple(frontier),
                ).fetchall()
                expanded = {
                    endpoint
                    for row in rows
                    for endpoint in (row["source_id"], row["target_id"])
                }
                frontier = expanded - visited
                visited.update(expanded)
        return tuple(sorted(visited))

    def get_nodes(self, canonical_ids: tuple[str, ...]) -> tuple[GraphNode, ...]:
        if not canonical_ids:
            return ()
        placeholders = ",".join("?" for _ in canonical_ids)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM nodes
                WHERE canonical_id IN ({placeholders})
                ORDER BY canonical_id
                """,
                canonical_ids,
            ).fetchall()
        return tuple(
            GraphNode(
                canonical_id=row["canonical_id"],
                node_type=row["node_type"],
                name=row["name"],
                summary=row["summary"],
                scope=row["scope"],
                version=row["version"],
                review_status=row["review_status"],
                confidence=row["confidence"],
                source_ids=tuple(json.loads(row["source_ids_json"])),
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        )

    def get_edges(
        self,
        *,
        source_ids: tuple[str, ...] = (),
        target_ids: tuple[str, ...] = (),
        relation: str | None = None,
        active_only: bool = True,
    ) -> tuple[GraphEdge, ...]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            clauses.append(f"source_id IN ({placeholders})")
            parameters.extend(source_ids)
        if target_ids:
            placeholders = ",".join("?" for _ in target_ids)
            clauses.append(f"target_id IN ({placeholders})")
            parameters.extend(target_ids)
        if relation is not None:
            clauses.append("relation = ?")
            parameters.append(relation)
        if active_only:
            clauses.append("review_status = 'active'")
        where = " AND ".join(clauses) if clauses else "1 = 1"
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM edges WHERE {where} ORDER BY edge_id",
                tuple(parameters),
            ).fetchall()
        return tuple(
            GraphEdge(
                edge_id=row["edge_id"],
                source_id=row["source_id"],
                relation=row["relation"],
                target_id=row["target_id"],
                scope=row["scope"],
                version=row["version"],
                review_status=row["review_status"],
                confidence=row["confidence"],
                source_ids=tuple(json.loads(row["source_ids_json"])),
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        )

    def search_evidence(
        self,
        query: str,
        *,
        layers: tuple[int, ...] = (0, 1, 2, 3, 4),
        scope: str | None = None,
        node_ids: tuple[str, ...] = (),
        limit: int = 12,
    ) -> tuple[tuple[EvidenceChunk, float], ...]:
        terms = [
            item
            for item in {
                token.lower()
                for token in __import__("re").findall(r"[\w\u4e00-\u9fff-]+", query)
                if len(token) > 1
            }
        ]
        if not terms:
            return ()
        match_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        layer_placeholders = ",".join("?" for _ in layers)
        clauses = [f"c.layer IN ({layer_placeholders})"]
        parameters: list[Any] = [match_query, *layers]
        if scope is not None:
            clauses.append("c.scope IN (?, 'global')")
            parameters.append(scope)
        if node_ids:
            clauses.append(
                "(" + " OR ".join("f.node_ids LIKE ?" for _ in node_ids) + ")"
            )
            parameters.extend(f"%{item}%" for item in node_ids)
        parameters.append(limit)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*, bm25(evidence_fts) AS rank
                FROM evidence_fts AS f
                JOIN evidence_chunks AS c ON c.chunk_id = f.chunk_id
                WHERE evidence_fts MATCH ?
                  AND {' AND '.join(clauses)}
                ORDER BY rank, c.layer, c.chunk_id
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        return tuple(
            (
                EvidenceChunk(
                    chunk_id=row["chunk_id"],
                    layer=row["layer"],
                    title=row["title"],
                    content=row["content"],
                    source_kind=row["source_kind"],
                    source_uri=row["source_uri"],
                    scope=row["scope"],
                    review_status=row["review_status"],
                    node_ids=tuple(json.loads(row["node_ids_json"])),
                    metadata=json.loads(row["metadata_json"]),
                ),
                float(row["rank"]),
            )
            for row in rows
        )

    def record_intervention(self, record: InterventionRecord) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO experiments(
                    experiment_id, problem_family, instance_hash, incumbent_hash,
                    state_hash, operator_id, factor_id, mechanism_id,
                    objective_name, mechanism_delta, objective_gain, valid,
                    validation_fidelity, runtime_seconds, token_cost,
                    failure_reason, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.experiment_id,
                    record.problem_family,
                    record.instance_hash,
                    record.incumbent_hash,
                    record.state_hash,
                    record.operator_id,
                    record.factor_id,
                    record.mechanism_id,
                    record.objective_name,
                    record.mechanism_delta,
                    record.objective_gain,
                    int(record.valid),
                    record.validation_fidelity,
                    record.runtime_seconds,
                    record.token_cost,
                    record.failure_reason,
                    _json(record.metadata),
                    _now(),
                ),
            )
        return cursor.rowcount == 1

    def intervention_records(
        self,
        *,
        problem_family: str | None = None,
        instance_hash: str | None = None,
        mechanism_id: str | None = None,
        factor_id: str | None = None,
        full_only: bool = False,
    ) -> tuple[InterventionRecord, ...]:
        clauses = []
        parameters: list[Any] = []
        for column, value in (
            ("problem_family", problem_family),
            ("instance_hash", instance_hash),
            ("mechanism_id", mechanism_id),
            ("factor_id", factor_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if full_only:
            clauses.append("validation_fidelity = 'full'")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM experiments {where} ORDER BY created_at, experiment_id",
                tuple(parameters),
            ).fetchall()
        return tuple(
            InterventionRecord(
                experiment_id=row["experiment_id"],
                problem_family=row["problem_family"],
                instance_hash=row["instance_hash"],
                incumbent_hash=row["incumbent_hash"],
                state_hash=row["state_hash"],
                operator_id=row["operator_id"],
                factor_id=row["factor_id"],
                mechanism_id=row["mechanism_id"],
                objective_name=row["objective_name"],
                mechanism_delta=row["mechanism_delta"],
                objective_gain=row["objective_gain"],
                valid=bool(row["valid"]),
                validation_fidelity=row["validation_fidelity"],
                runtime_seconds=row["runtime_seconds"],
                token_cost=row["token_cost"],
                failure_reason=row["failure_reason"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        )

    def stats(self) -> MemoryStats:
        with self.connection() as connection:
            def count(table: str, where: str = "") -> int:
                return int(
                    connection.execute(
                        f"SELECT COUNT(*) AS n FROM {table} {where}"
                    ).fetchone()["n"]
                )

            return MemoryStats(
                nodes=count("nodes"),
                edges=count("edges"),
                aliases=count("aliases"),
                evidence_chunks=count("evidence_chunks"),
                experiments=count("experiments"),
                active_nodes=count("nodes", "WHERE review_status = 'active'"),
                active_edges=count("edges", "WHERE review_status = 'active'"),
            )
