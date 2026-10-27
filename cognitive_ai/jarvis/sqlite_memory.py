"""SQLite + FTS5 memory backend (blueprint §7)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import utc_now
from .contracts import FactRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    value TEXT NOT NULL,
    units TEXT,
    source_id TEXT,
    source_version TEXT,
    page_or_row TEXT,
    observed_at TEXT,
    recorded_at TEXT,
    valid_from TEXT,
    valid_until TEXT,
    verification_status TEXT,
    access_scope TEXT,
    supersedes_fact_id TEXT,
    kind TEXT
);
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    summary TEXT,
    evidence_json TEXT,
    recorded_at TEXT
);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    path TEXT,
    version TEXT,
    excerpt TEXT,
    retrieved_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    source_id,
    title,
    body,
    tokenize = 'porter'
);
CREATE TABLE IF NOT EXISTS working (
    key TEXT PRIMARY KEY,
    value_json TEXT
);
"""


class SqliteMemory:
    """Structured facts + FTS5 document retrieval with BM25 ranking."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def set_working(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO working(key, value_json) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_working(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value_json FROM working WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        return json.loads(row["value_json"])

    def working_snapshot(self) -> dict[str, Any]:
        rows = self.conn.execute("SELECT key, value_json FROM working").fetchall()
        return {r["key"]: json.loads(r["value_json"]) for r in rows}

    def record_fact(self, fact: FactRecord) -> FactRecord:
        # supersede prior current fact with same subject+relation
        if fact.supersedes_fact_id is None:
            prior = self.conn.execute(
                "SELECT fact_id FROM facts WHERE subject=? AND relation=? AND valid_until IS NULL AND kind=?",
                (fact.subject, fact.relation, fact.kind),
            ).fetchone()
            if prior:
                fact.supersedes_fact_id = prior["fact_id"]
                self.conn.execute(
                    "UPDATE facts SET valid_until=? WHERE fact_id=?",
                    (fact.recorded_at, prior["fact_id"]),
                )
        elif fact.supersedes_fact_id:
            self.conn.execute(
                "UPDATE facts SET valid_until=? WHERE fact_id=?",
                (fact.recorded_at, fact.supersedes_fact_id),
            )
        self.conn.execute(
            """INSERT INTO facts(
                fact_id, subject, relation, value, units, source_id, source_version,
                page_or_row, observed_at, recorded_at, valid_from, valid_until,
                verification_status, access_scope, supersedes_fact_id, kind
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fact.fact_id,
                fact.subject,
                fact.relation,
                json.dumps(fact.value),
                fact.units,
                fact.source_id,
                fact.source_version,
                fact.page_or_row,
                fact.observed_at,
                fact.recorded_at,
                fact.valid_from,
                fact.valid_until,
                fact.verification_status,
                fact.access_scope,
                fact.supersedes_fact_id,
                fact.kind,
            ),
        )
        self.conn.commit()
        return fact

    def query_facts(
        self,
        *,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        kind: Optional[str] = None,
        current_only: bool = True,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM facts WHERE 1=1"
        args: list[Any] = []
        if subject:
            sql += " AND subject=?"
            args.append(subject)
        if relation:
            sql += " AND relation=?"
            args.append(relation)
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        if current_only:
            sql += " AND valid_until IS NULL"
        rows = self.conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["value"] = json.loads(d["value"])
            except Exception:
                pass
            out.append(d)
        return out

    def record_episode(self, summary: str, evidence: Optional[dict[str, Any]] = None) -> str:
        eid = new_id("ep")
        self.conn.execute(
            "INSERT INTO episodes(episode_id, summary, evidence_json, recorded_at) VALUES(?,?,?,?)",
            (eid, summary, json.dumps(evidence or {}), utc_now()),
        )
        self.conn.commit()
        return eid

    def index_document(self, source_id: str, title: str, body: str, *, path: str = "", version: str = "1") -> None:
        self.conn.execute(
            "INSERT INTO sources(source_id, path, version, excerpt, retrieved_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(source_id) DO UPDATE SET path=excluded.path, version=excluded.version, "
            "excerpt=excluded.excerpt, retrieved_at=excluded.retrieved_at",
            (source_id, path, version, body[:500], utc_now()),
        )
        # Replace FTS row
        self.conn.execute("DELETE FROM docs_fts WHERE source_id=?", (source_id,))
        self.conn.execute(
            "INSERT INTO docs_fts(source_id, title, body) VALUES(?,?,?)",
            (source_id, title, body),
        )
        self.conn.commit()

    def search_documents(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        try:
            rows = self.conn.execute(
                "SELECT source_id, title, snippet(docs_fts, 2, '[', ']', '…', 12) AS snippet, "
                "bm25(docs_fts) AS rank "
                "FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS query syntax error — fall back to LIKE
            like = f"%{query}%"
            rows = self.conn.execute(
                "SELECT source_id, title, substr(body,1,120) AS snippet, 0 AS rank "
                "FROM docs_fts WHERE body LIKE ? OR title LIKE ? LIMIT ?",
                (like, like, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        return {
            "facts": self.conn.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"],
            "episodes": self.conn.execute("SELECT COUNT(*) c FROM episodes").fetchone()["c"],
            "sources": self.conn.execute("SELECT COUNT(*) c FROM sources").fetchone()["c"],
            "path": self.path,
            "engine": "sqlite+fts5",
        }
