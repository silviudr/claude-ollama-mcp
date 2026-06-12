"""SQLite storage for tool call telemetry."""

import sqlite3
import threading

from .config import DB_PATH

_CREATE = """\
CREATE TABLE IF NOT EXISTS calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    tool        TEXT    NOT NULL,
    ok          INTEGER NOT NULL,
    error       TEXT,
    model       TEXT,
    backend     TEXT,
    cost        REAL,
    input_chars INTEGER,
    output_chars INTEGER,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    total_ms    INTEGER,
    wall_ms     INTEGER,
    eval_ms     INTEGER
)"""

# Per-dollar token rates (May 2025 list prices)
CLOUD_PRICING = {
    "opus": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
    "sonnet": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
}

_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH))
        _local.conn.execute(_CREATE)
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _migrate(_local.conn)
    return _local.conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(calls)").fetchall()}
    if "backend" not in cols:
        conn.execute("ALTER TABLE calls ADD COLUMN backend TEXT")
    if "cost" not in cols:
        conn.execute("ALTER TABLE calls ADD COLUMN cost REAL")


def log_call(event: dict) -> None:
    conn = _conn()
    conn.execute(
        """\
        INSERT INTO calls (ts, tool, ok, error, model, backend, cost,
                           input_chars, output_chars,
                           prompt_tokens, output_tokens,
                           total_ms, wall_ms, eval_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event.get("ts"),
            event.get("tool"),
            1 if event.get("ok") else 0,
            event.get("error"),
            event.get("model"),
            event.get("backend"),
            event.get("cost"),
            event.get("input_chars"),
            event.get("output_chars"),
            event.get("prompt_tokens"),
            event.get("output_tokens"),
            event.get("total_ms"),
            event.get("wall_ms"),
            event.get("eval_ms"),
        ),
    )
    conn.commit()


def get_stats() -> dict:
    conn = _conn()
    conn.row_factory = sqlite3.Row

    row = conn.execute("""\
        SELECT
            COUNT(*)                              AS total_calls,
            SUM(ok)                               AS successful,
            COUNT(*) - SUM(ok)                    AS failed,
            COALESCE(SUM(prompt_tokens), 0)       AS total_prompt_tokens,
            COALESCE(SUM(output_tokens), 0)       AS total_output_tokens,
            COALESCE(CAST(AVG(total_ms) AS INT), 0) AS avg_total_ms,
            COALESCE(MIN(ts), 0)                  AS first_call_ts,
            COALESCE(MAX(ts), 0)                  AS last_call_ts
        FROM calls
    """).fetchone()

    stats = dict(row)

    in_tok = stats["total_prompt_tokens"]
    out_tok = stats["total_output_tokens"]
    stats["estimated_cost_avoided"] = {}
    for tier, rates in CLOUD_PRICING.items():
        stats["estimated_cost_avoided"][tier] = round(
            in_tok * rates["input"] + out_tok * rates["output"], 4
        )

    per_tool = conn.execute("""\
        SELECT
            tool,
            COUNT(*)                              AS calls,
            SUM(ok)                               AS successful,
            COALESCE(SUM(prompt_tokens), 0)       AS prompt_tokens,
            COALESCE(SUM(output_tokens), 0)       AS output_tokens,
            COALESCE(CAST(AVG(total_ms) AS INT), 0) AS avg_ms
        FROM calls
        GROUP BY tool
        ORDER BY calls DESC
    """).fetchall()

    stats["per_tool"] = [dict(r) for r in per_tool]

    # Per-backend breakdown
    per_backend = conn.execute("""\
        SELECT
            COALESCE(backend, 'ollama')            AS backend,
            COUNT(*)                               AS calls,
            SUM(ok)                                AS successful,
            COUNT(*) - SUM(ok)                     AS failed,
            COALESCE(SUM(prompt_tokens), 0)        AS prompt_tokens,
            COALESCE(SUM(output_tokens), 0)        AS output_tokens,
            COALESCE(CAST(AVG(total_ms) AS INT), 0) AS avg_ms,
            COALESCE(SUM(cost), 0)                 AS total_cost
        FROM calls
        GROUP BY COALESCE(backend, 'ollama')
        ORDER BY calls DESC
    """).fetchall()

    backend_stats = {}
    for r in per_backend:
        row_dict = dict(r)
        name = row_dict.pop("backend")
        if name == "ollama":
            in_t = row_dict["prompt_tokens"]
            out_t = row_dict["output_tokens"]
            row_dict["estimated_cost_avoided"] = {}
            for tier, rates in CLOUD_PRICING.items():
                row_dict["estimated_cost_avoided"][tier] = round(
                    in_t * rates["input"] + out_t * rates["output"], 4
                )
        backend_stats[name] = row_dict

    stats["per_backend"] = backend_stats

    return stats
