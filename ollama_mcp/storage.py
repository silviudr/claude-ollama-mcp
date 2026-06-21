"""SQLite storage for tool call telemetry."""

import json
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

_CREATE_GRADES = """\
CREATE TABLE IF NOT EXISTS grades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    call_id         INTEGER,
    tool            TEXT    NOT NULL,
    grade_type      TEXT    NOT NULL,
    checker         TEXT    NOT NULL,
    score           REAL,
    passed          INTEGER,
    details         TEXT,
    grader_model    TEXT,
    grader_backend  TEXT,
    grader_ms       INTEGER,
    FOREIGN KEY (call_id) REFERENCES calls(id)
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
        _local.conn.execute(_CREATE_GRADES)
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

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "grades" not in tables:
        conn.execute(_CREATE_GRADES)


def log_call(event: dict) -> int:
    """Insert a tool-call record and return its row id."""
    conn = _conn()
    cur = conn.execute(
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
    return cur.lastrowid


def log_grade(grade: dict) -> None:
    """Insert a grading result."""
    conn = _conn()
    conn.execute(
        """\
        INSERT INTO grades (ts, call_id, tool, grade_type, checker,
                            score, passed, details,
                            grader_model, grader_backend, grader_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            grade.get("ts"),
            grade.get("call_id"),
            grade.get("tool"),
            grade.get("grade_type"),
            grade.get("checker"),
            grade.get("score"),
            1 if grade.get("passed") else 0,
            json.dumps(grade["details"]) if grade.get("details") else None,
            grade.get("grader_model"),
            grade.get("grader_backend"),
            grade.get("grader_ms"),
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

    # Grading stats
    grade_rows = conn.execute("""\
        SELECT
            tool,
            grade_type,
            COUNT(*)                          AS total,
            SUM(passed)                       AS passed,
            COALESCE(AVG(score), 0)           AS avg_score,
            COALESCE(CAST(AVG(grader_ms) AS INT), 0) AS avg_grader_ms
        FROM grades
        GROUP BY tool, grade_type
        ORDER BY tool, grade_type
    """).fetchall()

    grading: dict = {}
    for r in grade_rows:
        row_dict = dict(r)
        tool = row_dict.pop("tool")
        gtype = row_dict.pop("grade_type")
        grading.setdefault(tool, {})[gtype] = row_dict

    stats["grading"] = grading

    return stats


def get_grading_report(tool: str = "", limit: int = 50) -> dict:
    """Detailed grading report for observability.

    Returns per-tool breakdown of heuristic and semantic grades,
    top failing checkers, recent failures with details, and trends.
    """
    conn = _conn()
    conn.row_factory = sqlite3.Row

    where = "WHERE g.tool = ?" if tool else ""
    params: tuple = (tool,) if tool else ()

    # Overall summary
    summary = conn.execute(
        f"""\
        SELECT
            COUNT(*)                              AS total_checks,
            SUM(passed)                           AS passed,
            COUNT(*) - SUM(passed)                AS failed,
            COALESCE(AVG(score), 0)               AS avg_score,
            COUNT(DISTINCT call_id)               AS calls_graded
        FROM grades g {where}""",
        params,
    ).fetchone()

    # Per-tool, per-type breakdown
    breakdown = conn.execute(
        f"""\
        SELECT
            g.tool,
            g.grade_type,
            COUNT(*)                              AS total,
            SUM(g.passed)                         AS passed,
            COUNT(*) - SUM(g.passed)              AS failed,
            COALESCE(AVG(g.score), 0)             AS avg_score,
            COALESCE(CAST(AVG(g.grader_ms) AS INT), 0) AS avg_grader_ms
        FROM grades g {where}
        GROUP BY g.tool, g.grade_type
        ORDER BY g.tool, g.grade_type""",
        params,
    ).fetchall()

    # Top failing checkers
    top_failures = conn.execute(
        f"""\
        SELECT
            g.checker,
            g.tool,
            COUNT(*) AS fail_count
        FROM grades g
        {where + ' AND' if where else 'WHERE'} g.passed = 0
        GROUP BY g.checker, g.tool
        ORDER BY fail_count DESC
        LIMIT 10""",
        params,
    ).fetchall()

    # Recent failed grades with details
    recent_failures = conn.execute(
        f"""\
        SELECT
            g.ts, g.tool, g.grade_type, g.checker, g.score, g.details,
            g.grader_model, g.call_id
        FROM grades g
        {where + ' AND' if where else 'WHERE'} g.passed = 0
        ORDER BY g.ts DESC
        LIMIT ?""",
        params + (limit,),
    ).fetchall()

    # Semantic score distribution
    score_dist = conn.execute(
        f"""\
        SELECT
            g.tool,
            MIN(g.score)                          AS min_score,
            MAX(g.score)                          AS max_score,
            AVG(g.score)                          AS avg_score,
            COUNT(*)                              AS count
        FROM grades g
        {where + ' AND' if where else 'WHERE'} g.grade_type = 'semantic'
            AND g.score IS NOT NULL
        GROUP BY g.tool
        ORDER BY avg_score ASC""",
        params,
    ).fetchall()

    # Per-model scoreboard — JOIN grades to calls to see which models
    # produce the best output, broken down by tool
    model_scores = conn.execute(
        f"""\
        SELECT
            c.model,
            g.tool,
            g.grade_type,
            COUNT(*)                              AS total,
            SUM(g.passed)                         AS passed,
            COALESCE(AVG(g.score), 0)             AS avg_score,
            MIN(g.score)                          AS min_score,
            MAX(g.score)                          AS max_score
        FROM grades g
        JOIN calls c ON g.call_id = c.id
        {where + ' AND' if where else 'WHERE'} g.score IS NOT NULL
            AND c.model IS NOT NULL
        GROUP BY c.model, g.tool, g.grade_type
        ORDER BY g.tool, avg_score DESC, c.model""",
        params,
    ).fetchall()

    # Per-model overall ranking (aggregated across all tools)
    model_ranking = conn.execute(
        f"""\
        SELECT
            c.model,
            COUNT(*)                              AS total,
            SUM(g.passed)                         AS passed,
            COALESCE(AVG(g.score), 0)             AS avg_score,
            COUNT(DISTINCT g.tool)                AS tools_used,
            COALESCE(CAST(AVG(c.total_ms) AS INT), 0) AS avg_latency_ms
        FROM grades g
        JOIN calls c ON g.call_id = c.id
        {where + ' AND' if where else 'WHERE'} g.score IS NOT NULL
            AND c.model IS NOT NULL
        GROUP BY c.model
        ORDER BY avg_score DESC""",
        params,
    ).fetchall()

    return {
        "summary": dict(summary),
        "breakdown": [dict(r) for r in breakdown],
        "top_failures": [dict(r) for r in top_failures],
        "recent_failures": [
            {**dict(r), "details": json.loads(r["details"]) if r["details"] else None}
            for r in recent_failures
        ],
        "semantic_scores": [dict(r) for r in score_dist],
        "model_scores": [dict(r) for r in model_scores],
        "model_ranking": [dict(r) for r in model_ranking],
    }
