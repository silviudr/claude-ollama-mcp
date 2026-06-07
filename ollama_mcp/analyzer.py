"""CSV data analyzer with triage logic for local/cloud handoff."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import ANALYSIS_MAX_COLS, ANALYSIS_SAMPLE_ROWS, ANALYSIS_THRESHOLD

_FK_SUFFIXES = ("_id", "_uuid", "_key", "_ref", "_fk")
_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252")


@dataclass
class NumericStats:
    min: float
    max: float
    mean: float
    median: float

    def format(self) -> str:
        return f"min={self.min:g}, max={self.max:g}, mean={self.mean:.2f}, median={self.median:g}"


@dataclass
class ColumnProfile:
    name: str
    total: int = 0
    unique: int = 0
    nulls: int = 0
    has_nested_json: bool = False
    is_datetime: bool = False
    is_free_text: bool = False
    is_foreign_key: bool = False
    is_numeric: bool = False
    is_mixed_type: bool = False
    avg_value_length: float = 0.0
    numeric_stats: NumericStats | None = None

    @property
    def cardinality_ratio(self) -> float:
        return self.unique / self.total if self.total else 0.0


@dataclass
class DatasetMeta:
    file_path: str
    row_count: int
    col_count: int
    headers: list[str]
    sample_rows: list[dict]
    columns: list[ColumnProfile]
    delimiter: str = ","
    encoding: str = "utf-8"
    complexity_score: float = 0.0
    handoff: bool = False
    handoff_reasons: list[str] = field(default_factory=list)

    def format_summary(self) -> str:
        lines = [
            f"File: {self.file_path}",
            f"Rows: {self.row_count} (sampled {len(self.sample_rows)})",
            f"Columns: {self.col_count}",
            f"Delimiter: {repr(self.delimiter)}, Encoding: {self.encoding}",
            f"Complexity: {self.complexity_score:.2f} "
            f"(threshold: {ANALYSIS_THRESHOLD})",
            f"Handoff to Claude: {'yes' if self.handoff else 'no'}",
        ]
        if self.handoff_reasons:
            lines.append("Reasons:")
            for r in self.handoff_reasons:
                lines.append(f"  - {r}")

        lines.append("")
        lines.append("Column profiles:")
        for c in self.columns:
            flags = []
            if c.is_numeric:
                flags.append("numeric")
            if c.is_mixed_type:
                flags.append("mixed-type")
            if c.has_nested_json:
                flags.append("JSON")
            if c.is_datetime:
                flags.append("datetime")
            if c.is_free_text:
                flags.append("free-text")
            if c.is_foreign_key:
                flags.append("FK")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            stats_str = ""
            if c.numeric_stats:
                stats_str = f" ({c.numeric_stats.format()})"
            lines.append(
                f"  {c.name}: {c.unique}/{c.total} unique "
                f"({c.cardinality_ratio:.0%}), "
                f"{c.nulls} nulls{flag_str}{stats_str}"
            )
        return "\n".join(lines)

    def sample_as_text(self) -> str:
        if not self.sample_rows:
            return "No data."
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=self.headers)
        writer.writeheader()
        for row in self.sample_rows[:10]:
            writer.writerow(row)
        return out.getvalue()


_DATETIME_RE = re.compile(
    r"^\d{4}[-/]\d{2}[-/]\d{2}"
    r"([T ]\d{2}:\d{2}(:\d{2})?)?"
)

_JSON_RE = re.compile(r"^\s*[\[{]")

_FREE_TEXT_AVG_LEN = 60
_FREE_TEXT_AVG_WORDS = 8


def _looks_like_datetime(value: str) -> bool:
    return bool(_DATETIME_RE.match(value.strip()))


def _looks_like_json(value: str) -> bool:
    if not _JSON_RE.match(value):
        return False
    try:
        json.loads(value)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _looks_like_foreign_key(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(s) for s in _FK_SUFFIXES)


def _is_free_text(values: list[str]) -> bool:
    if not values:
        return False
    avg_len = sum(len(v) for v in values) / len(values)
    avg_words = sum(len(v.split()) for v in values) / len(values)
    return avg_len > _FREE_TEXT_AVG_LEN and avg_words > _FREE_TEXT_AVG_WORDS


def _try_float(value: str) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _compute_numeric_stats(values: list[float]) -> NumericStats:
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    median = (sorted_v[mid] + sorted_v[mid - 1]) / 2 if n % 2 == 0 else sorted_v[mid]
    return NumericStats(
        min=sorted_v[0],
        max=sorted_v[-1],
        mean=sum(sorted_v) / n,
        median=median,
    )


def _detect_encoding(path: Path) -> str:
    sample = path.read_bytes()[:8192]
    for enc in _ENCODINGS:
        try:
            sample.decode(enc)
            return enc
        except (UnicodeDecodeError, ValueError):
            continue
    return "utf-8"


def _detect_delimiter(path: Path, encoding: str) -> str:
    with open(path, newline="", encoding=encoding) as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def read_csv_meta(file_path: str) -> DatasetMeta:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV files are supported, got: {path.suffix}")

    encoding = _detect_encoding(path)
    delimiter = _detect_delimiter(path, encoding)

    sample_rows: list[dict] = []
    sampled_values: dict[str, list[str]] = {}
    unique_sets: dict[str, set[str]] = {}
    non_null_counts: dict[str, int] = {}
    row_count = 0

    with open(path, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = reader.fieldnames or []
        for h in headers:
            sampled_values[h] = []
            unique_sets[h] = set()
            non_null_counts[h] = 0

        for row in reader:
            row_count += 1
            if len(sample_rows) < ANALYSIS_SAMPLE_ROWS:
                sample_rows.append(dict(row))
            for h in headers:
                val = row.get(h, "")
                if val:
                    non_null_counts[h] += 1
                    unique_sets[h].add(val)
                    if len(sampled_values[h]) < ANALYSIS_SAMPLE_ROWS:
                        sampled_values[h].append(val)

    if row_count == 0:
        return DatasetMeta(
            file_path=file_path,
            row_count=0,
            col_count=len(headers),
            headers=headers,
            sample_rows=[],
            columns=[
                ColumnProfile(name=h, total=0) for h in headers
            ],
            delimiter=delimiter,
            encoding=encoding,
        )

    columns: list[ColumnProfile] = []
    for h in headers:
        sample_vals = sampled_values[h]
        avg_len = (
            sum(len(v) for v in sample_vals) / len(sample_vals)
            if sample_vals
            else 0.0
        )

        numeric_vals = [_try_float(v) for v in sample_vals]
        numeric_count = sum(1 for v in numeric_vals if v is not None)
        non_numeric_count = len(sample_vals) - numeric_count
        is_numeric = numeric_count > 0 and non_numeric_count == 0
        is_mixed = numeric_count > 0 and non_numeric_count > 0

        stats = None
        if is_numeric and numeric_count >= 2:
            stats = _compute_numeric_stats([v for v in numeric_vals if v is not None])

        columns.append(ColumnProfile(
            name=h,
            total=row_count,
            unique=len(unique_sets[h]),
            nulls=row_count - non_null_counts[h],
            has_nested_json=any(_looks_like_json(v) for v in sample_vals),
            is_datetime=any(_looks_like_datetime(v) for v in sample_vals),
            is_free_text=_is_free_text(sample_vals),
            is_foreign_key=_looks_like_foreign_key(h),
            is_numeric=is_numeric,
            is_mixed_type=is_mixed,
            avg_value_length=avg_len,
            numeric_stats=stats,
        ))

    meta = DatasetMeta(
        file_path=file_path,
        row_count=row_count,
        col_count=len(headers),
        headers=headers,
        sample_rows=sample_rows,
        columns=columns,
        delimiter=delimiter,
        encoding=encoding,
    )

    _score_complexity(meta)
    return meta


def _score_complexity(meta: DatasetMeta) -> None:
    scores: list[float] = []
    reasons: list[str] = []

    # 1. Structural complexity: column count
    col_ratio = min(meta.col_count / ANALYSIS_MAX_COLS, 1.0)
    scores.append(col_ratio)
    if meta.col_count > ANALYSIS_MAX_COLS:
        reasons.append(
            f"High column count: {meta.col_count} (max {ANALYSIS_MAX_COLS})"
        )

    # 2. Data cardinality: avg unique ratio across columns
    if meta.columns:
        avg_cardinality = sum(
            c.cardinality_ratio for c in meta.columns
        ) / len(meta.columns)
        scores.append(avg_cardinality)
        if avg_cardinality > 0.8:
            reasons.append(
                f"High cardinality: {avg_cardinality:.0%} avg unique ratio"
            )

    # 3. Implicit relationships: nested JSON or datetime columns
    json_cols = [c.name for c in meta.columns if c.has_nested_json]
    dt_cols = [c.name for c in meta.columns if c.is_datetime]

    if json_cols:
        scores.append(0.4 * len(json_cols))
        reasons.append(f"Nested JSON in: {', '.join(json_cols)}")

    if len(dt_cols) >= 2:
        scores.append(0.3)
        reasons.append(f"Multiple datetime columns: {', '.join(dt_cols)}")

    # 4. Free-text columns: need NLP reasoning
    text_cols = [c.name for c in meta.columns if c.is_free_text]
    if text_cols:
        scores.append(0.5 * len(text_cols))
        reasons.append(f"Free-text columns: {', '.join(text_cols)}")

    # 5. Foreign keys: suggest relational structure
    fk_cols = [c.name for c in meta.columns if c.is_foreign_key]
    if len(fk_cols) >= 2:
        scores.append(0.3)
        reasons.append(
            f"Multiple foreign keys ({len(fk_cols)}): "
            f"{', '.join(fk_cols)} — suggests relational joins needed"
        )

    # 6. Mixed-type columns: data quality concern
    mixed_cols = [c.name for c in meta.columns if c.is_mixed_type]
    if mixed_cols:
        scores.append(0.3)
        reasons.append(
            f"Mixed-type columns: {', '.join(mixed_cols)} "
            f"— contain both numeric and non-numeric values"
        )

    meta.complexity_score = min(sum(scores) / max(len(scores), 1), 1.0)
    meta.handoff = meta.complexity_score > ANALYSIS_THRESHOLD
    meta.handoff_reasons = reasons
