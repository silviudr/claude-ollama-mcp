import csv
import json

import pytest

from ollama_mcp.analyzer import (
    ColumnProfile,
    DatasetMeta,
    NumericStats,
    _compute_numeric_stats,
    _detect_delimiter,
    _detect_encoding,
    _is_free_text,
    _looks_like_datetime,
    _looks_like_foreign_key,
    _looks_like_json,
    _try_float,
    read_csv_meta,
)


@pytest.fixture
def simple_csv(tmp_path):
    p = tmp_path / "simple.csv"
    p.write_text("name,age,city\nAlice,30,London\nBob,25,Paris\nCarol,35,Berlin\n")
    return str(p)


@pytest.fixture
def complex_csv(tmp_path):
    p = tmp_path / "complex.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        headers = [f"col_{i}" for i in range(25)]
        writer.writerow(headers)
        for i in range(100):
            writer.writerow([f"val_{i}_{j}" for j in range(25)])
    return str(p)


@pytest.fixture
def json_csv(tmp_path):
    p = tmp_path / "nested.csv"
    p.write_text(
        'id,name,metadata\n'
        '1,Alice,"{""role"": ""admin"", ""dept"": ""eng""}"\n'
        '2,Bob,"{""role"": ""user"", ""dept"": ""sales""}"\n'
    )
    return str(p)


@pytest.fixture
def datetime_csv(tmp_path):
    p = tmp_path / "times.csv"
    p.write_text(
        "id,created_at,updated_at,name\n"
        "1,2024-01-15T10:30:00,2024-02-20T14:00:00,Alice\n"
        "2,2024-03-10T08:00:00,2024-04-05T12:30:00,Bob\n"
    )
    return str(p)


# --- Detection helpers ---


def test_looks_like_datetime():
    assert _looks_like_datetime("2024-01-15")
    assert _looks_like_datetime("2024-01-15T10:30:00")
    assert _looks_like_datetime("2024/01/15 10:30")
    assert not _looks_like_datetime("hello")
    assert not _looks_like_datetime("12345")


def test_looks_like_json():
    assert _looks_like_json('{"key": "value"}')
    assert _looks_like_json('[1, 2, 3]')
    assert not _looks_like_json("hello")
    assert not _looks_like_json("123")
    assert not _looks_like_json("{not json")


def test_try_float():
    assert _try_float("3.14") == 3.14
    assert _try_float("42") == 42.0
    assert _try_float("-7") == -7.0
    assert _try_float("hello") is None
    assert _try_float("") is None


# --- read_csv_meta ---


def test_read_simple_csv(simple_csv):
    meta = read_csv_meta(simple_csv)
    assert meta.row_count == 3
    assert meta.col_count == 3
    assert meta.headers == ["name", "age", "city"]
    assert len(meta.sample_rows) == 3
    assert len(meta.columns) == 3


def test_simple_csv_low_complexity(simple_csv):
    meta = read_csv_meta(simple_csv)
    assert meta.complexity_score < 0.7
    assert meta.handoff is False


def test_complex_csv_high_column_count(complex_csv):
    meta = read_csv_meta(complex_csv)
    assert meta.col_count == 25
    assert any("column count" in r.lower() for r in meta.handoff_reasons)


def test_json_columns_detected(json_csv):
    meta = read_csv_meta(json_csv)
    json_cols = [c for c in meta.columns if c.has_nested_json]
    assert len(json_cols) >= 1
    assert json_cols[0].name == "metadata"


def test_datetime_columns_detected(datetime_csv):
    meta = read_csv_meta(datetime_csv)
    dt_cols = [c for c in meta.columns if c.is_datetime]
    assert len(dt_cols) == 2
    dt_names = {c.name for c in dt_cols}
    assert dt_names == {"created_at", "updated_at"}


def test_multiple_datetimes_increase_complexity(datetime_csv):
    meta = read_csv_meta(datetime_csv)
    assert any("datetime" in r.lower() for r in meta.handoff_reasons)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_csv_meta("/nonexistent/path.csv")


def test_non_csv_rejected(tmp_path):
    p = tmp_path / "data.xlsx"
    p.write_text("fake")
    with pytest.raises(ValueError, match="CSV"):
        read_csv_meta(str(p))


def test_sample_rows_capped(tmp_path):
    p = tmp_path / "big.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "value"])
        for i in range(200):
            writer.writerow([i, f"val_{i}"])

    meta = read_csv_meta(str(p))
    assert meta.row_count == 200
    assert len(meta.sample_rows) == 50


def test_cardinality_ratio():
    col = ColumnProfile(name="id", total=100, unique=100)
    assert col.cardinality_ratio == 1.0

    col2 = ColumnProfile(name="status", total=100, unique=3)
    assert col2.cardinality_ratio == 0.03


def test_format_summary(simple_csv):
    meta = read_csv_meta(simple_csv)
    summary = meta.format_summary()
    assert "simple.csv" in summary
    assert "Rows: 3" in summary
    assert "Columns: 3" in summary
    assert "name" in summary


def test_sample_as_text(simple_csv):
    meta = read_csv_meta(simple_csv)
    text = meta.sample_as_text()
    assert "name,age,city" in text
    assert "Alice" in text


def test_high_cardinality_flagged(tmp_path):
    p = tmp_path / "unique.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "uuid", "hash"])
        for i in range(100):
            writer.writerow([i, f"uuid-{i}", f"hash-{i}"])

    meta = read_csv_meta(str(p))
    assert any("cardinality" in r.lower() for r in meta.handoff_reasons)


# --- Foreign key detection ---


def test_looks_like_foreign_key():
    assert _looks_like_foreign_key("user_id")
    assert _looks_like_foreign_key("order_uuid")
    assert _looks_like_foreign_key("parent_ref")
    assert _looks_like_foreign_key("category_fk")
    assert _looks_like_foreign_key("session_key")
    assert not _looks_like_foreign_key("name")
    assert not _looks_like_foreign_key("identifier")


def test_foreign_keys_detected(tmp_path):
    p = tmp_path / "relational.csv"
    p.write_text(
        "id,user_id,order_id,product_id,amount\n"
        "1,usr_1,ord_1,prod_1,99.99\n"
        "2,usr_2,ord_2,prod_2,49.50\n"
    )
    meta = read_csv_meta(str(p))
    fk_cols = [c for c in meta.columns if c.is_foreign_key]
    assert len(fk_cols) == 3
    assert {c.name for c in fk_cols} == {"user_id", "order_id", "product_id"}


def test_multiple_foreign_keys_flagged(tmp_path):
    p = tmp_path / "relational.csv"
    p.write_text(
        "id,user_id,order_id,amount\n"
        "1,usr_1,ord_1,99.99\n"
    )
    meta = read_csv_meta(str(p))
    assert any("foreign key" in r.lower() for r in meta.handoff_reasons)


# --- Free-text detection ---


def test_is_free_text():
    short = ["yes", "no", "maybe", "yes", "no"]
    assert not _is_free_text(short)

    long = [
        "The customer reported an issue with the billing system that caused "
        "double charges on their credit card for the monthly subscription.",
        "User feedback indicates the onboarding flow is confusing, especially "
        "the step where they need to verify their email address.",
        "Bug report: the search feature returns no results when using special "
        "characters like ampersands or quotation marks in the query string.",
    ]
    assert _is_free_text(long)


def test_free_text_detected(tmp_path):
    p = tmp_path / "feedback.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "feedback"])
        for i in range(10):
            writer.writerow([
                i,
                f"This is a detailed customer feedback entry number {i} that "
                f"contains multiple sentences about the product experience and "
                f"various suggestions for improvement in the user interface.",
            ])
    meta = read_csv_meta(str(p))
    text_cols = [c for c in meta.columns if c.is_free_text]
    assert len(text_cols) == 1
    assert text_cols[0].name == "feedback"
    assert any("free-text" in r.lower() for r in meta.handoff_reasons)


def test_free_text_shown_in_summary(tmp_path):
    p = tmp_path / "feedback.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "comment"])
        for i in range(5):
            writer.writerow([
                i,
                f"A long detailed comment about issue {i} that spans many "
                f"words and provides extensive context about the problem.",
            ])
    meta = read_csv_meta(str(p))
    summary = meta.format_summary()
    assert "free-text" in summary


# --- Column flags in summary ---


def test_fk_shown_in_summary(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("id,user_id,name\n1,usr_1,Alice\n")
    meta = read_csv_meta(str(p))
    summary = meta.format_summary()
    assert "FK" in summary


# --- Numeric detection and stats ---


def test_numeric_column_detected(simple_csv):
    meta = read_csv_meta(simple_csv)
    age_col = next(c for c in meta.columns if c.name == "age")
    assert age_col.is_numeric is True
    assert age_col.is_mixed_type is False


def test_non_numeric_column(simple_csv):
    meta = read_csv_meta(simple_csv)
    name_col = next(c for c in meta.columns if c.name == "name")
    assert name_col.is_numeric is False


def test_numeric_stats_computed(simple_csv):
    meta = read_csv_meta(simple_csv)
    age_col = next(c for c in meta.columns if c.name == "age")
    assert age_col.numeric_stats is not None
    assert age_col.numeric_stats.min == 25
    assert age_col.numeric_stats.max == 35
    assert age_col.numeric_stats.mean == 30.0


def test_numeric_stats_in_summary(simple_csv):
    meta = read_csv_meta(simple_csv)
    summary = meta.format_summary()
    assert "numeric" in summary
    assert "min=" in summary


def test_compute_numeric_stats():
    stats = _compute_numeric_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats.min == 1.0
    assert stats.max == 5.0
    assert stats.mean == 3.0
    assert stats.median == 3.0


def test_compute_numeric_stats_even_count():
    stats = _compute_numeric_stats([10.0, 20.0, 30.0, 40.0])
    assert stats.median == 25.0


def test_numeric_stats_format():
    stats = NumericStats(min=1.0, max=100.0, mean=50.5, median=48.0)
    text = stats.format()
    assert "min=1" in text
    assert "max=100" in text
    assert "mean=50.50" in text
    assert "median=48" in text


# --- Mixed-type detection ---


def test_mixed_type_detected(tmp_path):
    p = tmp_path / "mixed.csv"
    p.write_text("id,value\n1,100\n2,hello\n3,300\n4,world\n")
    meta = read_csv_meta(str(p))
    val_col = next(c for c in meta.columns if c.name == "value")
    assert val_col.is_mixed_type is True
    assert val_col.is_numeric is False


def test_mixed_type_flagged_in_reasons(tmp_path):
    p = tmp_path / "mixed.csv"
    p.write_text("id,score,grade\n1,95,A\n2,bad,B\n3,78,C\n")
    meta = read_csv_meta(str(p))
    assert any("mixed-type" in r.lower() for r in meta.handoff_reasons)


def test_mixed_type_shown_in_summary(tmp_path):
    p = tmp_path / "mixed.csv"
    p.write_text("id,value\n1,100\n2,hello\n")
    meta = read_csv_meta(str(p))
    summary = meta.format_summary()
    assert "mixed-type" in summary


# --- Encoding detection ---


def test_detect_encoding_utf8(tmp_path):
    p = tmp_path / "utf8.csv"
    p.write_bytes("name,city\nAlice,München\n".encode("utf-8"))
    assert _detect_encoding(p) in ("utf-8-sig", "utf-8")


def test_detect_encoding_latin1(tmp_path):
    p = tmp_path / "latin.csv"
    p.write_bytes("name,city\nAlice,M\xfcnchen\n".encode("latin-1"))
    enc = _detect_encoding(p)
    assert enc == "latin-1"


def test_read_latin1_csv(tmp_path):
    p = tmp_path / "latin.csv"
    p.write_bytes("name,city\nAlice,M\xfcnchen\nBob,Z\xfcrich\n".encode("latin-1"))
    meta = read_csv_meta(str(p))
    assert meta.row_count == 2
    assert meta.encoding == "latin-1"


# --- Delimiter detection ---


def test_detect_semicolon_delimiter(tmp_path):
    p = tmp_path / "semi.csv"
    p.write_text("name;age;city\nAlice;30;London\nBob;25;Paris\n")
    delim = _detect_delimiter(p, "utf-8")
    assert delim == ";"


def test_detect_tab_delimiter(tmp_path):
    p = tmp_path / "tabs.csv"
    p.write_text("name\tage\tcity\nAlice\t30\tLondon\nBob\t25\tParis\n")
    delim = _detect_delimiter(p, "utf-8")
    assert delim == "\t"


def test_read_semicolon_csv(tmp_path):
    p = tmp_path / "semi.csv"
    p.write_text("name;age;city\nAlice;30;London\nBob;25;Paris\n")
    meta = read_csv_meta(str(p))
    assert meta.row_count == 2
    assert meta.delimiter == ";"
    assert meta.headers == ["name", "age", "city"]


def test_read_tab_csv(tmp_path):
    p = tmp_path / "tabs.csv"
    p.write_text("name\tage\tcity\nAlice\t30\tLondon\n")
    meta = read_csv_meta(str(p))
    assert meta.row_count == 1
    assert meta.delimiter == "\t"


def test_delimiter_in_summary(tmp_path):
    p = tmp_path / "semi.csv"
    p.write_text("name;age\nAlice;30\n")
    meta = read_csv_meta(str(p))
    summary = meta.format_summary()
    assert "';'" in summary


# --- Empty file handling ---


def test_empty_csv_headers_only(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("name,age,city\n")
    meta = read_csv_meta(str(p))
    assert meta.row_count == 0
    assert meta.col_count == 3
    assert meta.headers == ["name", "age", "city"]
    assert len(meta.sample_rows) == 0
    assert meta.complexity_score == 0.0
    assert meta.handoff is False


def test_empty_csv_format_summary(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("name,age\n")
    meta = read_csv_meta(str(p))
    summary = meta.format_summary()
    assert "Rows: 0" in summary


def test_empty_csv_sample_as_text(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("name,age\n")
    meta = read_csv_meta(str(p))
    assert meta.sample_as_text() == "No data."


# --- Memory safety: value collection capped ---


def test_value_collection_capped(tmp_path):
    """Ensure we don't store more than ANALYSIS_SAMPLE_ROWS values per column
    while still counting all rows and tracking unique values."""
    p = tmp_path / "big.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "status"])
        for i in range(500):
            writer.writerow([i, "active" if i % 2 == 0 else "inactive"])

    meta = read_csv_meta(str(p))
    assert meta.row_count == 500
    assert len(meta.sample_rows) == 50
    status_col = next(c for c in meta.columns if c.name == "status")
    assert status_col.unique == 2
    assert status_col.total == 500
