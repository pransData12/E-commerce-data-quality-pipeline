"""Unit tests: each check type on tiny hand-made frames (known right answers)."""
import pandas as pd
import pytest

from dq_pipeline import validators as v

AS_OF = pd.Timestamp("2025-07-01")


def run(check, tables, rule):
    failed, checked = v.REGISTRY[check](tables, rule, AS_OF)
    return list(failed), checked


def test_not_null_flags_none_and_blank():
    t = {"t": pd.DataFrame({"a": ["x", None, "  ", "y"]})}
    assert run("not_null", t, {"table": "t", "column": "a"})[0] == [1, 2]


def test_unique_flags_only_repeats():
    t = {"t": pd.DataFrame({"id": [1, 2, 2, 3, 2]})}
    assert run("unique", t, {"table": "t", "column": "id"})[0] == [2, 4]


def test_regex_email():
    t = {"t": pd.DataFrame({"e": ["a@b.com", "ab.com", "a@b", None]})}
    failed, checked = run("regex", t, {"table": "t", "column": "e", "pattern": r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}"})
    assert failed == [1, 2] and checked == 3  # null is not "invalid", it is "missing"


def test_range_min_max_and_garbage():
    t = {"t": pd.DataFrame({"q": [1, 0, 101, "abc", 50]})}
    assert run("range", t, {"table": "t", "column": "q", "min": 1, "max": 100})[0] == [1, 2, 3]


def test_allowed_values():
    t = {"t": pd.DataFrame({"s": ["a", "b", "Shipped ", None]})}
    assert run("allowed_values", t, {"table": "t", "column": "s", "values": ["a", "b"]})[0] == [2]


def test_date_not_future():
    t = {"t": pd.DataFrame({"d": ["2025-06-30", "2025-07-02", "garbage"]})}
    assert run("date_not_future", t, {"table": "t", "column": "d"})[0] == [1, 2]


def test_foreign_key():
    t = {"c": pd.DataFrame({"id": [1, 2]}), "o": pd.DataFrame({"cid": [1, 3, 2]})}
    assert run("foreign_key", t, {"table": "o", "column": "cid", "ref_table": "c", "ref_column": "id"})[0] == [1]


def test_date_order_ignores_orphans():
    t = {"c": pd.DataFrame({"id": [1], "signup": ["2024-01-01"]}),
         "o": pd.DataFrame({"cid": [1, 1, 99], "d": ["2024-02-01", "2023-12-31", "2020-01-01"]})}
    rule = {"table": "o", "column": "d", "key": "cid", "ref_table": "c", "ref_key": "id", "ref_column": "signup"}
    assert run("date_order", t, rule)[0] == [1]


def test_price_consistency_is_relative_to_catalog_not_global():
    # 50,000 is fine for the laptop, wrong for the book. A global IQR cannot tell.
    t = {"p": pd.DataFrame({"pid": [1, 2], "price": [50000.0, 500.0]}),
         "i": pd.DataFrame({"pid": [1, 2, 2], "unit": [49000.0, 50000.0, 510.0]})}
    rule = {"table": "i", "column": "unit", "product_column": "pid", "ref_table": "p", "ref_key": "pid",
            "ref_price": "price", "tolerance": 0.15}
    assert run("price_consistency", t, rule)[0] == [1]


def test_sum_match():
    t = {"o": pd.DataFrame({"oid": [1, 2], "total": [30.0, 99.0]}),
         "i": pd.DataFrame({"oid": [1, 1, 2], "q": [1, 2, 1], "p": [10.0, 10.0, 20.0]})}
    rule = {"table": "o", "key": "oid", "column": "total", "child_table": "i", "child_key": "oid",
            "child_qty": "q", "child_price": "p", "tolerance": 0.01}
    assert run("sum_match", t, rule)[0] == [1]


def test_config_rejects_unknown_type(tmp_path):
    bad = tmp_path / "r.yaml"
    bad.write_text("rules:\n  - {id: x, type: nope, table: t, severity: error, description: d}\n")
    with pytest.raises(v.ConfigError, match="unknown type"):
        v.load_config(bad)


def test_config_rejects_duplicate_ids(tmp_path):
    bad = tmp_path / "r.yaml"
    row = "{id: x, type: unique, table: t, column: c, severity: error, description: d}"
    bad.write_text(f"rules:\n  - {row}\n  - {row}\n")
    with pytest.raises(v.ConfigError, match="duplicate"):
        v.load_config(bad)
