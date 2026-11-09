"""Integration tests: generate -> validate -> quarantine -> warehouse -> ML, scored against ground truth."""
import sqlite3

import pandas as pd
import pytest

from dq_pipeline import generate_data, run_pipeline, stats_ml, validators


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    return run_pipeline.run(seed=7, data_dir=d / "raw", db_path=d / "wh.db", report_path=d / "r.html")


def test_generator_is_deterministic(tmp_path):
    generate_data.generate(tmp_path / "a", seed=1)
    generate_data.generate(tmp_path / "b", seed=1)
    for name in run_pipeline.TABLE_NAMES:
        assert (tmp_path / "a" / f"{name}.csv").read_text() == (tmp_path / "b" / f"{name}.csv").read_text()


def test_every_injected_defect_is_found_with_no_false_alarms(out):
    ev = out["evaluation"].dropna(subset=["recall_pct"])
    assert (ev["recall_pct"] == 100.0).all()
    assert (ev["precision_pct"] == 100.0).all()


def test_warehouse_is_clean_on_revalidation(out):
    cfg = validators.load_config(run_pipeline.ROOT / "config" / "rules.yaml")
    with sqlite3.connect(out["db_path"]) as con:
        reloaded = {n: pd.read_sql_query(f"SELECT * FROM {n}", con) for n in run_pipeline.TABLE_NAMES}
    results, _ = validators.run_validation(reloaded, cfg)
    errors = results[results["severity"] == "error"]
    assert (errors["rows_failed"] == 0).all()


def test_row_accounting_adds_up(out):
    a = out["accounting"]
    assert (a["raw_rows"] - a["failed_error_rules"] - a["removed_by_cascade"] == a["loaded_rows"]).all()


def test_raw_revenue_is_overstated(out):
    assert out["revenue"]["raw"] > out["revenue"]["clean"]


def test_ml_with_domain_feature_beats_generic_statistics(out):
    m = out["method_comparison"].set_index("method")
    forest = m.loc["Isolation Forest - + price/catalog ratio", "recall_pct"]
    assert forest > 70  # varies with seed; the ordering below is the real claim
    assert forest > m.loc["Global IQR (Tukey, 1.5x)", "recall_pct"]
    assert forest > m.loc["Isolation Forest - raw features", "recall_pct"]


def test_ml_cannot_see_orphan_keys(out):
    r = out["ml_recall"].set_index("defect_type")
    assert r.loc["orphan_order_item", "recall_pct"] < 20  # structural defect: needs a rule


def test_report_written(out):
    html = out["report_path"].read_text()
    assert "Data Quality Report" in html and "Isolation Forest" in html


def test_isolation_forest_is_reproducible(out):
    X = stats_ml.item_features(out["tables"])
    assert stats_ml.isolation_forest(X, seed=3)[1] == stats_ml.isolation_forest(X, seed=3)[1]
