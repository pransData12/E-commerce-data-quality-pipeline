"""End-to-end pipeline:  generate -> validate -> quarantine -> warehouse -> SQL -> report."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from . import generate_data, report, stats_ml, validators, warehouse

ROOT = Path(__file__).resolve().parents[1]
TABLE_NAMES = ["customers", "products", "orders", "order_items"]
REVENUE_STATUSES = ["completed", "shipped"]
log = logging.getLogger("dq_pipeline")


def load_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(data_dir / f"{name}.csv") for name in TABLE_NAMES}


def _revenue(orders: pd.DataFrame) -> float:
    return float(orders.loc[orders["status"].isin(REVENUE_STATUSES), "total_amount"].sum())


def run(seed: int = 42, config: Path = ROOT / "config" / "rules.yaml", data_dir: Path = ROOT / "data" / "raw",
        db_path: Path = ROOT / "warehouse" / "ecommerce.db", report_path: Path = ROOT / "reports" / "dq_report.html",
        sql_path: Path = ROOT / "sql" / "business_queries.sql", generate: bool = True) -> dict:
    data_dir = Path(data_dir)
    if generate:
        log.info("1/6 generating synthetic data (seed=%s)", seed)
        generate_data.generate(data_dir, seed=seed)
    tables = load_raw(data_dir)
    log.info("loaded raw tables: %s", {k: len(v) for k, v in tables.items()})

    log.info("2/6 validating")
    cfg = validators.load_config(config)
    results, violations = validators.run_validation(tables, cfg)

    evaluation = None
    answer_key = data_dir / "_injected_defects.csv"
    if answer_key.exists():
        evaluation = validators.evaluate_detection(violations, pd.read_csv(answer_key), list(results["rule_id"]))

    log.info("3/6 quarantining + loading warehouse")
    clean, accounting = warehouse.build_warehouse(db_path, tables, violations, results)

    log.info("4/6 statistics + ML (profiling, drift tests, Isolation Forest)")
    profile = stats_ml.profile_columns(tables)
    drift = stats_ml.drift_tests(tables, clean)
    method_comparison = ml_recall = None
    if answer_key.exists():
        defects = pd.read_csv(answer_key)
        method_comparison = stats_ml.compare_outlier_methods(tables, violations, defects, seed=seed)
        ml_recall = stats_ml.ml_recall_by_defect(tables, defects, seed=seed)

    log.info("5/6 running business queries")
    queries = warehouse.run_queries(db_path, warehouse.read_sql_file(sql_path))

    log.info("6/6 writing report")
    samples = {}
    for rule_id, group in violations.groupby("rule_id"):
        table = group["table_name"].iloc[0]
        samples[rule_id] = tables[table].loc[group["row_id"].head(5)]
    ctx = {"seed": seed, "results": results, "violations": violations, "accounting": accounting,
           "evaluation": evaluation, "queries": queries, "samples": samples, "profile": profile, "drift": drift,
           "method_comparison": method_comparison, "ml_recall": ml_recall,
           "revenue": {"raw": _revenue(tables["orders"]), "clean": _revenue(clean["orders"])}}
    out = report.write_report(report_path, ctx)
    return {**ctx, "report_path": out, "db_path": Path(db_path), "tables": tables, "clean": clean}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E-commerce data quality & validation pipeline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=Path, default=ROOT / "config" / "rules.yaml")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--db", type=Path, default=ROOT / "warehouse" / "ecommerce.db")
    p.add_argument("--report", type=Path, default=ROOT / "reports" / "dq_report.html")
    p.add_argument("--skip-generate", action="store_true", help="reuse existing CSVs in --data-dir")
    p.add_argument("--strict", action="store_true", help="exit with code 1 if any error-severity rule fails (CI gate)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = run(args.seed, args.config, args.data_dir, args.db, args.report, generate=not args.skip_generate)

    res, ev = out["results"], out["evaluation"]
    failing = res[res["rows_failed"] > 0]
    print(f"\n{len(failing)}/{len(res)} rules have failures; report -> {out['report_path']}")
    print(failing[["rule_id", "severity", "rows_failed", "fail_rate_pct"]].to_string(index=False))
    if ev is not None:
        scored = ev.dropna(subset=["recall_pct"])
        print(f"\nMean recall {scored['recall_pct'].mean():.1f}% | mean precision {scored['precision_pct'].mean():.1f}% vs injected defects")
    if args.strict and ((res["severity"] == "error") & (res["rows_failed"] > 0)).any():
        return 1
    return 0
