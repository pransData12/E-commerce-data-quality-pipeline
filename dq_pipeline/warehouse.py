"""SQLite warehouse: load only trustworthy rows, keep an audit trail of the rest.

Quarantine logic
  1. Rows that fail any ``severity: error`` rule are kept out of the warehouse.
  2. Referential cascade: children of removed parents are removed too
     (orders of a removed customer, items of a removed order/product).
  3. An order that lost any line item is removed as well. Its total can no
     longer be trusted, and keeping it would break ``total_amount_mismatch``
     on the next run.

Result: re-validating the warehouse gives zero error-severity failures.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def quarantine(tables: dict[str, pd.DataFrame], violations: pd.DataFrame
               ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    errors = violations[violations["severity"] == "error"]
    bad = errors.groupby("table_name")["row_id"].apply(set).to_dict()
    direct = {name: df.drop(index=list(bad.get(name, set()))) for name, df in tables.items()}

    customers, products = direct["customers"], direct["products"]
    orders = direct["orders"][direct["orders"]["customer_id"].isin(customers["customer_id"])]
    items = direct["order_items"]
    items = items[items["order_id"].isin(orders["order_id"]) & items["product_id"].isin(products["product_id"])]

    lost = tables["order_items"].loc[~tables["order_items"].index.isin(items.index)]
    orders = orders[~orders["order_id"].isin(set(lost["order_id"]))]
    items = items[items["order_id"].isin(orders["order_id"])]

    clean = {"customers": customers, "products": products, "orders": orders, "order_items": items}
    accounting = pd.DataFrame([{
        "table_name": name,
        "raw_rows": len(tables[name]),
        "failed_error_rules": len(bad.get(name, set())),
        "removed_by_cascade": len(direct[name]) - len(clean[name]),
        "loaded_rows": len(clean[name]),
    } for name in tables])
    return clean, accounting


def build_warehouse(db_path: str | Path, tables: dict[str, pd.DataFrame], violations: pd.DataFrame,
                    results: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    clean, accounting = quarantine(tables, violations)
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        for name, df in clean.items():
            df.to_sql(name, con, if_exists="replace", index=False)
        violations.to_sql("dq_violations", con, if_exists="replace", index=False)
        results.to_sql("dq_rule_results", con, if_exists="replace", index=False)
        accounting.to_sql("dq_row_accounting", con, if_exists="replace", index=False)
        con.executescript("""
            CREATE INDEX IF NOT EXISTS ix_orders_customer  ON orders(customer_id);
            CREATE INDEX IF NOT EXISTS ix_items_order      ON order_items(order_id);
            CREATE INDEX IF NOT EXISTS ix_items_product    ON order_items(product_id);
            CREATE INDEX IF NOT EXISTS ix_viol_rule        ON dq_violations(rule_id);
        """)
    return clean, accounting


def read_sql_file(path: str | Path) -> dict[str, str]:
    """Parse a .sql file where each query is preceded by ``-- name: query_name``."""
    queries: dict[str, list[str]] = {}
    current = None
    for line in Path(path).read_text().splitlines():
        if line.strip().lower().startswith("-- name:"):
            current = line.split(":", 1)[1].strip()
            queries[current] = []
        elif current:
            queries[current].append(line)
    return {k: "\n".join(v).strip().rstrip(";") for k, v in queries.items()}


def run_queries(db_path: str | Path, queries: dict[str, str]) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(db_path) as con:
        return {name: pd.read_sql_query(sql, con) for name, sql in queries.items()}
