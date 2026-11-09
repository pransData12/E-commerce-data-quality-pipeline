"""Config-driven validation engine.

Every rule in ``config/rules.yaml`` names a ``type``. Each type maps to one small
function registered below. A check returns ``(failed_row_labels, rows_checked)``.
Adding a new check type = write one function + decorate it. No other code changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

Tables = dict[str, pd.DataFrame]
CheckFn = Callable[[Tables, dict, pd.Timestamp], tuple[pd.Index, int]]

REGISTRY: dict[str, CheckFn] = {}
SEVERITIES = {"error", "warning"}


class ConfigError(ValueError):
    """Raised when rules.yaml is malformed."""


def register(name: str) -> Callable[[CheckFn], CheckFn]:
    def deco(fn: CheckFn) -> CheckFn:
        REGISTRY[name] = fn
        return fn
    return deco


def _failed(df: pd.DataFrame, mask) -> pd.Index:
    return df.index[pd.Series(mask, index=df.index).fillna(False).to_numpy(dtype=bool)]


# --------------------------------------------------------------------------- #
# Column-level checks
# --------------------------------------------------------------------------- #
@register("not_null")
def not_null(tables, rule, as_of):
    df, s = tables[rule["table"]], tables[rule["table"]][rule["column"]]
    blank = s.isna() | (s.astype(str).str.strip() == "")
    return _failed(df, blank), len(df)


@register("unique")
def unique(tables, rule, as_of):
    df, s = tables[rule["table"]], tables[rule["table"]][rule["column"]]
    # keep="first": only the *repeat* occurrences fail, the original row is fine
    return _failed(df, s.duplicated(keep="first") & s.notna()), int(s.notna().sum())


@register("regex")
def regex(tables, rule, as_of):
    df, s = tables[rule["table"]], tables[rule["table"]][rule["column"]]
    present = s.notna()
    matches = s.astype(str).str.fullmatch(rule["pattern"])
    return _failed(df, present & ~matches), int(present.sum())


@register("range")
def value_range(tables, rule, as_of):
    df, s = tables[rule["table"]], tables[rule["table"]][rule["column"]]
    num, present = pd.to_numeric(s, errors="coerce"), s.notna()
    bad = present & num.isna()  # non-numeric garbage
    if "min" in rule:
        bad = bad | (num < rule["min"])
    if "max" in rule:
        bad = bad | (num > rule["max"])
    return _failed(df, bad), int(present.sum())


@register("allowed_values")
def allowed_values(tables, rule, as_of):
    df, s = tables[rule["table"]], tables[rule["table"]][rule["column"]]
    return _failed(df, s.notna() & ~s.isin(rule["values"])), int(s.notna().sum())


@register("date_not_future")
def date_not_future(tables, rule, as_of):
    df, s = tables[rule["table"]], tables[rule["table"]][rule["column"]]
    parsed, present = pd.to_datetime(s, errors="coerce"), s.notna()
    return _failed(df, present & (parsed.isna() | (parsed > as_of))), int(present.sum())


# --------------------------------------------------------------------------- #
# Cross-table / business-logic checks
# --------------------------------------------------------------------------- #
@register("foreign_key")
def foreign_key(tables, rule, as_of):
    df, s = tables[rule["table"]], tables[rule["table"]][rule["column"]]
    parent = tables[rule["ref_table"]][rule["ref_column"]]
    return _failed(df, s.notna() & ~s.isin(parent)), int(s.notna().sum())


@register("date_order")
def date_order(tables, rule, as_of):
    """Child date must not be earlier than the parent's date (order >= signup)."""
    df = tables[rule["table"]]
    parents = tables[rule["ref_table"]].drop_duplicates(rule["ref_key"]).set_index(rule["ref_key"])
    parent_date = pd.to_datetime(df[rule["key"]].map(parents[rule["ref_column"]]), errors="coerce")
    child_date = pd.to_datetime(df[rule["column"]], errors="coerce")
    testable = child_date.notna() & parent_date.notna()  # orphans are caught by the FK rule
    return _failed(df, testable & (child_date < parent_date)), int(testable.sum())


@register("price_consistency")
def price_consistency(tables, rule, as_of):
    """Transaction price must be within +/- tolerance of the product's catalog price.

    Replaces a global IQR test: a 50,000 price is an "outlier" for Books but totally
    normal for Electronics, and IQR cannot know that. The catalog can.
    """
    df = tables[rule["table"]]
    ref = tables[rule["ref_table"]].drop_duplicates(rule["ref_key"]).set_index(rule["ref_key"])
    catalog = pd.to_numeric(df[rule["product_column"]].map(ref[rule["ref_price"]]), errors="coerce")
    price = pd.to_numeric(df[rule["column"]], errors="coerce")
    testable = catalog.gt(0) & price.notna()
    deviation = (price / catalog - 1).abs()
    return _failed(df, testable & (deviation > rule["tolerance"])), int(testable.sum())


@register("sum_match")
def sum_match(tables, rule, as_of):
    """Parent total must equal the sum of its child lines (qty * price)."""
    df, child = tables[rule["table"]], tables[rule["child_table"]]
    line = pd.to_numeric(child[rule["child_qty"]], errors="coerce") * pd.to_numeric(child[rule["child_price"]], errors="coerce")
    expected = df[rule["key"]].map(line.groupby(child[rule["child_key"]]).sum()).fillna(0)
    total = pd.to_numeric(df[rule["column"]], errors="coerce")
    testable = total.notna()
    return _failed(df, testable & ((total - expected).abs() > rule["tolerance"])), int(testable.sum())


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or "rules" not in cfg:
        raise ConfigError("rules.yaml must contain a top-level 'rules' list")
    seen: set[str] = set()
    for rule in cfg["rules"]:
        for key in ("id", "type", "table", "severity", "description"):
            if key not in rule:
                raise ConfigError(f"rule {rule.get('id', rule)!r} is missing required key '{key}'")
        if rule["id"] in seen:
            raise ConfigError(f"duplicate rule id: {rule['id']}")
        seen.add(rule["id"])
        if rule["type"] not in REGISTRY:
            raise ConfigError(f"rule '{rule['id']}': unknown type '{rule['type']}' (known: {sorted(REGISTRY)})")
        if rule["severity"] not in SEVERITIES:
            raise ConfigError(f"rule '{rule['id']}': severity must be one of {sorted(SEVERITIES)}")
    return cfg


def run_validation(tables: Tables, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every rule. Returns (rule_results, violations).

    rule_results: one row per rule (rows_checked, rows_failed, fail_rate ...)
    violations:   one row per failing record (rule_id, table_name, row_id, severity)
    """
    as_of = pd.Timestamp(cfg.get("as_of", pd.Timestamp.today().normalize()))
    results, violations = [], []
    for rule in cfg["rules"]:
        try:
            failed, checked = REGISTRY[rule["type"]](tables, rule, as_of)
        except KeyError as exc:
            raise ConfigError(f"rule '{rule['id']}' references missing table/column/param {exc}") from exc
        results.append({
            "rule_id": rule["id"], "description": rule["description"], "rule_type": rule["type"],
            "table_name": rule["table"], "severity": rule["severity"],
            "rows_checked": checked, "rows_failed": len(failed),
            "fail_rate_pct": round(100 * len(failed) / checked, 2) if checked else 0.0,
        })
        violations.extend(
            {"rule_id": rule["id"], "table_name": rule["table"], "row_id": int(r), "severity": rule["severity"]}
            for r in failed
        )
    cols = ["rule_id", "table_name", "row_id", "severity"]
    return pd.DataFrame(results), pd.DataFrame(violations, columns=cols)


def evaluate_detection(violations: pd.DataFrame, defects: pd.DataFrame, rule_ids: list[str]) -> pd.DataFrame:
    """Score the validator against the generator's answer key (precision / recall per rule)."""
    rows = []
    for rid in rule_ids:
        truth = set(zip(defects.loc[defects.defect_type == rid, "table"], defects.loc[defects.defect_type == rid, "row_id"]))
        v = violations[violations.rule_id == rid]
        flagged = set(zip(v["table_name"], v["row_id"]))
        tp = len(truth & flagged)
        rows.append({
            "rule_id": rid, "injected": len(truth), "flagged": len(flagged), "true_positives": tp,
            "recall_pct": round(100 * tp / len(truth), 1) if truth else None,
            "precision_pct": round(100 * tp / len(flagged), 1) if flagged else None,
        })
    return pd.DataFrame(rows)
