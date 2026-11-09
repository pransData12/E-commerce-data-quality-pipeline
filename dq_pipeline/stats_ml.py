"""Statistics + machine learning layer.

Rules catch defects you can *name*. Statistics and ML help with the rest:

  * profile_columns       - descriptive statistics (mean, std, skew, null %) per numeric column
  * drift_tests           - KS test / chi-square: did cleaning change the distributions?
  * isolation_forest      - unsupervised anomaly detection on order-item features
  * compare_outlier_methods - Tukey IQR vs MAD z-score vs Isolation Forest vs catalog rule,
                              all scored against the injected-defect answer key
  * ml_recall_by_defect   - which defect types can ML see, and which need rules?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

Tables = dict[str, pd.DataFrame]


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #
def profile_columns(tables: Tables) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        for col in df.columns:
            s = df[col]
            row = {"table": name, "column": col, "null_pct": round(100 * s.isna().mean(), 2), "distinct": int(s.nunique())}
            if pd.api.types.is_numeric_dtype(s) and not col.endswith("_id"):
                row.update(mean=round(s.mean(), 2), std=round(s.std(), 2), min=s.min(),
                           median=s.median(), max=s.max(), skew=round(float(stats.skew(s.dropna())), 2))
            rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["mean"]).reset_index(drop=True)


def drift_tests(raw: Tables, clean: Tables) -> pd.DataFrame:
    """Two-sample tests: raw vs. cleaned distributions.

    KS statistic = largest gap between the two empirical CDFs (0 = identical, 1 = disjoint).
    Cramer's V = effect size of the chi-square test for categorical columns.
    With thousands of rows almost everything is 'significant', so read the *effect size*.
    """
    rows = []
    for table, col in [("order_items", "unit_price"), ("order_items", "quantity"), ("orders", "total_amount")]:
        a = pd.to_numeric(raw[table][col], errors="coerce").dropna()
        b = pd.to_numeric(clean[table][col], errors="coerce").dropna()
        ks = stats.ks_2samp(a, b)
        rows.append({"table": table, "column": col, "test": "Kolmogorov-Smirnov",
                     "effect_size": round(float(ks.statistic), 4), "p_value": float(ks.pvalue)})
    for table, col in [("orders", "status"), ("customers", "country")]:
        counts = pd.concat([raw[table][col].value_counts().rename("raw"), clean[table][col].value_counts().rename("clean")],
                           axis=1).fillna(0)
        chi2, p, _, _ = stats.chi2_contingency(counts.to_numpy())
        v = np.sqrt(chi2 / counts.to_numpy().sum())  # Cramer's V for a 2-column table
        rows.append({"table": table, "column": col, "test": "Chi-square",
                     "effect_size": round(float(v), 4), "p_value": float(p)})
    out = pd.DataFrame(rows)
    out["p_value"] = out["p_value"].map(lambda p: f"{p:.2e}")
    return out


# --------------------------------------------------------------------------- #
# Features + Isolation Forest
# --------------------------------------------------------------------------- #
def item_features(tables: Tables, engineered: bool = True) -> pd.DataFrame:
    """Numeric features per order line.

    engineered=False -> only raw columns (price, quantity, catalog price)
    engineered=True  -> adds log(price / catalog price): the domain-knowledge feature
    """
    items = tables["order_items"]
    prod = tables["products"].drop_duplicates("product_id").set_index("product_id")
    catalog = pd.to_numeric(items["product_id"].map(prod["catalog_price"]), errors="coerce")
    price = pd.to_numeric(items["unit_price"], errors="coerce")
    qty = pd.to_numeric(items["quantity"], errors="coerce")
    X = pd.DataFrame({"log_price": np.log1p(price.clip(lower=0)), "quantity": qty,
                      "log_catalog": np.log1p(catalog.clip(lower=0))}, index=items.index)
    if engineered:
        ratio = (price / catalog).where(catalog > 0).clip(lower=1e-3)
        X["log_price_ratio"] = np.log(ratio)
    return X.fillna(X.median())


def isolation_forest(X: pd.DataFrame, contamination: float = 0.03, seed: int = 42) -> tuple[pd.Series, set]:
    """Returns (anomaly_score, flagged_row_labels). Higher score = more anomalous."""
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    model.fit(X)
    score = pd.Series(-model.score_samples(X), index=X.index, name="anomaly_score")
    flagged = set(X.index[model.predict(X) == -1])
    return score, flagged


# --------------------------------------------------------------------------- #
# Method comparison against ground truth
# --------------------------------------------------------------------------- #
def _score(name: str, flagged: set, truth: set) -> dict:
    tp = len(flagged & truth)
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"method": name, "flagged": len(flagged), "true_positives": tp,
            "precision_pct": round(100 * precision, 1), "recall_pct": round(100 * recall, 1), "f1_pct": round(100 * f1, 1)}


def compare_outlier_methods(tables: Tables, violations: pd.DataFrame, defects: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """How well does each method find the injected *wrong-price* lines?"""
    items = tables["order_items"]
    truth = set(defects.loc[(defects.table == "order_items") & (defects.defect_type == "price_inconsistency"), "row_id"])
    price = pd.to_numeric(items["unit_price"], errors="coerce")

    q1, q3 = price.quantile([0.25, 0.75])
    iqr = q3 - q1
    tukey = set(items.index[(price < q1 - 1.5 * iqr) | (price > q3 + 1.5 * iqr)])

    log_p = np.log(price.clip(lower=0.01))
    mad = (log_p - log_p.median()).abs().median()
    modified_z = 0.6745 * (log_p - log_p.median()) / mad  # Iglewicz-Hoaglin modified z-score
    mad_flags = set(items.index[modified_z.abs() > 3.5])

    _, if_raw = isolation_forest(item_features(tables, engineered=False), seed=seed)
    _, if_eng = isolation_forest(item_features(tables, engineered=True), seed=seed)
    rule = set(violations.loc[violations.rule_id == "price_inconsistency", "row_id"])

    return pd.DataFrame([
        _score("Global IQR (Tukey, 1.5x)", tukey, truth),
        _score("Global modified z-score (MAD, log price)", mad_flags, truth),
        _score("Isolation Forest - raw features", if_raw, truth),
        _score("Isolation Forest - + price/catalog ratio", if_eng, truth),
        _score("Catalog-consistency rule (+/-15%)", rule, truth),
    ])


def ml_recall_by_defect(tables: Tables, defects: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Which injected order-item defects does Isolation Forest catch on its own?"""
    _, flagged = isolation_forest(item_features(tables, engineered=True), seed=seed)
    rows = []
    for dtype, grp in defects[defects.table == "order_items"].groupby("defect_type"):
        truth = set(grp["row_id"])
        rows.append({"defect_type": dtype, "injected": len(truth), "caught_by_ml": len(truth & flagged),
                     "recall_pct": round(100 * len(truth & flagged) / len(truth), 1)})
    return pd.DataFrame(rows)
