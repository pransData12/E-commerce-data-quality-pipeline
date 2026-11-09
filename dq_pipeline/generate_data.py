"""Synthetic e-commerce data generator with controlled, *recorded* defect injection.

Why synthetic? Real dirty data has no answer key. Here every defect we inject is
logged to ``_injected_defects.csv``, so the validator can later be scored with
precision and recall against ground truth.

Design rules that keep the answer key honest:
  * The clean data is fully consistent before any defect is injected.
  * Each defect touches a *disjoint* set of rows, so one row = one defect type.
  * Order totals are recomputed after item-level defects, so a bad item does not
    silently create a second (cascading) ``total_amount_mismatch``.
  * Future-dated signups are only injected on customers with no orders, so they
    do not cascade into ``order_before_signup``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2022-01-01")
SIGNUP_END = pd.Timestamp("2024-06-30")
END = pd.Timestamp("2025-06-30")
AS_OF = pd.Timestamp("2025-07-01")  # fixed "today" so runs are reproducible

FIRST_NAMES = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Vikram", "Isha",
               "Arjun", "Sneha", "Liam", "Emma", "Noah", "Olivia", "Lukas", "Mia"]
LAST_NAMES = ["Sharma", "Iyer", "Reddy", "Patel", "Nair", "Singh", "Gupta", "Menon",
              "Smith", "Brown", "Muller", "Tan", "Wilson", "Khan", "Das", "Rao"]
COUNTRIES = ["India", "United States", "United Kingdom", "Germany", "Singapore", "Australia"]
COUNTRY_P = [0.55, 0.15, 0.10, 0.08, 0.07, 0.05]

# category -> (min, max) catalog price. Wide, *different* ranges per category are
# exactly why a global IQR outlier test is a poor fit here (see README).
CATEGORY_PRICE_RANGE = {
    "Electronics": (2000, 90000),
    "Books": (150, 1500),
    "Home & Kitchen": (400, 12000),
    "Fashion": (300, 6000),
    "Sports": (500, 15000),
    "Beauty": (100, 3000),
}
STATUSES = ["completed", "shipped", "pending", "cancelled", "returned"]
STATUS_P = [0.55, 0.15, 0.08, 0.12, 0.10]

# defect_type (== rule id in config/rules.yaml) -> fraction of its table affected
DEFECT_RATES = {
    "null_email": 0.02,
    "invalid_email": 0.02,
    "future_signup_date": 0.01,
    "duplicate_customer_id": 0.01,
    "non_positive_catalog_price": 0.02,
    "invalid_status": 0.015,
    "orphan_order_customer": 0.01,
    "order_before_signup": 0.01,
    "total_amount_mismatch": 0.015,
    "invalid_quantity": 0.01,
    "price_inconsistency": 0.015,
    "orphan_order_item": 0.01,
}


# --------------------------------------------------------------------------- #
# Clean data
# --------------------------------------------------------------------------- #
def generate_clean(n_customers: int, n_products: int, n_orders: int,
                   rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    ids = np.arange(1, n_customers + 1)
    first = rng.choice(FIRST_NAMES, n_customers)
    last = rng.choice(LAST_NAMES, n_customers)
    signup_days = rng.integers(0, (SIGNUP_END - START).days + 1, n_customers)
    customers = pd.DataFrame({
        "customer_id": ids,
        "name": [f"{f} {l}" for f, l in zip(first, last)],
        "email": [f"{f}.{l}{i}@example.com".lower() for f, l, i in zip(first, last, ids)],
        "signup_date": START + pd.to_timedelta(signup_days, unit="D"),
        "country": rng.choice(COUNTRIES, n_customers, p=COUNTRY_P),
    })

    cats = rng.choice(list(CATEGORY_PRICE_RANGE), n_products)
    lo = np.array([CATEGORY_PRICE_RANGE[c][0] for c in cats])
    hi = np.array([CATEGORY_PRICE_RANGE[c][1] for c in cats])
    products = pd.DataFrame({
        "product_id": np.arange(1, n_products + 1),
        "name": [f"{c} Item {i}" for i, c in enumerate(cats, start=1)],
        "category": cats,
        "catalog_price": np.round(rng.uniform(lo, hi), 2),
    })

    # ~10% of customers never order: handy for safe defect injection later.
    active = rng.choice(ids, size=int(n_customers * 0.9), replace=False)
    cust_of_order = rng.choice(active, n_orders)
    signup = customers.set_index("customer_id").loc[cust_of_order, "signup_date"]
    span = (END - signup).dt.days.to_numpy()
    offset = (rng.random(n_orders) * span).astype(int)
    orders = pd.DataFrame({
        "order_id": np.arange(1, n_orders + 1),
        "customer_id": cust_of_order,
        "order_date": pd.to_datetime(signup.to_numpy()) + pd.to_timedelta(offset, unit="D"),
        "status": rng.choice(STATUSES, n_orders, p=STATUS_P),
        "total_amount": 0.0,  # filled in after item defects are injected
    })

    n_items = rng.integers(1, 5, n_orders)
    order_ids = np.repeat(orders["order_id"].to_numpy(), n_items)
    product_ids = rng.integers(1, n_products + 1, len(order_ids))
    catalog = products.set_index("product_id").loc[product_ids, "catalog_price"].to_numpy()
    items = pd.DataFrame({
        "item_id": np.arange(1, len(order_ids) + 1),
        "order_id": order_ids,
        "product_id": product_ids,
        "quantity": rng.integers(1, 5, len(order_ids)),
        # small legitimate discounts / markups around catalog price
        "unit_price": np.round(catalog * rng.uniform(0.92, 1.08, len(order_ids)), 2),
    })
    return {"customers": customers, "products": products, "orders": orders, "order_items": items}


# --------------------------------------------------------------------------- #
# Defect injection
# --------------------------------------------------------------------------- #
class _Injector:
    """Tracks used rows per table (so defects never overlap) and the answer key."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.used: dict[str, set[int]] = {}
        self.log: list[tuple[str, int, str]] = []

    def pick(self, table: str, pool, n: int) -> np.ndarray:
        used = self.used.setdefault(table, set())
        avail = np.setdiff1d(np.asarray(pool, dtype=int), np.array(sorted(used), dtype=int))
        chosen = np.sort(self.rng.choice(avail, size=min(n, len(avail)), replace=False))
        used.update(chosen.tolist())
        return chosen

    def record(self, table: str, rows, defect: str) -> None:
        self.log.extend((table, int(r), defect) for r in rows)


def _count(rates: dict, key: str, n_rows: int) -> int:
    return max(1, round(rates[key] * n_rows))


def inject_defects(t: dict[str, pd.DataFrame], rng: np.random.Generator,
                   rates: dict | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rates = {**DEFECT_RATES, **(rates or {})}
    inj = _Injector(rng)
    cust, prod = t["customers"].copy(), t["products"].copy()
    orders, items = t["orders"].copy(), t["order_items"].copy()

    # ---- customers -------------------------------------------------------- #
    n = len(cust)
    idx = inj.pick("customers", cust.index, _count(rates, "null_email", n))
    cust.loc[idx, "email"] = None
    inj.record("customers", idx, "null_email")

    idx = inj.pick("customers", cust.index, _count(rates, "invalid_email", n))
    half = len(idx) // 2
    cust.loc[idx[:half], "email"] = cust.loc[idx[:half], "email"].str.replace("@", "", regex=False)
    cust.loc[idx[half:], "email"] = cust.loc[idx[half:], "email"].str.replace(".com", "", regex=False)
    inj.record("customers", idx, "invalid_email")

    no_orders = cust.index[~cust["customer_id"].isin(set(orders["customer_id"]))]
    idx = inj.pick("customers", no_orders, _count(rates, "future_signup_date", n))
    cust.loc[idx, "signup_date"] = (AS_OF + pd.to_timedelta(rng.integers(30, 400, len(idx)), unit="D")).to_numpy()
    inj.record("customers", idx, "future_signup_date")

    signup_lookup = cust.drop_duplicates("customer_id").set_index("customer_id")["signup_date"]

    src = inj.pick("customers", cust.index, _count(rates, "duplicate_customer_id", n))
    dups = cust.loc[src].copy()
    start = len(cust)
    cust = pd.concat([cust, dups], ignore_index=True)  # appended => original row indices unchanged
    inj.record("customers", np.arange(start, start + len(dups)), "duplicate_customer_id")

    # ---- products --------------------------------------------------------- #
    idx = inj.pick("products", prod.index, _count(rates, "non_positive_catalog_price", len(prod)))
    prod.loc[idx, "catalog_price"] = rng.choice([0.0, -5.0], len(idx))
    inj.record("products", idx, "non_positive_catalog_price")
    bad_products = set(prod.loc[idx, "product_id"])

    # ---- order_items ------------------------------------------------------ #
    n = len(items)
    idx = inj.pick("order_items", items.index, _count(rates, "invalid_quantity", n))
    items.loc[idx, "quantity"] = rng.choice([0, -1, -3, 500], len(idx))
    inj.record("order_items", idx, "invalid_quantity")

    pool = items.index[~items["product_id"].isin(bad_products)]  # need a valid catalog price to compare
    idx = inj.pick("order_items", pool, _count(rates, "price_inconsistency", n))
    factor = rng.choice([10.0, 0.1], len(idx))
    items.loc[idx, "unit_price"] = np.round(items.loc[idx, "unit_price"].to_numpy() * factor, 2)
    inj.record("order_items", idx, "price_inconsistency")

    idx = inj.pick("order_items", items.index, _count(rates, "orphan_order_item", n))
    items.loc[idx, "order_id"] = 800_000 + np.arange(len(idx))
    inj.record("order_items", idx, "orphan_order_item")

    # ---- order totals: computed AFTER item defects so they stay consistent --- #
    line = items["quantity"] * items["unit_price"]
    orders["total_amount"] = orders["order_id"].map(line.groupby(items["order_id"]).sum()).fillna(0).round(2)

    # ---- orders ----------------------------------------------------------- #
    n = len(orders)
    idx = inj.pick("orders", orders.index, _count(rates, "invalid_status", n))
    orders.loc[idx, "status"] = rng.choice(["COMPLETE", "delivered", "unknown", "Shipped "], len(idx))
    inj.record("orders", idx, "invalid_status")

    idx = inj.pick("orders", orders.index, _count(rates, "orphan_order_customer", n))
    orders.loc[idx, "customer_id"] = 900_000 + np.arange(len(idx))
    inj.record("orders", idx, "orphan_order_customer")

    idx = inj.pick("orders", orders.index, _count(rates, "order_before_signup", n))
    signup = orders.loc[idx, "customer_id"].map(signup_lookup)
    orders.loc[idx, "order_date"] = (signup - pd.to_timedelta(rng.integers(1, 120, len(idx)), unit="D")).to_numpy()
    inj.record("orders", idx, "order_before_signup")

    idx = inj.pick("orders", orders.index, _count(rates, "total_amount_mismatch", n))
    orders.loc[idx, "total_amount"] = (orders.loc[idx, "total_amount"] + np.round(rng.uniform(50, 500, len(idx)), 2)).round(2)
    inj.record("orders", idx, "total_amount_mismatch")

    # ---- serialise dates as ISO strings (what a CSV would really contain) --- #
    cust["signup_date"] = pd.to_datetime(cust["signup_date"]).dt.strftime("%Y-%m-%d")
    orders["order_date"] = pd.to_datetime(orders["order_date"]).dt.strftime("%Y-%m-%d")

    defects = pd.DataFrame(inj.log, columns=["table", "row_id", "defect_type"])
    return {"customers": cust, "products": prod, "orders": orders, "order_items": items}, defects


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate(out_dir: str | Path, seed: int = 42, n_customers: int = 1500, n_products: int = 200,
             n_orders: int = 5000, rates: dict | None = None) -> pd.DataFrame:
    """Write the four raw CSVs plus the defect answer key. Returns the answer key."""
    rng = np.random.default_rng(seed)
    clean = generate_clean(n_customers, n_products, n_orders, rng)
    dirty, defects = inject_defects(clean, rng, rates)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in dirty.items():
        df.to_csv(out / f"{name}.csv", index=False)
    defects.to_csv(out / "_injected_defects.csv", index=False)
    return defects
