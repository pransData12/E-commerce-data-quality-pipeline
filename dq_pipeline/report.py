"""Self-contained HTML data-quality report (no external assets, opens anywhere)."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f6f8;color:#1f2933}
main{max-width:1100px;margin:0 auto;padding:32px 20px 60px}
h1{margin:0 0 4px}h2{margin:36px 0 12px;border-bottom:2px solid #dde1e7;padding-bottom:6px}
.sub{color:#66788a;margin-bottom:24px}
.cards{display:flex;gap:14px;flex-wrap:wrap}
.card{background:#fff;border-radius:10px;padding:16px 20px;min-width:170px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.card b{display:block;font-size:26px}.card span{color:#66788a;font-size:13px}
table{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden;
      box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:13.5px;margin-bottom:8px}
th{background:#243b53;color:#fff;text-align:left;padding:9px 10px}
td{padding:8px 10px;border-top:1px solid #edf0f3}
.err{color:#b42318;font-weight:600}.warn{color:#b54708;font-weight:600}.ok{color:#067647;font-weight:600}
.bar{background:#e4e7ec;border-radius:4px;height:10px;width:140px;display:inline-block;vertical-align:middle}
.bar i{display:block;height:10px;border-radius:4px;background:#d92d20}
details{background:#fff;border-radius:8px;padding:8px 14px;margin:6px 0;box-shadow:0 1px 3px rgba(0,0,0,.08)}
summary{cursor:pointer;font-weight:600}.note{color:#66788a;font-size:13px}
"""


def _df(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='note'>No rows.</p>"
    return df.to_html(index=False, border=0, escape=True, na_rep="-")


def _rules_table(results: pd.DataFrame) -> str:
    peak = max(results["fail_rate_pct"].max(), 1e-9)
    rows = []
    for r in results.itertuples():
        sev = "err" if r.severity == "error" else "warn"
        status = "<span class='ok'>PASS</span>" if r.rows_failed == 0 else f"<span class='{sev}'>{r.severity.upper()}</span>"
        width = 100 * r.fail_rate_pct / peak
        rows.append(
            f"<tr><td><code>{html.escape(r.rule_id)}</code></td><td>{html.escape(r.table_name)}</td>"
            f"<td>{html.escape(r.description)}</td><td>{r.rows_checked:,}</td><td>{r.rows_failed:,}</td>"
            f"<td><span class='bar'><i style='width:{width:.0f}%'></i></span> {r.fail_rate_pct}%</td><td>{status}</td></tr>"
        )
    head = "<tr><th>Rule</th><th>Table</th><th>What it checks</th><th>Checked</th><th>Failed</th><th>Fail rate</th><th>Status</th></tr>"
    return f"<table>{head}{''.join(rows)}</table>"


def render_report(ctx: dict) -> str:
    results: pd.DataFrame = ctx["results"]
    violations: pd.DataFrame = ctx["violations"]
    accounting: pd.DataFrame = ctx["accounting"]
    raw_rows = int(accounting["raw_rows"].sum())
    loaded = int(accounting["loaded_rows"].sum())
    errors = int((violations["severity"] == "error").sum())
    warnings = int((violations["severity"] == "warning").sum())
    rules_failing = int((results["rows_failed"] > 0).sum())

    cards = "".join(f"<div class='card'><b>{v}</b><span>{k}</span></div>" for k, v in [
        ("raw rows ingested", f"{raw_rows:,}"),
        ("rows loaded to warehouse", f"{loaded:,}"),
        ("error violations", f"{errors:,}"),
        ("warning violations", f"{warnings:,}"),
        ("rules failing", f"{rules_failing} / {len(results)}"),
    ])

    parts = [
        f"<h1>Data Quality Report</h1><div class='sub'>E-commerce pipeline &middot; seed {ctx['seed']} &middot; "
        f"generated {datetime.now():%Y-%m-%d %H:%M}</div>",
        f"<div class='cards'>{cards}</div>",
        "<h2>Rule results</h2>", _rules_table(results),
        "<h2>Row accounting</h2>",
        "<p class='note'>Rows failing an <b>error</b> rule are quarantined. Children of removed parents (and orders that lost a "
        "line item) are removed by cascade, so the warehouse stays referentially consistent.</p>", _df(accounting),
    ]

    if ctx.get("revenue"):
        rev = ctx["revenue"]
        inflation = 100 * (rev["raw"] - rev["clean"]) / rev["clean"] if rev["clean"] else 0
        parts += [
            "<h2>Why it matters: impact on revenue</h2>",
            f"<p>Reporting revenue straight from the raw data gives <b>{rev['raw']:,.0f}</b>. After validation it is "
            f"<b>{rev['clean']:,.0f}</b>. The raw figure is off by <b>{inflation:+.1f}%</b>: mispriced lines, broken totals "
            f"and invalid orders would have gone straight into the dashboard.</p>",
        ]

    if ctx.get("evaluation") is not None:
        parts += [
            "<h2>Detection accuracy vs. injected defects</h2>",
            "<p class='note'>The generator logs every defect it injects, so the validator can be scored: "
            "recall = injected defects found; precision = flagged rows that were really injected.</p>",
            _df(ctx["evaluation"]),
        ]

    parts += ["<h2>Statistics: column profile (raw data)</h2>",
              "<p class='note'>Extreme skew and max values far above the median are the first statistical hint that something is wrong "
              "(e.g. an order total of 26 million when the median is 33 thousand).</p>", _df(ctx["profile"])]
    parts += ["<h2>Statistics: did cleaning change the distributions?</h2>",
              "<p class='note'>KS statistic / Cramer's V are effect sizes (0 = identical). Numeric shapes barely move, which is what we want: "
              "we removed corrupt rows, not a segment of real customers. Order status shifts because invalid labels were removed.</p>",
              _df(ctx["drift"])]
    if ctx.get("method_comparison") is not None:
        parts += ["<h2>ML: finding wrong-price lines, method comparison</h2>",
                  "<p class='note'>Each method is scored against the injected wrong-price lines. Global IQR and MAD ignore that a "
                  "normal price differs by category; Isolation Forest only works once it is given a price-vs-catalog feature. "
                  "Domain knowledge (the catalog) beats a generic statistic.</p>", _df(ctx["method_comparison"]),
                  "<h3>What can Isolation Forest see on its own? (recall by defect type)</h3>",
                  "<p class='note'>It catches value anomalies but is nearly blind to structural defects (orphan keys) and weak on small "
                  "deviations. That is why rules and ML are combined rather than one replacing the other.</p>", _df(ctx["ml_recall"])]

    parts.append("<h2>Business questions (clean warehouse)</h2>")
    for name, df in ctx["queries"].items():
        parts += [f"<h3>{html.escape(name.replace('_', ' ').title())}</h3>", _df(df)]

    parts.append("<h2>Sample violating rows</h2>")
    for rule_id, sample in ctx["samples"].items():
        parts.append(f"<details><summary><code>{html.escape(rule_id)}</code></summary>{_df(sample)}</details>")

    return f"<!doctype html><html><head><meta charset='utf-8'><title>Data Quality Report</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>"


def write_report(path: str | Path, ctx: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(ctx), encoding="utf-8")
    return out
