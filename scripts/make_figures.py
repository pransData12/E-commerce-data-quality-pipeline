"""Regenerate the README figures from a real pipeline run:  python scripts/make_figures.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dq_pipeline import run_pipeline  # noqa: E402

OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def main() -> None:
    o = run_pipeline.run(seed=42)

    # 1. rows failing each rule
    res = o["results"].sort_values("rows_failed")
    colors = res["severity"].map({"error": "#d92d20", "warning": "#f79009"})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(res["rule_id"], res["rows_failed"], color=colors)
    ax.set_xlabel("rows failing the rule")
    ax.set_title("Rule results (red = error, quarantined; orange = warning)")
    fig.tight_layout(); fig.savefig(OUT / "rules_failed.png", dpi=150); plt.close(fig)

    # 2. which method finds wrong-price lines?
    mc = o["method_comparison"].set_index("method")[["precision_pct", "recall_pct", "f1_pct"]]
    ax = mc.plot.barh(figsize=(9, 4.5), color=["#486581", "#9fb3c8", "#d92d20"])
    ax.set_xlabel("%"); ax.set_ylabel("")
    ax.set_title("Finding wrong-price order lines: generic statistics vs ML vs a domain rule")
    plt.tight_layout(); plt.savefig(OUT / "method_comparison.png", dpi=150); plt.close()

    # 3. what dirty data does to revenue
    rev = o["revenue"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["Raw data", "After validation"], [rev["raw"] / 1e6, rev["clean"] / 1e6], color=["#d92d20", "#067647"])
    ax.set_ylabel("revenue (millions)")
    ax.set_title(f"Raw data overstates revenue by {100 * (rev['raw'] - rev['clean']) / rev['clean']:.0f}%")
    fig.tight_layout(); fig.savefig(OUT / "revenue_impact.png", dpi=150); plt.close(fig)
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
