"""Figures and the auto-generated Markdown report."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C

PALETTE = ["#2a6f97", "#c1121f", "#588157", "#e09f3e", "#6d597a", "#468189", "#9c6644"]


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    if df.empty:
        return "_(no rows)_"
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def fig_km(curves: dict, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    t = np.linspace(0, 1100, 400)
    for i, (name, curve) in enumerate(sorted(curves.items())):
        ax.plot(t / C.DAYS_PER_MONTH, curve(t), label=f"{name.replace('|', ' / ')} (n={curve.n:,})",
                color=PALETTE[i % len(PALETTE)], lw=1.8)
    ax.set_xlabel("Months since subscription start"); ax.set_ylabel("Survival probability S(t)")
    ax.set_title("Kaplan-Meier retention curves by plan and billing period")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.25); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_calibration(pred: pd.DataFrame, path, horizon=6, model="Cox PH"):
    d = pred[(pred["horizon_months"] == horizon) & (pred["model"] == model)]
    g = d.groupby("plan").agg(pred=("pred_churn", "mean"), actual=("churned", "mean"), n=("churned", "size"))
    g = g[g["n"] >= 200].sort_values("actual")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(g)); w = 0.38
    ax.bar(x - w / 2, g["pred"], w, label="Predicted", color=PALETTE[0])
    ax.bar(x + w / 2, g["actual"], w, label="Actual", color=PALETTE[3])
    ax.set_xticks(x); ax.set_xticklabels([i.replace("|", "\n") for i in g.index], fontsize=8)
    ax.set_ylabel(f"Churn rate within {horizon} months")
    ax.set_title(f"Out-of-time calibration by plan ({model}, pooled over forecast origins)")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_revenue(summary: pd.DataFrame, path):
    hs = sorted(summary["horizon_months"].unique())
    models = list(summary["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, title in ((axes[0], "RMSE", "Per-customer RMSE (lower is better)"),
                              (axes[1], "portfolio_bias_pct", "Portfolio bias, % (0 is perfect)")):
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            vals = [summary[(summary["horizon_months"] == h) & (summary["model"] == m)][metric].mean() for h in hs]
            ax.bar(np.arange(len(hs)) + i * w, vals, w, label=m, color=PALETTE[i % len(PALETTE)])
        ax.set_xticks(np.arange(len(hs)) + 0.4 - w / 2); ax.set_xticklabels([f"{h} months" for h in hs])
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.25)
        if metric == "portfolio_bias_pct":
            ax.axhline(0, color="black", lw=.8)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Revenue forecasting backtest (paid monthly customers, rolling origins)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_clv(scores: pd.DataFrame, path, col="clv_36m"):
    positive = scores[scores["clv_36m"] > 0]
    order = positive.groupby("segment")[col].median().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].boxplot([positive.loc[positive["segment"] == s, col] for s in order], vert=False,
                    tick_labels=[s.replace("|", " / ") for s in order], showfliers=False)
    axes[0].set_xlabel("36-month discounted CLV"); axes[0].set_title("CLV by plan (positive-CLV customers)", fontsize=10)
    axes[0].grid(axis="x", alpha=.25)
    v = np.sort(positive[col].to_numpy())[::-1]
    cum = np.cumsum(v) / v.sum(); share = np.arange(1, len(v) + 1) / len(v)
    axes[1].plot(share * 100, cum * 100, color=PALETTE[0], lw=2)
    axes[1].plot([0, 100], [0, 100], "--", color="grey", lw=1)
    axes[1].set_xlabel("% of positive-CLV customers (highest CLV first)"); axes[1].set_ylabel("% of total CLV")
    axes[1].set_title("Value concentration", fontsize=10); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_uncertainty(total_reps, predictive_draws, point, path):
    """Histogram of bootstrap totals (estimation) and simulated revenue (estimation + randomness)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    scale = 1e6
    ax.hist(np.asarray(predictive_draws) / scale, bins=40, density=True, alpha=.45, color=PALETTE[1], label="Predictive (estimation + random churn/payments)")
    ax.hist(np.asarray(total_reps) / scale, bins=25, density=True, alpha=.75, color=PALETTE[0], label="Bootstrap (estimation only)")
    ax.axvline(point / scale, color="black", ls="--", lw=1.2, label="Point estimate")
    ax.set_xlabel("Portfolio 36-month discounted revenue (millions)"); ax.set_ylabel("Density")
    ax.set_title("Uncertainty around portfolio CLV", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
