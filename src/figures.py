"""Phase 4 figures (PNG, 300 dpi) -> results/figures/. Run after evaluate.py."""
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIGS, PROC, RESULTS, enforce_seal  # noqa: E402

# fixed categorical slots (validated default palette, light mode) - colour follows the model
COLORS = {"M0": "#2a78d6", "M1": "#eb6834", "M2": "#1baf7a", "M3": "#eda100", "PLACEBO": "#e87ba4",
          "LGBM": "#008300", "LSTM": "#6250d6", "M4": "#e34948"}
# LGBM_M4 (secondary check) is in the tables only: an 8-slot palette never gets a generated 9th hue
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.4, "figure.dpi": 100})


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    tmp = FIGS / f"{name}.tmp.png"
    fig.savefig(tmp, dpi=300, bbox_inches="tight", facecolor="white")
    os.replace(tmp, FIGS / f"{name}.png")
    plt.close(fig)


def cumulative_long_short():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    for r, group in enumerate(["full", "li7"]):
        net = pd.read_csv(RESULTS / f"portfolio_daily_net_{group}.csv", index_col=0, parse_dates=True)
        for c, kind in enumerate(["gross", "net"]):
            ax = axes[r, c]
            for m in [c for c in net.columns if c in COLORS]:
                s = net[m]
                if kind == "gross":
                    s = gross_series(group, m)
                ax.plot(s.index, s.cumsum() * 100, color=COLORS.get(m, MUTED), label=m)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_title(f"{'Full universe' if group == 'full' else 'Li-7 subset'}: "
                         f"{'gross' if kind == 'gross' else 'net of 10 bps/side'}", fontsize=8, color=INK, loc="left")
    fig.supylabel("cumulative long-short return (%, daily returns summed)", fontsize=8, color=INK)
    axes[0, 0].legend(ncol=4, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.32))
    save(fig, "cumulative_long_short")


_GROSS = {}


def gross_series(group, m):
    """Recompute gross daily spread for one model (evaluate.portfolio), cached."""
    from config import LI_STOCKS
    from evaluate import load_preds, portfolio
    if group not in _GROSS:
        p = load_preds()
        _GROSS[group] = p if group == "full" else p.loc[p["ticker"].isin(LI_STOCKS)]
    return portfolio(_GROSS[group], m)[0]


def accuracy_by_fold():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    for ax, group in zip(axes, ["full", "li7"]):
        m = pd.read_csv(RESULTS / f"metrics_{group}.csv")
        m = m.loc[m["fold"] != "pooled"]
        m["fold"] = m["fold"].astype(int)
        models = [x for x in dict.fromkeys(m["model"]) if x in COLORS]
        off = np.linspace(-0.3, 0.3, len(models))
        for o, mod in zip(off, models):
            d = m.loc[m["model"] == mod]
            ax.scatter(d["fold"] + o, d["accuracy"] * 100, s=14, color=COLORS.get(mod, MUTED), label=mod,
                       edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(50, color=MUTED, linewidth=0.8)
        ax.set_xticks(sorted(m["fold"].unique()))
        ax.set_ylabel("out-of-sample accuracy (%)")
        ax.set_title("Full universe" if group == "full" else "Li-7 subset", fontsize=8, color=INK, loc="left")
    axes[0].legend(ncol=8, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 1.22))
    save(fig, "accuracy_by_fold")


def regime_timeline():
    r = pd.read_parquet(PROC / "regimes.parquet")
    r["date"] = r["date"].astype("datetime64[ns]")
    px = enforce_seal(pd.read_parquet(PROC / "prices.parquet"))
    spy = px.loc[px["yf_ticker"] == "SPY"].set_index("date")["close"]
    oos = pd.concat([r.loc[(r["fold"] == y) & (r["date"].dt.year == y)] for y in r["fold"].unique()]).set_index("date")
    last = r.loc[r["fold"] == r["fold"].max()].set_index("date")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 3.8), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    a1.plot(spy.index, spy.values, color=INK, linewidth=1.0)
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_yticks([100, 200, 300, 400])
    a1.set_ylabel("SPY (adj. close, log)")
    hi = oos["p_high"] > 0.5
    for ax in (a1, a2):
        ax.fill_between(oos.index, 0, 1, where=hi.values, transform=ax.get_xaxis_transform(),
                        color="#eb6834", alpha=0.12, linewidth=0)
    ins = last.loc[last.index < oos.index.min()]
    a2.plot(ins.index, ins["p_high"], color="#2a78d6", linewidth=0.8, label=f"{ins.index.min().year}-{ins.index.max().year % 100:02d}: filtered with the {int(last['fold'].max())}-fold fit (in-sample)")
    a2.plot(oos.index, oos["p_high"], color="#eb6834", linewidth=0.8, label=f"{oos.index.min().year}-{oos.index.max().year % 100:02d}: filtered, each test year with its own fold fit")
    a2.set_ylabel("P(high-vol regime)")
    a2.set_ylim(0, 1)
    a2.legend(fontsize=7, loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
    a1.set_title("SPY with out-of-sample P(high-vol) > 0.5 shaded", fontsize=8, color=INK, loc="left")
    save(fig, "regime_timeline")


def sentiment_coef_by_regime():
    d = pd.read_csv(RESULTS / "sentiment_coef_by_regime.csv")
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for o, (reg, col) in zip([-0.12, 0.12], [("low-vol", "#2a78d6"), ("high-vol", "#eb6834")]):
        q = d.loc[d["regime"] == reg]
        ax.errorbar(q["fold"] + o, q["coef"], yerr=1.96 * q["se"], fmt="o", ms=4, color=col, capsize=2,
                    elinewidth=1, label=reg)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(sorted(d["fold"].unique()))
    ax.set_xlabel("fold (test year; coefficients from its training fit)")
    ax.set_ylabel("M3 sentiment coefficient (per SD)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Sentiment effect by regime, 95% CI (date-clustered)", fontsize=8, color=INK, loc="left")
    save(fig, "sentiment_coef_by_regime")


if __name__ == "__main__":
    cumulative_long_short()
    accuracy_by_fold()
    regime_timeline()
    sentiment_coef_by_regime()
    print("figures DONE", flush=True)
