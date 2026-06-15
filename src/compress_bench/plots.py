from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Palette aligned with the web dashboard.
STRAT_COLORS = {
    "llmlingua": "#2dd4bf",
    "embedding_chunk_drop": "#818cf8",
    "kv_cache_eviction": "#f472b6",
    "hard_prompt_pruning": "#f59e0b",
}
BG = "#0a0e16"
PANEL = "#121a28"
INK = "#eef2f8"
MUTED = "#93a1b5"
GRID = "#233044"


def _style(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, alpha=0.4, linewidth=0.6)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def _scatter(ax, group):
    for strat, rows in group.groupby("strategy"):
        color = STRAT_COLORS.get(strat, "#93a1b5")
        ax.scatter(
            rows["tokens_saved_pct"],
            rows["quality_retained"],
            s=40 + rows["latency_ms"].clip(0, 400) * 1.5,
            color=color,
            alpha=0.7,
            edgecolors="none",
            label=strat.replace("_", " "),
            zorder=2,
        )
        front = rows[rows.get("on_frontier", False) == True]  # noqa: E712
        if not front.empty:
            ax.scatter(
                front["tokens_saved_pct"],
                front["quality_retained"],
                s=40 + front["latency_ms"].clip(0, 400) * 1.5,
                facecolors="none",
                edgecolors=color,
                linewidths=1.6,
                zorder=3,
            )
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.03, 1.05)


def plot_pareto(results_path: str, out_dir: str) -> list:
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    points = results.get("aggregates") or results.get("pareto", [])
    frame = pd.DataFrame(points)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        return []
    if "on_frontier" not in frame.columns:
        frame["on_frontier"] = True

    paths = []

    # Per (task, model) detail plots.
    for (task, model), group in frame.groupby(["task", "model"]):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _style(ax, fig)
        _scatter(ax, group)
        ax.set_title(f"{task}  ·  {model}")
        ax.set_xlabel("Tokens saved (%)")
        ax.set_ylabel("Quality retained")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=8, loc="lower left")
        fig.tight_layout()
        path = out / f"pareto_{task}_{model.replace('/', '_')}.png"
        fig.savefig(path, dpi=160, facecolor=BG)
        plt.close(fig)
        paths.append(str(path))

    # Overview: one subplot per task for the primary tokenizer.
    primary = "gpt-4o" if "gpt-4o" in set(frame["model"]) else sorted(frame["model"])[0]
    tasks = sorted(frame["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.4 * len(tasks), 4.3), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        _style(ax, fig)
        _scatter(ax, frame[(frame["task"] == task) & (frame["model"] == primary)])
        ax.set_title(task)
        ax.set_xlabel("Tokens saved (%)")
    axes[0][0].set_ylabel("Quality retained")
    axes[0][-1].legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=8, loc="lower left")
    fig.suptitle(f"Context Forge — Pareto frontier ({primary})", color=INK, fontsize=14, y=1.02)
    fig.tight_layout()
    overview = out / "overview.png"
    fig.savefig(overview, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(overview))

    # Strategy comparison bar chart.
    by_strategy = results.get("by_strategy")
    if by_strategy:
        bs = pd.DataFrame(by_strategy).sort_values("quality_retained")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        _style(ax, fig)
        y = range(len(bs))
        colors = [STRAT_COLORS.get(s, "#93a1b5") for s in bs["strategy"]]
        ax.barh([i + 0.2 for i in y], bs["quality_retained"] * 100, height=0.38, color=colors, label="Quality retained (%)")
        ax.barh([i - 0.2 for i in y], bs["tokens_saved_pct"], height=0.38, color=colors, alpha=0.45, label="Tokens saved (%)")
        ax.set_yticks(list(y))
        ax.set_yticklabels([s.replace("_", " ") for s in bs["strategy"]], color=INK)
        ax.set_xlabel("Percent")
        ax.set_title("Strategy comparison (mean across tasks · tokenizers · ratios)")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=8, loc="lower right")
        fig.tight_layout()
        comp = out / "strategy_comparison.png"
        fig.savefig(comp, dpi=160, facecolor=BG)
        plt.close(fig)
        paths.append(str(comp))

    return paths
