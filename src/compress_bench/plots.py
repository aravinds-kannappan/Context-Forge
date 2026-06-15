from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_pareto(results_path: str, out_dir: str) -> list:
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    frame = pd.DataFrame(results.get("pareto", []))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        return []

    paths = []
    for (task, model), group in frame.groupby(["task", "model"]):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        scatter = ax.scatter(
            group["tokens_saved_pct"],
            group["quality_retained"],
            s=40 + group["latency_ms"].clip(0, 2000) / 20,
            c=group["latency_ms"],
            cmap="viridis_r",
            alpha=0.85,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["strategy"].replace("_", " "),
                (row["tokens_saved_pct"], row["quality_retained"]),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
        ax.set_title(f"{task} / {model}")
        ax.set_xlabel("Tokens saved (%)")
        ax.set_ylabel("Quality retained")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Latency (ms)")
        fig.tight_layout()
        path = out / f"pareto_{task}_{model.replace('/', '_')}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths
