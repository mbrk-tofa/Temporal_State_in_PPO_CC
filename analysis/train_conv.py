"""train_conv.py
Generates training convergence curves for the Results section.

Layout
------
2×2 grid, two columns:
  Col 1: Stacking k=3  (top)   |  Stacking k=5  (bottom)
  Col 2: Stacking k=10 (top)   |  LSTM          (bottom)

Per subplot:
  - One bold EWMA line (mean across all seeds and workers)
  - One shaded ±1 SD band (across seeds)
  - No raw reward traces, no per-seed lines, no secondary axis

Axes:
  x = Training Episode
  y = Mean Episode Reward (EWMA-smoothed)

Output
------
  results/plots/convergence.png   (600 dpi)
  results/plots/convergence.pdf   (vector)
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
# TRAIN_LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "train")
# OUT_DIR       = os.path.join(SCRIPT_DIR, "results", "plots")
# os.makedirs(OUT_DIR, exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_LOG_DIR = str(REPO_ROOT / "logs" / "train")
OUT_DIR       = str(REPO_ROOT / "results" / "plots")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS  = [2, 7, 13, 18, 24]
N_ENVS = 4   # workers per seed

# Panel order: column-major — (0,0)=k3, (1,0)=k5, (0,1)=k10, (1,1)=lstm
MODEL_TYPES = ["stacking3HL", "stacking10HL", "stacking5HL", "lstm"]

MODEL_LABELS = {
    "stacking3HL":  "Stacking k=3",
    "stacking5HL":  "Stacking k=5",
    "stacking10HL": "Stacking k=10",
    "lstm":         "LSTM",
}

MODEL_COLORS = {
    "stacking3HL":  "#2ca02c",
    "stacking5HL":  "#d62728",
    "stacking10HL": "#ff7f0e",
    "lstm":         "#1f77b4",
}

EWMA_ALPHA = 0.05   # display smoothing applied to the grand-mean EWMA line
                    # (lower = smoother; the stored reward_ewma from the env
                    # already has 0.99/0.01 smoothing baked in)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_worker_log(model_type, seed_id):
    """Load one worker's compact training log. Returns list of dicts."""
    path = os.path.join(TRAIN_LOG_DIR, f"{model_type}_seed_{seed_id}.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_model_data(model_type):
    """Load and aggregate all workers across all seeds for one model type.

    Returns
    -------
    episodes   : 1-D int array, common x axis trimmed to shortest seed
    grand_ewma : 1-D float array, mean reward_ewma across all seeds/workers
    sd_band    : 1-D float array, ±1 SD across per-seed means (for shading)
    n_seeds    : int, number of seeds with valid data (for reporting)
    """
    seed_ewma_means = []   # one 1-D array per seed

    for seed in SEEDS:
        worker_ewmas = []
        for rank in range(N_ENVS):
            records = load_worker_log(model_type, seed + rank)
            if records:
                worker_ewmas.append([r["reward_ewma"] for r in records])

        if not worker_ewmas:
            continue

        min_len = min(len(w) for w in worker_ewmas)
        ewma_mat = np.array([w[:min_len] for w in worker_ewmas])  # (W, E)
        seed_ewma_means.append(ewma_mat.mean(axis=0))             # (E,)

    if not seed_ewma_means:
        return None, None, None, 0

    min_ep     = min(len(m) for m in seed_ewma_means)
    seed_mat   = np.array([m[:min_ep] for m in seed_ewma_means])  # (S, E)
    grand_ewma = seed_mat.mean(axis=0)
    sd_band    = seed_mat.std(axis=0)
    episodes   = np.arange(1, min_ep + 1)

    return episodes, grand_ewma, sd_band, len(seed_ewma_means)


def smooth(x, alpha=EWMA_ALPHA):
    """Exponential moving average for display smoothing only."""
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_convergence():
    # 2×2 grid, filled column-major: k3/k5 in col 0, k10/lstm in col 1
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 7),
        sharex=False,
        sharey=False,
    )
    # axes[row][col]: flatten to column-major order matching MODEL_TYPES
    panel_order = [axes[0][0], axes[1][0], axes[0][1], axes[1][1]]

    missing = []

    for ax, model_type in zip(panel_order, MODEL_TYPES):
        color = MODEL_COLORS[model_type]
        label = MODEL_LABELS[model_type]

        episodes, grand_ewma, sd_band, n_seeds = load_model_data(model_type)

        if episodes is None:
            ax.set_title(f"{label}  [no data]", fontsize=11, fontweight="bold")
            missing.append(model_type)
            continue

        s_ewma = smooth(grand_ewma)
        s_lo   = smooth(grand_ewma - sd_band)
        s_hi   = smooth(grand_ewma + sd_band)

        # Shaded ±1 SD band across seeds
        ax.fill_between(episodes, s_lo, s_hi,
                        color=color, alpha=0.18, linewidth=0)

        # Bold EWMA line
        ax.plot(episodes, s_ewma,
                color=color, linewidth=2.2,
                label=f"{label}  (n={n_seeds} seeds)")

        ax.set_title(label, fontsize=11, fontweight="bold", pad=5)
        ax.set_xlabel("Training Episode", fontsize=9)
        ax.set_ylabel("Mean Episode Reward (EWMA)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v/1000:.0f}k" if abs(v) >= 1000 else f"{v:.1f}"
        ))
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.4)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.85)

    fig.suptitle(
        "Learning Convergence of PPO Policies Across Training Episodes",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.01,
        "Bold line = mean EWMA across seeds and workers  |  "
        "Band = \u00b11 SD across seeds",
        ha="center", fontsize=8, color="grey",
    )

    fig.tight_layout()

    png_path = os.path.join(OUT_DIR, "convergence.png")
    pdf_path = os.path.join(OUT_DIR, "convergence.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path,           bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    if missing:
        print("\nWarning — no data found for:")
        for m in missing:
            print(f"  {m}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.isdir(TRAIN_LOG_DIR):
        raise FileNotFoundError(
            f"Training log directory not found: {TRAIN_LOG_DIR}\n"
            "Run Train5.py first, or check TRAIN_LOG_DIR."
        )
    plot_convergence()
