"""plot_inference_stats.py

Generates inference time CDF plots and summary statistics from the per-step
cost files written by run_eval_only.py.

Inputs
------
  logs/cost_files/cost_<model_type>_<scenario>_seed<seed>.json
      Each file contains a flat list of {"episode", "step", "inference_ms"}
      records — one per model.predict() call during evaluation.

Outputs
-------
  logs/inference/cdf_<scenario>.png/pdf
      CDF of per-step inference time, one curve per model, per scenario.

  logs/inference/boxplot_all.png/pdf
      Box plots of inference time across models and scenarios.

  logs/inference/seed_variance.png/pdf
      Per-seed mean inference time — shows cross-seed reproducibility.

  logs/inference/step_drift_<scenario>.png/pdf
      Mean inference time vs step index within episode — detects LSTM
      hidden-state accumulation effects on latency.

  logs/inference/summary_statistics.csv
      Full summary table: mean, median, std, p95, p99, max per group.

  logs/inference/summary_statistics.txt
      Human-readable version of the same table.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COST_DIR   = "logs/cost_files" #os.path.join(SCRIPT_DIR, "logs")
OUT_DIR    = "results/plots" #os.path.join(, "inference")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [2, 7, 13, 18, 24]

MODEL_TYPES = ["stacking3HL", "stacking5HL", "stacking10HL", "lstm"]

SCENARIOS = ["crossrtt", "flat", "step"]

MODEL_LABELS = {
    "stacking3HL":  "Stacking k=3",
    "stacking5HL":  "Stacking k=5",
    "stacking10HL": "Stacking k=10",
    "lstm":         "LSTM",
}

MODEL_COLORS = {
        "lstm":         "#1f77b4",
        "stacking10HL": "#ff7f0e",
        "stacking3HL":  "#2ca02c",
        "stacking5HL":  "#d62728",
        }

SCENARIO_LABELS = {
    "crossrtt": "Cross-RTT",
    "flat":     "Flat",
    "step":     "Step Change",
}

DPI = 600


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_cost_file(model_type, seed, scenario):
    """Load one cost JSON file. Returns list of dicts or None if missing."""
    path = os.path.join(
        COST_DIR, f"cost_{model_type}_{scenario}_seed{seed}.json"
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_dataframe():
    """Load all cost files into a single tidy DataFrame.

    Columns: model_type, seed, scenario, episode, step, inference_ms
    Missing files are reported and skipped.
    """
    rows = []
    missing = []

    for model_type in MODEL_TYPES:
        for seed in SEEDS:
            for scenario in SCENARIOS:
                records = load_cost_file(model_type, seed, scenario)
                if records is None:
                    missing.append(f"{model_type}_{scenario}_seed{seed}")
                    continue
                for r in records:
                    rows.append({
                        "model_type":   model_type,
                        "seed":         seed,
                        "scenario":     scenario,
                        "episode":      r["episode"],
                        "step":         r["step"],
                        "inference_ms": r["inference_ms"],
                    })

    if missing:
        print(f"Warning — {len(missing)} missing cost files:")
        for m in missing:
            print(f"  {m}")

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df):,} step records from "
          f"{len(df.groupby(['model_type','scenario','seed'])):,} files.\n")
    return df


# ─────────────────────────────────────────────
# PLOT 1 — CDF per scenario
# ─────────────────────────────────────────────

def plot_cdfs(df):
    """One figure per scenario. Each figure has one CDF curve per model,
    pooling all steps across all seeds and episodes.
    Shaded band = ±1 std of per-seed CDFs (shows cross-seed stability).
    """
    for scenario in SCENARIOS:
        fig, ax = plt.subplots(figsize=(8, 5))

        for model_type in MODEL_TYPES:
            color = MODEL_COLORS[model_type]
            label = MODEL_LABELS[model_type]

            # Pool all steps across all seeds for the grand CDF
            all_times = df[
                (df["model_type"] == model_type) &
                (df["scenario"]   == scenario)
            ]["inference_ms"].values

            if len(all_times) == 0:
                continue

            # Grand CDF
            sorted_t = np.sort(all_times)
            cdf      = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
            ax.plot(sorted_t, cdf,
                    color=color, linewidth=2.0,
                    label=f"{label}  (n={len(all_times):,})")

            # Per-seed CDFs for the shaded variance band
            # Interpolate each seed's CDF onto a common x grid
            x_grid = np.linspace(sorted_t.min(), np.percentile(sorted_t, 99.5), 500)
            seed_cdfs = []
            for seed in SEEDS:
                seed_t = df[
                    (df["model_type"] == model_type) &
                    (df["scenario"]   == scenario) &
                    (df["seed"]       == seed)
                ]["inference_ms"].values
                if len(seed_t) == 0:
                    continue
                s_sorted = np.sort(seed_t)
                s_cdf    = np.arange(1, len(s_sorted) + 1) / len(s_sorted)
                # Interpolate onto common grid (fill 1.0 beyond max)
                interp = np.interp(x_grid, s_sorted, s_cdf, right=1.0)
                seed_cdfs.append(interp)

            if len(seed_cdfs) > 1:
                seed_mat = np.array(seed_cdfs)
                ax.fill_between(
                    x_grid,
                    seed_mat.min(axis=0),
                    seed_mat.max(axis=0),
                    color=color, alpha=0.12,
                )

        ax.set_xlabel("Inference Time (ms)", fontsize=11)
        ax.set_ylabel("CDF", fontsize=11)
        ax.set_title(f"Inference Time CDF — {SCENARIO_LABELS[scenario]}",
                     fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, _: f"{y:.0%}"
        ))
        ax.grid(axis="both", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

        # Mark p95 and p99 reference lines
        ax.axhline(0.95, color="grey", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.axhline(0.99, color="grey", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.text(ax.get_xlim()[1], 0.95, " p95", va="center",
                fontsize=7, color="grey")
        ax.text(ax.get_xlim()[1], 0.99, " p99", va="center",
                fontsize=7, color="grey")

        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(OUT_DIR, f"cdf_{scenario}.{ext}")
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: cdf_{scenario}.png/pdf")


# ─────────────────────────────────────────────
# PLOT 2 — Box plots across models and scenarios
# ─────────────────────────────────────────────

def plot_boxplots(df):
    """One subplot per scenario. Box = IQR, whiskers = 1.5×IQR, fliers hidden.
    One box per model. Overlaid strip of per-seed means shows seed variance.
    """
    fig, axes = plt.subplots(1, len(SCENARIOS),
                             figsize=(5 * len(SCENARIOS), 5),
                             sharey=False)

    for ax, scenario in zip(axes, SCENARIOS):
        data_by_model = []
        labels        = []
        colors        = []

        for model_type in MODEL_TYPES:
            times = df[
                (df["model_type"] == model_type) &
                (df["scenario"]   == scenario)
            ]["inference_ms"].values
            if len(times) == 0:
                continue
            data_by_model.append(times)
            labels.append(MODEL_LABELS[model_type])
            colors.append(MODEL_COLORS[model_type])

        bp = ax.boxplot(
            data_by_model,
            patch_artist=True,
            showfliers=False,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Per-seed mean dots overlaid on each box
        for pos, (model_type, color) in enumerate(
                zip(MODEL_TYPES, colors), start=1):
            seed_means = []
            for seed in SEEDS:
                t = df[
                    (df["model_type"] == model_type) &
                    (df["scenario"]   == scenario) &
                    (df["seed"]       == seed)
                ]["inference_ms"].values
                if len(t) > 0:
                    seed_means.append(t.mean())
            if seed_means:
                jitter = np.random.default_rng(42).uniform(
                    -0.12, 0.12, len(seed_means)
                )
                ax.scatter(
                    pos + jitter, seed_means,
                    color=color, edgecolors="black",
                    linewidths=0.5, s=30, zorder=3, alpha=0.9,
                )

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=10, fontweight="bold")
        ax.set_ylabel("Inference Time (ms)" if ax == axes[0] else "",
                      fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.suptitle("Inference Time Distribution by Model and Scenario",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"boxplot_all.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved: boxplot_all.png/pdf")


# ─────────────────────────────────────────────
# PLOT 3 — Seed variance
# ─────────────────────────────────────────────

def plot_seed_variance(df):
    """Per-seed mean inference time, grouped by model and scenario.
    Shows whether inference cost is reproducible across seeds.
    Low variance here means the reported mean is reliable.
    """
    fig, axes = plt.subplots(1, len(SCENARIOS),
                             figsize=(5 * len(SCENARIOS), 4),
                             sharey=False)

    for ax, scenario in zip(axes, SCENARIOS):
        x_positions = np.arange(len(MODEL_TYPES))
        width       = 0.15
        offsets     = np.linspace(-width * 2, width * 2, len(SEEDS))

        for i, seed in enumerate(SEEDS):
            means = []
            for model_type in MODEL_TYPES:
                t = df[
                    (df["model_type"] == model_type) &
                    (df["scenario"]   == scenario) &
                    (df["seed"]       == seed)
                ]["inference_ms"].values
                means.append(t.mean() if len(t) > 0 else np.nan)

            ax.bar(
                x_positions + offsets[i], means,
                width=width, label=f"seed {seed}",
                alpha=0.8,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODEL_TYPES],
            rotation=20, ha="right", fontsize=8
        )
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Inference Time (ms)" if ax == axes[0] else "",
                      fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        if ax == axes[-1]:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Per-Seed Mean Inference Time (Cross-Seed Reproducibility)",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"seed_variance.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved: seed_variance.png/pdf")


# ─────────────────────────────────────────────
# PLOT 4 — Step drift within episode
# ─────────────────────────────────────────────

def plot_step_drift(df):
    """Mean inference time vs step index within episode, per model.
    Detects whether LSTM hidden-state accumulation causes latency to grow
    as the episode progresses. Stacking models should be flat; LSTM may drift.
    One subplot per scenario.
    """
    fig, axes = plt.subplots(1, len(SCENARIOS),
                             figsize=(5 * len(SCENARIOS), 5),
                             sharex=False)

    for ax, scenario in zip(axes, SCENARIOS):
        for model_type in MODEL_TYPES:
            color = MODEL_COLORS[model_type]
            label = MODEL_LABELS[model_type]

            sub = df[
                (df["model_type"] == model_type) &
                (df["scenario"]   == scenario)
            ]
            if sub.empty:
                continue

            # Mean and std of inference_ms at each step index,
            # pooled across all seeds and episodes
            step_stats = (
                sub.groupby("step")["inference_ms"]
                   .agg(["mean", "std"])
                   .reset_index()
            )

            ax.plot(
                step_stats["step"], step_stats["mean"],
                color=color, linewidth=1.8, label=label,
            )
            ax.fill_between(
                step_stats["step"],
                step_stats["mean"] - step_stats["std"],
                step_stats["mean"] + step_stats["std"],
                color=color, alpha=0.12,
            )

        ax.set_title(f"Step-Level Inference Drift — {SCENARIO_LABELS[scenario]}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Step Index within Episode", fontsize=9)
        ax.set_ylabel("Mean Inference Time (ms)", fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        ax.grid(axis="both", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"step_drift.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved: step_drift.png/pdf")


# ─────────────────────────────────────────────
# STATISTICS TABLE
# ─────────────────────────────────────────────

def compute_summary_statistics(df):
    """Compute and save a full summary statistics table.

    Rows: (model_type, scenario)
    Columns: n, mean, median, std, p25, p75, p95, p99, max, cv

    Also runs a Kruskal-Wallis test per scenario (non-parametric equivalent
    of one-way ANOVA) to test whether inference times differ significantly
    across model types. Reports H-statistic and p-value.
    """
    rows = []

    for scenario in SCENARIOS:
        for model_type in MODEL_TYPES:
            times = df[
                (df["model_type"] == model_type) &
                (df["scenario"]   == scenario)
            ]["inference_ms"].values

            if len(times) == 0:
                continue

            rows.append({
                "model":    MODEL_LABELS[model_type],
                "scenario": SCENARIO_LABELS[scenario],
                "n":        len(times),
                "mean_ms":  round(np.mean(times),            6),
                "median_ms":round(np.median(times),          6),
                "std_ms":   round(np.std(times),             6),
                "p25_ms":   round(np.percentile(times, 25),  6),
                "p75_ms":   round(np.percentile(times, 75),  6),
                "p95_ms":   round(np.percentile(times, 95),  6),
                "p99_ms":   round(np.percentile(times, 99),  6),
                "max_ms":   round(np.max(times),             6),
                # Coefficient of variation: std/mean — dimensionless stability metric
                "cv":       round(np.std(times) / np.mean(times), 4),
            })

    summary = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "summary_statistics.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved: summary_statistics.csv")

    # Save human-readable text table
    txt_path = os.path.join(OUT_DIR, "summary_statistics.txt")
    with open(txt_path, "w") as f:
        f.write("INFERENCE TIME SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        for scenario in SCENARIOS:
            f.write(f"Scenario: {SCENARIO_LABELS[scenario]}\n")
            f.write("-" * 80 + "\n")
            sub = summary[summary["scenario"] == SCENARIO_LABELS[scenario]]
            f.write(sub.drop(columns="scenario").to_string(index=False))
            f.write("\n\n")

            # Kruskal-Wallis test across model types for this scenario
            groups = [
                df[
                    (df["model_type"] == m) &
                    (df["scenario"]   == scenario)
                ]["inference_ms"].values
                for m in MODEL_TYPES
                if len(df[
                    (df["model_type"] == m) &
                    (df["scenario"]   == scenario)
                ]) > 0
            ]
            if len(groups) >= 2:
                h_stat, p_val = stats.kruskal(*groups)
                sig = "YES" if p_val < 0.05 else "NO"
                f.write(
                    f"Kruskal-Wallis test (inference time differs across models?):\n"
                    f"  H = {h_stat:.4f}   p = {p_val:.6f}   "
                    f"Significant at alpha=0.05: {sig}\n\n"
                )

    print(f"Saved: summary_statistics.txt")

    # Print condensed version to console
    print("\n" + "=" * 80)
    print("INFERENCE TIME SUMMARY (mean ± std, p99)  [ms]")
    print("=" * 80)
    for scenario in SCENARIOS:
        print(f"\n{SCENARIO_LABELS[scenario]}:")
        sub = summary[summary["scenario"] == SCENARIO_LABELS[scenario]]
        for _, row in sub.iterrows():
            print(f"  {row['model']:15s}  "
                  f"mean={row['mean_ms']:.4f}  "
                  f"std={row['std_ms']:.4f}  "
                  f"p99={row['p99_ms']:.4f}  "
                  f"cv={row['cv']:.4f}")

    return summary


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    if not os.path.isdir(COST_DIR):
        raise FileNotFoundError(
            f"Cost files directory not found: {COST_DIR}\n"
            "Run run_eval_only.py first."
        )

    print(f"Loading cost files from: {COST_DIR}\n")
    df = build_dataframe()

    if df.empty:
        print("No data loaded — check that cost files exist and are non-empty.")
        raise SystemExit(1)

    print("Generating plots and statistics...\n")
    plot_cdfs(df)
    plot_boxplots(df)
    plot_seed_variance(df)
    plot_step_drift(df)
    summary = compute_summary_statistics(df)

    print(f"\nAll outputs written to: {OUT_DIR}/")