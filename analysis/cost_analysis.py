"""
cost_analysis.py — CostAnalyser

Loads inference timing files and training time records,
computes summary statistics, and generates all cost plots.

Outputs
-------
results/csv/inference_cost_per_episode.csv
results/csv/inference_cost_summary.csv
results/plots/inference_cost_bar.png
results/plots/inference_time_per_episode.png
"""

import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CostAnalyser:
    """Load and analyse inference and training cost data.

    Parameters
    ----------
    cost_dir  : str  Directory containing cost_*.json files.
                     Default: "logs/cost_files".
    csv_dir   : str  Output directory for CSVs. Default: "results/csv".
    plots_dir : str  Output directory for PNGs. Default: "results/plots".
    """

    MODEL_COLOURS = {
        "lstm":         "#1f77b4",
        "stacking10HL": "#ff7f0e",
        "stacking3HL":  "#2ca02c",
        "stacking5HL":  "#d62728",
    }

    def __init__(
        self,
        cost_dir:  str = "logs/cost_files",
        csv_dir:   str = "results/csv",
        plots_dir: str = "results/plots",
    ):
        self.cost_dir  = cost_dir
        self.csv_dir   = csv_dir
        self.plots_dir = plots_dir
        os.makedirs(csv_dir,   exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        self.episode_df  = None   # per episode
        self.seed_df     = None   # per seed
        self.summary_df  = None   # across seeds

        self._models    = []
        self._scenarios = []

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self):
        """Load data, write CSVs, generate all plots."""
        self._load()
        self._aggregate()
        self._write_csvs()
        self._plot_bar()
        self._plot_per_episode()
        print(f"[CostAnalyser] All cost outputs saved.")
        return self

    def get_summary(self) -> pd.DataFrame:
        return self.summary_df

    # ------------------------------------------------------------------ #
    # Loading                                                             #
    # ------------------------------------------------------------------ #

    # -------------------------------------------------------------- #
    # Current file format (cost_{model}_{scenario}_seed{N}.json):    #
    #   A bare JSON list of episode dicts, each with:                #
    #     "episode"     : int                                         #
    #     "mean_ms"     : float  (pre-computed mean inference time)   #
    #     "total_steps" : int                                         #
    #   model/scenario/seed are parsed from the filename.            #
    #   No per-step timings are stored, so CDF plots are unavailable. #
    # -------------------------------------------------------------- #

    @staticmethod
    def _parse_filename(filepath):
        # cost_{model}_{scenario}_seed{N}.json
        name  = os.path.basename(filepath).replace(".json", "")
        parts = name.split("_")
        try:
            model    = parts[1]
            scenario = parts[2]
            seed     = parts[3].replace("seed", "")
        except IndexError:
            raise ValueError(
                f"Cannot parse model/scenario/seed from filename: {filepath}\n"
                f"Expected 'cost_MODEL_SCENARIO_seedN.json'."
            )
        return model, scenario, seed

    def _load(self):
        pattern = os.path.join(self.cost_dir, "cost_*.json")
        files   = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(
                f"No cost files found in '{self.cost_dir}'. "
                "Run the experiment first."
            )

        rows = []
        for filepath in sorted(files):
            with open(filepath) as f:
                episodes = json.load(f)   # bare list

            if not episodes:
                print(f"  [CostAnalyser] Skipping empty file: {filepath}")
                continue

            model, scenario, seed = self._parse_filename(filepath)
            # key = (model, scenario)
            # self.step_ms.setdefault(key, [])

            for ep in episodes:
                rows.append({
                    "model":       model,
                    "scenario":    scenario,
                    "seed":        seed,
                    "episode":     int(ep["episode"]),
                    "mean_ms":     float(ep["mean_ms"]),
                    "total_steps": int(ep["total_steps"]),
                })

        self.episode_df = pd.DataFrame(rows)
        self._models    = sorted(self.episode_df["model"].unique())
        self._scenarios = sorted(self.episode_df["scenario"].unique())
        print(f"[CostAnalyser] Loaded {len(files)} files — "
              f"{len(self.episode_df)} episode records")

    # ------------------------------------------------------------------ #
    # Aggregation                                                         #
    # ------------------------------------------------------------------ #

    def _aggregate(self):
        # per-seed: average mean_ms across episodes
        self.seed_df = (
            self.episode_df
            .groupby(["model", "scenario", "seed"])
            .agg(
                mean_ms    = ("mean_ms", "mean"),
                std_ms     = ("mean_ms", "std"),
                n_episodes = ("episode", "count"),
            )
            .reset_index()
        )

        # across-seed: average seed-level means
        self.summary_df = (
            self.seed_df
            .groupby(["model", "scenario"])
            .agg(
                mean_ms     = ("mean_ms", "mean"),
                mean_ms_std = ("mean_ms", "std"),
                n_seeds     = ("seed",    "count"),
            )
            .reset_index()
        )

    # ------------------------------------------------------------------ #
    # CSV output                                                          #
    # ------------------------------------------------------------------ #

    def _write_csvs(self):
        ep_path = os.path.join(self.csv_dir, "inference_cost_per_episode.csv")
        sm_path = os.path.join(self.csv_dir, "inference_cost_summary.csv")
        self.episode_df.to_csv(ep_path, index=False)
        self.summary_df.to_csv(sm_path, index=False)
        print(f"  [CostAnalyser] CSVs written:")
        print(f"    {ep_path}")
        print(f"    {sm_path}")
        print(self.summary_df.to_string())

    # ------------------------------------------------------------------ #
    # Plots                                                               #
    # ------------------------------------------------------------------ #

    def _colour(self, model):
        return self.MODEL_COLOURS.get(model, "#888888")

    def _save(self, filename):
        path = os.path.join(self.plots_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  [CostAnalyser] Saved: {path}")

    def _plot_bar(self):
        """Mean inference time bar chart, one subplot per scenario."""
        fig, axes = plt.subplots(
            len(self._scenarios), 1,
            figsize=(max(6, len(self._models) * 2), 4 * len(self._scenarios)),
            sharey=False,
        )
        if len(self._scenarios) == 1:
            axes = [axes]

        x = np.arange(len(self._models))
        for ax, scenario in zip(axes, self._scenarios):
            s = self.summary_df[self.summary_df["scenario"] == scenario]
            means = [
                s[s["model"] == m]["mean_ms"].values[0]
                if m in s["model"].values else 0
                for m in self._models
            ]
            stds = [
                s[s["model"] == m]["mean_ms_std"].values[0]
                if m in s["model"].values else 0
                for m in self._models
            ]
            colours = [self._colour(m) for m in self._models]
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.75,
                   width=0.5, color=colours)
            ax.set_xticks(x)
            ax.set_xticklabels(self._models)
            ax.set_ylabel("Mean Inference Time (ms)")
            ax.set_title(f"Inference Cost — {scenario}")
            ax.grid(axis="y", alpha=0.4)

        self._save("inference_cost_bar.png")

    def _plot_per_episode(self):
        """Mean inference time across evaluation episodes per model."""
        fig, axes = plt.subplots(
            len(self._scenarios), 1,
            figsize=(9, 4 * len(self._scenarios)),
            sharey=False,
        )
        if len(self._scenarios) == 1:
            axes = [axes]

        for ax, scenario in zip(axes, self._scenarios):
            sub = self.episode_df[self.episode_df["scenario"] == scenario]
            for model in self._models:
                m_sub = sub[sub["model"] == model]
                if m_sub.empty:
                    continue
                ep_avg = (
                    m_sub.groupby("episode")["mean_ms"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                c = self._colour(model)
                ax.plot(ep_avg["episode"], ep_avg["mean"],
                        label=model, marker="o", markersize=3, color=c)
                ax.fill_between(
                    ep_avg["episode"],
                    ep_avg["mean"] - ep_avg["std"],
                    ep_avg["mean"] + ep_avg["std"],
                    alpha=0.2, color=c,
                )
            ax.set_title(scenario)
            ax.set_ylabel("Mean Inference Time (ms)")
            ax.grid(alpha=0.4)
            ax.legend()

        axes[-1].set_xlabel("Eval Episode")
        fig.suptitle("Inference Time per Eval Episode")
        self._save("inference_time_per_episode.png")