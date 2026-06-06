"""
training_curve.py — TrainingCurve

Plots mean reward per training episode across the full training run.
Each point on the x-axis is one training episode.
Each point on the y-axis is the mean reward across all 400 steps
within that episode, averaged across seeds.

This reveals:
    - How quickly each model converges
    - Where peak reward occurs (optimal stopping point)
    - Whether degradation is present and when it begins

Data source: JSONL logs from the default (training) scenario only.

Output
------
results/plots/reward_per_training_episode.png
results/csv/reward_per_training_episode.csv
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TrainingCurve:
    """Plot mean reward per training episode for each model.

    Parameters
    ----------
    log_dir   : str  Directory containing JSONL episode logs.
                     Default: "logs".
    csv_dir   : str  Output directory for CSV. Default: "results/csv".
    plots_dir : str  Output directory for PNG. Default: "results/plots".
    scenario  : str  Training scenario name. Default: "default".
    smoothing : int  Rolling average window over episodes. Default: 5.
                     Set to 1 to disable smoothing.
    """

    MODEL_COLOURS = {
        "lstm":         "#1f77b4",
        "stacking10HL": "#ff7f0e",
        "stacking3HL":  "#2ca02c",
        "stacking5HL":  "#d62728",
    }

    def __init__(
        self,
        log_dir:   str = "logs",
        csv_dir:   str = "results/csv",
        plots_dir: str = "results/plots",
        scenario:  str = "default",
        smoothing: int = 5,
    ):
        self.log_dir   = log_dir
        self.csv_dir   = csv_dir
        self.plots_dir = plots_dir
        self.scenario  = scenario
        self.smoothing = smoothing
        os.makedirs(csv_dir,   exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        self.episode_df = None

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self):
        """Load training logs, build episode DataFrame, plot and save."""
        self._load()
        self._plot()
        print("[TrainingCurve] Done.")
        return self

    def get_episode_df(self) -> pd.DataFrame:
        return self.episode_df

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    def _load(self):
        """Read all JSONL files for the training scenario.

        Each line in a JSONL file is one episode object:
            {"Scenario": "default", "Seed": 1, "Episode": 1, "Steps": [...]}

        We compute mean reward per episode per seed, then average
        across seeds per episode number.
        """
        pattern = os.path.join(self.log_dir, f"*_{self.scenario}_seed_*.jsonl")
        files   = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(
                f"No JSONL files found matching pattern: {pattern}\n"
                f"Check that log_dir='{self.log_dir}' is correct and "
                f"that the experiment has been run."
            )

        rows = []
        for filepath in files:
            # extract model name from filename: {model}_{scenario}_seed_{seed}.jsonl
            basename = os.path.basename(filepath).replace(".jsonl", "")
            parts    = basename.split("_")
            model    = parts[0]

            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ep_obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # only process the training scenario
                    if ep_obj.get("Scenario", "") != self.scenario:
                        continue

                    steps = ep_obj.get("Steps", [])
                    if not steps:
                        continue

                    ep_reward = np.mean([s["Reward"] for s in steps])
                    rows.append({
                        "model":         model,
                        "seed":          int(ep_obj.get("Seed", 0)),
                        "episode":       int(ep_obj.get("Episode", 0)),
                        "mean_reward":   ep_reward,
                        "num_steps":     len(steps),
                    })

        if not rows:
            raise ValueError(
                f"No episode data found for scenario='{self.scenario}'. "
                "Check that training logs exist and contain Steps data."
            )

        raw_df = pd.DataFrame(rows)
        print(f"[TrainingCurve] Loaded {len(raw_df)} episode records "
              f"from {len(files)} files "
              f"({raw_df['model'].nunique()} models, "
              f"{raw_df['seed'].nunique()} seeds, "
              f"up to {raw_df['episode'].max()} episodes)")

        # average across seeds per (model, episode)
        self.episode_df = (
            raw_df
            .groupby(["model", "episode"])
            .agg(
                mean_reward = ("mean_reward", "mean"),
                std_reward  = ("mean_reward", "std"),
                n_seeds     = ("seed",        "count"),
            )
            .reset_index()
        )

        # save CSV
        path = os.path.join(self.csv_dir, "reward_per_training_episode.csv")
        self.episode_df.to_csv(path, index=False)
        print(f"[TrainingCurve] CSV written: {path}")

    # ------------------------------------------------------------------ #
    # Plotting                                                             #
    # ------------------------------------------------------------------ #

    def _smooth(self, series: pd.Series) -> pd.Series:
        """Apply rolling mean smoothing."""
        if self.smoothing <= 1:
            return series
        return series.rolling(window=self.smoothing, min_periods=1, center=True).mean()

    def _plot(self):
        fig, ax = plt.subplots(figsize=(12, 5))

        models = sorted(self.episode_df["model"].unique())

        for model in models:
            sub  = self.episode_df[self.episode_df["model"] == model].sort_values("episode")
            eps  = sub["episode"].values
            mean = self._smooth(sub["mean_reward"]).values
            std  = sub["std_reward"].fillna(0).values
            c    = self.MODEL_COLOURS.get(model, "#888888")

            ax.plot(eps, mean, label=model, color=c, linewidth=1.5)
            ax.fill_between(
                eps,
                mean - std,
                mean + std,
                alpha=0.15,
                color=c,
            )

        # mark approximate peak per model
        for model in models:
            sub      = self.episode_df[self.episode_df["model"] == model]
            smoothed = self._smooth(sub.sort_values("episode")["mean_reward"])
            peak_idx = smoothed.idxmax()
            peak_ep  = sub.loc[peak_idx, "episode"]
            peak_val = smoothed.loc[peak_idx]
            c        = self.MODEL_COLOURS.get(model, "#888888")
            ax.axvline(x=peak_ep, color=c, linestyle="--", alpha=0.4, linewidth=0.8)
            ax.annotate(
                f"{model}\npeak ep {peak_ep}",
                xy=(peak_ep, peak_val),
                xytext=(peak_ep + 2, peak_val + 0.02),
                fontsize=7,
                color=c,
                alpha=0.8,
            )

        ax.set_xlabel("Training Episode")
        ax.set_ylabel("Mean Reward per Episode")
        ax.set_title(
            f"Reward per Training Episode — {self.scenario} scenario\n"
            f"(mean ± std across seeds, smoothing window={self.smoothing})"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.4)

        path = os.path.join(self.plots_dir, "reward_per_training_episode.png")
        plt.tight_layout()
        plt.savefig(path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"[TrainingCurve] Plot saved: {path}")

        # also print peak episode summary to console
        print("\n[TrainingCurve] Peak reward by model:")
        print(f"  {'Model':<16} {'Peak episode':>14} {'Peak reward':>12}")
        print(f"  {'-'*44}")
        for model in models:
            sub      = self.episode_df[self.episode_df["model"] == model]
            smoothed = self._smooth(sub.sort_values("episode")["mean_reward"])
            peak_idx = smoothed.idxmax()
            peak_ep  = sub.loc[peak_idx, "episode"]
            peak_val = smoothed.loc[peak_idx]
            print(f"  {model:<16} {peak_ep:>14} {peak_val:>12.4f}")
        print()
