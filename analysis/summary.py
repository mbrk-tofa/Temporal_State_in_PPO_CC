"""
summary.py — SummaryStats

Three-level aggregation of episode log data:
    Level 1 — per-episode rows   (model, scenario, seed, episode)
    Level 2 — per-seed means     (model, scenario, seed)
    Level 3 — across-seed means  (model, scenario)

All CSVs are written to results/csv/.
"""

import os
import pandas as pd


class SummaryStats:
    """Compute and save three-level aggregation tables.

    Parameters
    ----------
    episode_df : pd.DataFrame
        Episode-level DataFrame from DataLoader.get_episodes().
    csv_dir : str
        Output directory for CSV files. Default: "results/csv".
    """

    def __init__(self, episode_df: pd.DataFrame, csv_dir: str = "results/csv"):
        self.episode_df = episode_df
        self.csv_dir    = csv_dir
        os.makedirs(csv_dir, exist_ok=True)

        self.per_episode_df   = None
        self.per_seed_df      = None
        self.across_seed_df   = None

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self):
        """Compute all three levels and write CSVs."""
        self._level1_per_episode()
        self._level2_per_seed()
        self._level3_across_seed()
        print("[SummaryStats] All aggregation CSVs written.")
        return self

    def get_per_episode(self) -> pd.DataFrame:
        return self.per_episode_df

    def get_per_seed(self) -> pd.DataFrame:
        return self.per_seed_df

    def get_across_seed(self) -> pd.DataFrame:
        return self.across_seed_df

    # ------------------------------------------------------------------ #
    # Level 1 — per episode                                               #
    # ------------------------------------------------------------------ #

    def _level1_per_episode(self):
        """Save episode-level DataFrame directly — already built by DataLoader."""
        self.per_episode_df = self.episode_df.copy()
        path = os.path.join(self.csv_dir, "per_episode_results.csv")
        self.per_episode_df.to_csv(path, index=False)
        print(f"  [SummaryStats] Level 1 written: {path}  "
              f"({len(self.per_episode_df)} rows)")

    # ------------------------------------------------------------------ #
    # Level 2 — per seed                                                   #
    # ------------------------------------------------------------------ #

    def _level2_per_seed(self):
        """Average episode-level metrics within each (model, scenario, seed)."""
        self.per_seed_df = (
            self.episode_df
            .groupby(["model", "scenario", "seed"])
            .agg(
                num_episodes          = ("episode",        "count"),
                throughput_mean       = ("throughput_mean", "mean"),
                throughput_std_ep     = ("throughput_mean", "std"),
                latency_mean          = ("latency_mean",   "mean"),
                latency_std_ep        = ("latency_mean",   "std"),
                loss_mean             = ("loss_mean",      "mean"),
                loss_std_ep           = ("loss_mean",      "std"),
                send_rate_instability = ("send_rate_std",  "mean"),
                reward_mean           = ("reward_mean",    "mean"),
                reward_std_ep         = ("reward_mean",    "std"),
            )
            .reset_index()
        )
        path = os.path.join(self.csv_dir, "per_seed_results.csv")
        self.per_seed_df.to_csv(path, index=False)
        print(f"  [SummaryStats] Level 2 written: {path}  "
              f"({len(self.per_seed_df)} rows)")

    # ------------------------------------------------------------------ #
    # Level 3 — across seeds                                               #
    # ------------------------------------------------------------------ #

    def _level3_across_seed(self):
        """Average seed-level metrics across seeds per (model, scenario)."""
        self.across_seed_df = (
            self.per_seed_df
            .groupby(["model", "scenario"])
            .agg(
                num_seeds             = ("seed",                "count"),
                throughput_mean       = ("throughput_mean",     "mean"),
                throughput_std_seeds  = ("throughput_mean",     "std"),
                latency_mean          = ("latency_mean",        "mean"),
                latency_std_seeds     = ("latency_mean",        "std"),
                loss_mean             = ("loss_mean",           "mean"),
                loss_std_seeds        = ("loss_mean",           "std"),
                send_rate_instability = ("send_rate_instability","mean"),
                reward_mean           = ("reward_mean",         "mean"),
                reward_std_seeds      = ("reward_mean",         "std"),
            )
            .reset_index()
        )
        path = os.path.join(self.csv_dir, "across_seeds_aggregated.csv")
        self.across_seed_df.to_csv(path, index=False)
        print(f"  [SummaryStats] Level 3 written: {path}  "
              f"({len(self.across_seed_df)} rows)")
        print(self.across_seed_df.to_string())
