"""
loader.py — DataLoader

Reads all JSONL episode logs produced by network_sim and builds a
single shared DataFrame used by all analysis modules.

Each row in the DataFrame represents one monitor-interval step, tagged
with model, scenario, seed, and episode. This avoids each analysis
module loading files independently — the I/O cost is paid once.

JSONL format (one JSON object per line, one line per episode):
    {"Scenario": "crossrtt", "Seed": 1, "Episode": 1, "Steps": [...]}
"""

import json
import glob
import os
import re
import numpy as np
import pandas as pd


class DataLoader:
    """Load all JSONL episode logs into a structured DataFrame.

    Parameters
    ----------
    log_dir : str
        Directory containing *.jsonl episode log files produced by network_sim.
        Default: "logs/"

    Attributes
    ----------
    steps_df : pd.DataFrame
        One row per monitor-interval step with columns:
        model, scenario, seed, episode, step,
        throughput, latency, loss, send_rate, reward

    episode_df : pd.DataFrame
        One row per episode with per-episode mean/std of each metric.
        Built from steps_df by _build_episode_df().
    """

    STEP_COLUMNS = [
        "model", "scenario", "seed", "episode", "step",
        "throughput", "latency", "loss", "send_rate", "reward",
    ]

    def __init__(self, log_dir: str = "logs/eval"):
        self.log_dir   = log_dir
        self.steps_df  = None
        self.episode_df = None
        self._load()

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def get_steps(self) -> pd.DataFrame:
        """Return the step-level DataFrame."""
        return self.steps_df

    def get_episodes(self) -> pd.DataFrame:
        """Return the episode-level DataFrame (per-episode means)."""
        return self.episode_df

    def models(self):
        return sorted(self.steps_df["model"].unique())

    def scenarios(self):
        return sorted(self.steps_df["scenario"].unique())

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    def _load(self):
        pattern = os.path.join(self.log_dir, "*.jsonl")
        files   = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(
                f"No .jsonl files found in '{self.log_dir}'. "
                "Run the experiment first or check log_dir."
            )

        rows = []
        for filepath in files:
            rows.extend(self._read_jsonl(filepath))

        self.steps_df   = pd.DataFrame(rows, columns=self.STEP_COLUMNS)
        self.episode_df = self._build_episode_df()

        print(f"[DataLoader] Loaded {len(files)} files — "
              f"{len(self.steps_df):,} steps across "
              f"{len(self.episode_df):,} episodes")

    def _read_jsonl(self, filepath: str) -> list:
        """Read one JSONL file and return a list of step rows."""
        rows = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep_obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [DataLoader] Skipping malformed line in {filepath}: {e}")
                    continue

                model    = ep_obj.get("model_type") or self._model_from_path(filepath)
                scenario = ep_obj.get("Scenario", "unknown")
                seed     = int(ep_obj.get("Seed", 0))
                episode  = int(ep_obj.get("Episode", 0))
                steps    = ep_obj.get("Steps", [])

                if not steps:
                    continue

                for step in steps:
                    rows.append({
                        "model":      model,
                        "scenario":   scenario,
                        "seed":       seed,
                        "episode":    episode,
                        "step":       int(step.get("Time", 0)),
                        "throughput": float(step.get("Throughput", 0.0)),
                        "latency":    float(step.get("Latency", 0.0)),
                        "loss":       float(step.get("Loss Rate", 0.0)),
                        "send_rate":  float(step.get("Send Rate", 0.0)),
                        "reward":     float(step.get("Reward", 0.0)),
                    })
        return rows

    @staticmethod
    def _model_from_path(filepath: str) -> str:
        """Extract model name from filename as fallback.

        Expected pattern: {model}_{scenario}_seed_{seed}.jsonl
        """
        name = os.path.basename(filepath).replace(".jsonl", "")
        return name.split("_")[0]

    # ------------------------------------------------------------------ #
    # Episode-level aggregation                                           #
    # ------------------------------------------------------------------ #

    def _build_episode_df(self) -> pd.DataFrame:
        """Aggregate steps to episode-level means and std."""
        grp = self.steps_df.groupby(["model", "scenario", "seed", "episode"])

        ep = grp.agg(
            throughput_mean = ("throughput", "mean"),
            latency_mean    = ("latency",    "mean"),
            loss_mean       = ("loss",       "mean"),
            send_rate_std   = ("send_rate",  "std"),   # instability proxy
            reward_mean     = ("reward",     "mean"),
            num_steps       = ("step",       "count"),
        ).reset_index()

        # scale throughput to Mbps and send_rate_std to Mbps
        ep["throughput_mean"] = ep["throughput_mean"] / 1e6
        ep["send_rate_std"]   = ep["send_rate_std"]   / 1e6

        return ep
