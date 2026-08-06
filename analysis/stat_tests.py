"""Seed-level paired bootstrap analysis for the evaluation study.

Episodes are repeated rollouts of each trained policy, not independent model
replications. This module therefore averages the ten episodes within each
model/seed/scenario cell and performs inference on the five trained seeds.
"""

from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "throughput_mean": "throughput_mbps",
    "latency_mean": "latency_s",
    "loss_mean": "loss_rate",
    "send_rate_std": "send_rate_instability",
    "reward_mean": "reward",
}
SOURCE_METRIC = {value: key for key, value in METRICS.items()}

MODEL_LABELS = {
    "lstm": "LSTM",
    "stacking3HL": "Stacking k=3",
    "stacking5HL": "Stacking k=5",
    "stacking10HL": "Stacking k=10",
}


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 20_000):
    """Percentile 95% CI for the mean of paired seed differences."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    means = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


class StatTester:
    """Generate seed-level summaries and paired bootstrap contrasts.

    The retained class name preserves compatibility with ``run_analysis.py``.
    It deliberately does not report episode-level t-tests or p-values.
    """

    def __init__(self, episode_df: pd.DataFrame, csv_dir: str = "results/csv", **_):
        self.episode_df = episode_df.copy()
        self.csv_dir = Path(csv_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.seed_metrics_df = None
        self.summary_df = None
        self.results_df = None

    def run(self):
        self._aggregate_to_seed_level()
        self._build_summary()
        self._build_paired_contrasts()
        self._write_outputs()
        print("[StatTester] Seed-level bootstrap analysis complete.")
        return self

    def get_results(self) -> pd.DataFrame:
        return self.results_df

    def _aggregate_to_seed_level(self):
        required = {"model", "scenario", "seed", *METRICS}
        missing = required.difference(self.episode_df.columns)
        if missing:
            raise ValueError(f"Episode data missing columns: {sorted(missing)}")

        self.seed_metrics_df = (
            self.episode_df.groupby(["model", "scenario", "seed"], as_index=False)
            .agg({metric: "mean" for metric in METRICS})
            .rename(columns=METRICS)
            .sort_values(["scenario", "model", "seed"])
        )
        self.seed_metrics_df["model_label"] = self.seed_metrics_df["model"].map(MODEL_LABELS)

    def _build_summary(self):
        rows = []
        rng = np.random.default_rng(20260803)
        for (scenario, model), group in self.seed_metrics_df.groupby(["scenario", "model"]):
            for metric in METRICS.values():
                values = group[metric]
                ci_low, ci_high = bootstrap_mean_ci(values.to_numpy(), rng)
                rows.append({
                    "scenario": scenario,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "metric": SOURCE_METRIC[metric],
                    "n_seeds": values.size,
                    "mean": values.mean(),
                    "sd_across_seeds": values.std(ddof=1),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                })
        self.summary_df = pd.DataFrame(rows)

    def _build_paired_contrasts(self):
        rows = []
        # Advance the same deterministic RNG after the model-mean intervals,
        # preserving the exact sequence used to produce the manuscript CIs.
        rng = np.random.default_rng(20260803)
        for _, group in self.seed_metrics_df.groupby(["scenario", "model"]):
            for metric in METRICS.values():
                bootstrap_mean_ci(group[metric].to_numpy(), rng)
        for scenario, scenario_df in self.seed_metrics_df.groupby("scenario"):
            baseline = scenario_df[scenario_df["model"] == "lstm"].set_index("seed")
            for candidate in ("stacking3HL", "stacking5HL", "stacking10HL"):
                comparison = scenario_df[scenario_df["model"] == candidate].set_index("seed").reindex(baseline.index)
                if comparison[list(METRICS.values())].isna().any().any():
                    raise ValueError(f"Unpaired seeds in {scenario}: {candidate} vs lstm")
                for metric in METRICS.values():
                    diffs = (comparison[metric].to_numpy() - baseline[metric].to_numpy())
                    ci_low, ci_high = bootstrap_mean_ci(diffs, rng)
                    rows.append({
                        "scenario": scenario,
                        "comparison": f"{MODEL_LABELS[candidate]} minus LSTM",
                        "metric": SOURCE_METRIC[metric],
                        "n_paired_seeds": diffs.size,
                        "mean_difference": diffs.mean(),
                        "sd_difference": diffs.std(ddof=1),
                        "bootstrap_ci95_low": ci_low,
                        "bootstrap_ci95_high": ci_high,
                        "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
                        "bootstrap_draws": 20_000,
                        "bootstrap_seed": 20260803,
                    })
        self.results_df = pd.DataFrame(rows)

    def _write_outputs(self):
        self.seed_metrics_df.to_csv(self.csv_dir / "seed_level_metrics.csv", index=False)
        self.summary_df.to_csv(self.csv_dir / "seed_level_summary_ci.csv", index=False)
        self.results_df.to_csv(self.csv_dir / "paired_seed_contrasts_ci.csv", index=False)
