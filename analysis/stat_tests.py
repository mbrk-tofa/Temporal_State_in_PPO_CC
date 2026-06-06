"""
stat_tests.py — StatTester

Pairwise Welch's t-tests across all model pairs, scenarios, and metrics.
Bonferroni correction applied over evaluation scenarios only (excluding
the training distribution / default scenario).

Unit of observation: per-episode means (episodes x seeds).
For 10 seeds x 10 episodes = 100 observations per group.

Outputs
-------
results/csv/stat_test_results.csv  — full results table
results/latex/stat_test_results.tex — paper-ready LaTeX table
"""

import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats


# Metrics tested and the direction in which "better" is defined.
# True  = higher is better  (throughput, reward)
# False = lower  is better  (latency, loss, send_rate_std)
METRIC_DIRECTION = {
    "throughput_mean": True,
    "latency_mean":    False,
    "loss_mean":       False,
    "send_rate_std":   False,
    "reward_mean":     True,
}

# The training distribution scenario is excluded from all reported results.
EXCLUDE_SCENARIOS = {"default"}


class StatTester:
    """Run pairwise Welch t-tests with Bonferroni correction.

    Parameters
    ----------
    episode_df : pd.DataFrame
        Episode-level DataFrame from DataLoader.get_episodes().
        Must contain columns: model, scenario, seed, episode,
        throughput_mean, latency_mean, loss_mean, send_rate_std, reward_mean.
    csv_dir : str
        Output directory for CSV. Default: "results/csv".
    latex_dir : str
        Output directory for LaTeX table. Default: "results/latex".
    alpha : float
        Significance level before correction. Default: 0.05.
    """

    def __init__(
        self,
        episode_df:  pd.DataFrame,
        csv_dir:     str = "results/csv",
        latex_dir:   str = "results/latex",
        alpha:       float = 0.05,
    ):
        self.episode_df = episode_df
        self.csv_dir    = csv_dir
        self.latex_dir  = latex_dir
        self.alpha      = alpha
        os.makedirs(csv_dir,   exist_ok=True)
        os.makedirs(latex_dir, exist_ok=True)

        self.results_df = None

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self):
        """Run all tests, print summary, write CSV and LaTeX."""
        self._run_tests()
        self._print_summary()
        self._write_csv()
        self._write_latex()
        print("[StatTester] Done.")
        return self

    def get_results(self) -> pd.DataFrame:
        return self.results_df

    # ------------------------------------------------------------------ #
    # Core testing                                                        #
    # ------------------------------------------------------------------ #

    def _run_tests(self):
        # filter to evaluation scenarios only
        eval_df = self.episode_df[
            ~self.episode_df["scenario"].isin(EXCLUDE_SCENARIOS)
        ]

        scenarios = sorted(eval_df["scenario"].unique())
        models    = sorted(eval_df["model"].unique())
        metrics   = list(METRIC_DIRECTION.keys())
        pairs     = list(itertools.combinations(models, 2))

        # Bonferroni denominator: evaluation scenarios only
        n_tests = len(pairs) * len(scenarios) * len(metrics)
        print(f"[StatTester] Running {n_tests} tests "
              f"({len(pairs)} pairs x {len(scenarios)} scenarios "
              f"x {len(metrics)} metrics) — Bonferroni α/{n_tests}")

        rows = []
        for scenario in scenarios:
            sc_df = eval_df[eval_df["scenario"] == scenario]
            for model_a, model_b in pairs:
                for metric in metrics:
                    vec_a = sc_df[sc_df["model"] == model_a][metric].values
                    vec_b = sc_df[sc_df["model"] == model_b][metric].values

                    if len(vec_a) == 0 or len(vec_b) == 0:
                        print(f"  [StatTester] Missing data: "
                              f"{model_a} vs {model_b} | {scenario} | {metric}")
                        continue

                    t_stat, p_val = stats.ttest_ind(vec_a, vec_b,
                                                     equal_var=False)
                    p_corrected   = min(p_val * n_tests, 1.0)

                    higher_is_better = METRIC_DIRECTION[metric]
                    if higher_is_better:
                        better = model_a if vec_a.mean() >= vec_b.mean() else model_b
                    else:
                        better = model_a if vec_a.mean() <= vec_b.mean() else model_b

                    rows.append({
                        "scenario":         scenario,
                        "model_a":          model_a,
                        "model_b":          model_b,
                        "metric":           metric,
                        "mean_a":           round(vec_a.mean(), 6),
                        "mean_b":           round(vec_b.mean(), 6),
                        "std_a":            round(vec_a.std(),  6),
                        "std_b":            round(vec_b.std(),  6),
                        "n_a":              len(vec_a),
                        "n_b":              len(vec_b),
                        "t_stat":           round(t_stat, 4),
                        "p_value":          round(p_val, 6),
                        "p_corrected":      round(p_corrected, 6),
                        "significant_raw":  bool(p_val       < self.alpha),
                        "significant_bonf": bool(p_corrected < self.alpha),
                        "better_model":     better,
                    })

        self.results_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Output                                                              #
    # ------------------------------------------------------------------ #

    def _print_summary(self):
        df = self.results_df
        n_bonf = df["significant_bonf"].sum()
        n_raw  = df["significant_raw"].sum()
        print(f"\n[StatTester] Results: "
              f"{n_bonf} Bonferroni-significant, "
              f"{n_raw} raw-significant, "
              f"{len(df) - n_raw} not significant\n")

        for scenario in sorted(df["scenario"].unique()):
            print(f"  Scenario: {scenario.upper()}")
            sub = df[df["scenario"] == scenario]
            for _, row in sub.iterrows():
                if row["significant_bonf"]:
                    marker = "** Bonferroni"
                elif row["significant_raw"]:
                    marker = "*  raw only"
                else:
                    marker = "   ns"
                print(
                    f"    {row['model_a']:>16s} vs {row['model_b']:<16s} | "
                    f"{row['metric']:<18s} | "
                    f"t={row['t_stat']:7.3f}  "
                    f"p={row['p_value']:.4f}  "
                    f"p*={row['p_corrected']:.4f}  {marker}"
                )
            print()

        print("Legend: ** Bonferroni-significant (p* < 0.05)  "
              "* raw only (p < 0.05)  ns = not significant")

    def _write_csv(self):
        path = os.path.join(self.csv_dir, "stat_test_results.csv")
        self.results_df.to_csv(path, index=False)
        print(f"  [StatTester] CSV written: {path}")

    def _write_latex(self):
        """Write a paper-ready LaTeX table with Bonferroni-significant
        results highlighted in bold."""
        df   = self.results_df
        rows = []
        for _, r in df.iterrows():
            if r["significant_bonf"]:
                sig = r"$\mathbf{p^*<0.05}$"
            elif r["significant_raw"]:
                sig = r"$p<0.05$"
            else:
                sig = "ns"

            rows.append(
                f"  {r['scenario']} & "
                f"{r['model_a']} vs {r['model_b']} & "
                f"{r['metric']} & "
                f"{r['mean_a']:.4f} & {r['mean_b']:.4f} & "
                f"{r['t_stat']:.3f} & {r['p_value']:.4f} & "
                f"{r['p_corrected']:.4f} & {sig} \\\\"
            )

        latex = (
            r"\begin{table}[ht]" + "\n"
            r"\centering" + "\n"
            r"\caption{Pairwise Welch's t-test results (per-episode means, "
            r"Bonferroni corrected over evaluation scenarios). "
            r"$\mathbf{p^*<0.05}$ = significant after correction; "
            r"$p<0.05$ = raw only; ns = not significant.}" + "\n"
            r"\label{tab:stat_tests}" + "\n"
            r"\resizebox{\textwidth}{!}{%" + "\n"
            r"\begin{tabular}{lllrrrrrr}" + "\n"
            r"\toprule" + "\n"
            r"Scenario & Comparison & Metric & "
            r"Mean A & Mean B & $t$ & $p$ & $p^*$ & Sig. \\" + "\n"
            r"\midrule" + "\n"
            + "\n".join(rows) + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}}" + "\n"
            r"\end{table}"
        )

        path = os.path.join(self.latex_dir, "stat_test_results.tex")
        with open(path, "w") as f:
            f.write(latex)
        print(f"  [StatTester] LaTeX written: {path}")
