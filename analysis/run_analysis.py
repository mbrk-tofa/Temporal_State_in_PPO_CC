"""
run_analysis.py — single entry point for all post-experiment analysis.

Run this script once after the experiment completes to generate all
CSVs, plots, and LaTeX tables.

Usage
-----
    python run_analysis.py

    # Run specific modules only:
    python run_analysis.py --only summary plots

    # Use non-default log directory:
    python run_analysis.py --log-dir /path/to/logs

Output structure
----------------
results/
    csv/
        per_episode_results.csv
        per_seed_results.csv
        across_seeds_aggregated.csv
        stat_test_results.csv
        inference_cost_per_episode.csv
        inference_cost_summary.csv
    plots/
        Latency_vs_Time_by_scenario.png
        Loss_Rate_vs_Time_by_scenario.png
        Reward_vs_Time_by_scenario.png
        Send_Rate_vs_Time_by_scenario.png
        Send_Rate_Oscillation_vs_Time.png
        Throughput_vs_Latency_Pareto.png
        Throughput_vs_Loss_Pareto.png
        bar_throughput_summary.png
        bar_latency_summary.png
        bar_reward_summary.png
        inference_cost_bar.png
        inference_latency_cdf.png
        inference_latency_cdf_by_scenario.png
        inference_time_per_episode.png
    latex/
        stat_test_results.tex
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------ #
# Allow running from the project root without installing the package  #
# ------------------------------------------------------------------ #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader        import DataLoader
from summary       import SummaryStats
from plots         import Plotter
from stat_tests    import StatTester
from cost_analysis import CostAnalyser
from train_conv import plot_convergence
from inference_analysis  import (
    build_dataframe, plot_cdfs, plot_boxplots,
    plot_seed_variance, plot_step_drift, compute_summary_statistics,
    COST_DIR as INFER_COST_DIR,
)

# ------------------------------------------------------------------ #
# Output directories — all results live under results/               #
# ------------------------------------------------------------------ #
REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR   = str(REPO_ROOT / "results" / "csv")
PLOTS_DIR = str(REPO_ROOT / "results" / "plots")
LATEX_DIR = str(REPO_ROOT / "results" / "latex")

# ------------------------------------------------------------------ #
# Available analysis modules                                          #
# ------------------------------------------------------------------ #
ALL_MODULES = ["summary", "plots", "stat_tests", "cost", "training_curve", "inference"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all post-experiment analysis for the RL CC study."
    )
    parser.add_argument(
        "--log-dir",
        default=str(REPO_ROOT / "logs" / "eval"),
        help="Directory containing raw evaluation JSONL logs.",
    )
    parser.add_argument(
        "--cost-dir",
        default=str(REPO_ROOT / "logs" / "cost_files"),
        help="Directory containing inference cost JSON files "
             "(default: logs/cost_files/)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ALL_MODULES,
        default=ALL_MODULES,
        help="Run only the specified modules. Default: all.",
    )
    return parser.parse_args()


def banner(text):
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main():
    args = parse_args()
    modules = set(args.only)

    # ---------------------------------------------------------------- #
    # Create output directories                                         #
    # ---------------------------------------------------------------- #
    for d in [CSV_DIR, PLOTS_DIR, LATEX_DIR]:
        os.makedirs(d, exist_ok=True)

    t_start = time.time()

    # ---------------------------------------------------------------- #
    # Step 1 — Load episode logs (shared by summary, plots, stat_tests) #
    # ---------------------------------------------------------------- #
    loader      = None
    episode_df  = None
    steps_df    = None

    needs_episode_data = modules & {"summary", "plots", "stat_tests"}
    if needs_episode_data:
        banner("Loading episode logs")
        loader     = DataLoader(log_dir=args.log_dir)
        steps_df   = loader.get_steps()
        episode_df = loader.get_episodes()
        print(f"  Models   : {loader.models()}")
        print(f"  Scenarios: {loader.scenarios()}")

    # ---------------------------------------------------------------- #
    # Step 2 — Summary statistics                                       #
    # ---------------------------------------------------------------- #
    if "summary" in modules:
        banner("Summary statistics (3-level aggregation)")
        SummaryStats(
            episode_df=episode_df,
            csv_dir=CSV_DIR,
        ).run()

    # ---------------------------------------------------------------- #
    # Step 3 — Episode-metric plots                                     #
    # ---------------------------------------------------------------- #
    if "plots" in modules:
        banner("Episode-metric plots")
        Plotter(
            steps_df=steps_df,
            plots_dir=PLOTS_DIR,
        ).run()

    # ---------------------------------------------------------------- #
    # Step 4 — Statistical tests                                        #
    # ---------------------------------------------------------------- #
    if "stat_tests" in modules:
        banner("Statistical significance tests")
        StatTester(
            episode_df=episode_df,
            csv_dir=CSV_DIR,
            latex_dir=LATEX_DIR,
        ).run()

    # ---------------------------------------------------------------- #
    # Step 5 — Inference and training cost analysis                     #
    # ---------------------------------------------------------------- #
    if "cost" in modules:
        banner("Inference and training cost analysis")
        CostAnalyser(
            cost_dir=args.cost_dir,
            csv_dir=CSV_DIR,
            plots_dir=PLOTS_DIR,
        ).run()

    # ---------------------------------------------------------------- #
    # Step 6 — Training convergence curve                               #
    # ---------------------------------------------------------------- #
    if "training_curve" in modules:
        banner("Training convergence curve")
        plot_convergence()
 
    # ---------------------------------------------------------------- #
    # Step 7 — Inference time CDF and statistics                        #
    # ---------------------------------------------------------------- #
    if "inference" in modules:
        banner("Inference time CDF and statistics")
        try:
            infer_df = build_dataframe()
            if not infer_df.empty:
                plot_cdfs(infer_df)
                plot_boxplots(infer_df)
                plot_seed_variance(infer_df)
                plot_step_drift(infer_df)
                compute_summary_statistics(infer_df)
            else:
                print("  [inference] No per-step cost data found — "
                      "skipping CDF plots. Run run_eval_only.py first.")
        except Exception as e:
            print(f"  [inference] Skipped: {e}")

    # ---------------------------------------------------------------- #
    # Done                                                              #
    # ---------------------------------------------------------------- #
    elapsed = time.time() - t_start
    banner(f"All analysis complete in {elapsed:.1f}s")
    print(f"\n  CSVs      -> {CSV_DIR}/")
    print(f"  Plots     -> {PLOTS_DIR}/")
    print(f"  LaTeX     -> {LATEX_DIR}/")
    print(f"  Inference -> results/plots/  (cdf, boxplot, seed_variance, step_drift)\n")


if __name__ == "__main__":
    main()
