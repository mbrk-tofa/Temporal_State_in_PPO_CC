"""
analysis — post-experiment analysis package for RL congestion control study.

Modules
-------
loader       : DataLoader  — reads JSONL episode logs into a shared DataFrame
summary      : SummaryStats — three-level aggregation (episode, seed, across-seed)
plots        : Plotter      — all episode-metric time-series and scatter plots
stat_tests   : StatTester   — pairwise Welch t-tests with Bonferroni correction
cost_analysis: CostAnalyser — inference and training cost plots and summaries

All outputs are written to:
    results/csv/    — CSV files
    results/plots/  — PNG figures
    results/latex/  — LaTeX tables
"""
