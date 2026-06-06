"""
plots.py — Plotter

Generates all episode-metric time-series and scatter plots from the
step-level DataFrame produced by DataLoader.

Plots generated
---------------
1.  Latency vs Time          (by scenario)
2.  Loss Rate vs Time        (by scenario)
3.  Reward vs Time           (by scenario)
4.  Send Rate vs Time        (by scenario)
5.  Send Rate Oscillation    (by scenario)
6.  Throughput vs Latency Pareto scatter (by scenario)
7.  Throughput vs Loss Pareto scatter    (by scenario)
8.  Summary bar charts       (throughput, latency, reward)

All PNGs are written to results/plots/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    """Generate all episode-metric plots.

    Parameters
    ----------
    steps_df : pd.DataFrame
        Step-level DataFrame from DataLoader.get_steps().
    plots_dir : str
        Output directory for PNG files. Default: "results/plots".
    """

    # consistent model colour map
    MODEL_COLOURS = {
        "lstm":         "#1f77b4",
        "stacking10HL": "#ff7f0e",
        "stacking3HL":  "#2ca02c",
        "stacking5HL":  "#d62728",
    }

    def __init__(self, steps_df: pd.DataFrame, plots_dir: str = "results/plots"):
        self.df        = steps_df
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

        self._models    = sorted(self.df["model"].unique())
        # Scenario display order: flat → step → crossrtt (per paper layout)
        _order = ["flat", "step", "crossrtt"]
        _avail = self.df["scenario"].unique()
        self._scenarios = [s for s in _order if s in _avail] +                           [s for s in sorted(_avail) if s not in _order]

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self):
        """Generate and save all plots."""
        self._plot_metric_by_scenario("latency",   "Latency vs Time",   "Latency (s)")
        self._plot_metric_by_scenario("loss",      "Loss Rate vs Time",  "Loss Rate")
        self._plot_metric_by_scenario("reward",    "Reward vs Time",     "Reward")
        self._plot_metric_by_scenario("send_rate", "Send Rate vs Time",  "Send Rate")
        self._plot_oscillation()
        self._plot_pareto("latency", "Latency",   "Throughput_vs_Latency_Pareto")
        self._plot_pareto("loss",    "Loss Rate",  "Throughput_vs_Loss_Pareto")
        self._plot_summary_bars()
        print(f"[Plotter] All plots saved to '{self.plots_dir}/'")
        return self

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _colour(self, model: str) -> str:
        return self.MODEL_COLOURS.get(model, "#888888")

    def _aggregate(self, arr2d: np.ndarray):
        """Return (mean, ci_95) across rows of a 2D array."""
        mean = arr2d.mean(axis=0)
        std  = arr2d.std(axis=0)
        ci   = 1.96 * std / np.sqrt(arr2d.shape[0])
        return mean, ci

    def _subplot_grid(self, n: int, sharey: bool = False):
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=sharey)
        if n == 1:
            axes = [axes]
        return fig, list(axes)

    def _episode_arrays(self, model: str, scenario: str, metric: str):
        """Return list of step-value arrays, one per episode."""
        sub = self.df[(self.df["model"] == model) &
                      (self.df["scenario"] == scenario)]
        arrays = []
        #ep_count = 0 #set to track episode range for convergence analysis
        for (_, ep), grp in sub.groupby(["seed", "episode"]):
            #ep_count +=1
            #if ep_count <=400: continue #use only last 100episodes
            arrays.append(grp.sort_values("step")[metric].values)
            #if ep_count == 100: break #use only first 100 episodes
        return arrays

    def _save(self, filename: str):
        path = os.path.join(self.plots_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  [Plotter] Saved: {path}")

    # ------------------------------------------------------------------ #
    # Plot 1–4 — metric vs time, one subplot per scenario                 #
    # ------------------------------------------------------------------ #

    def _plot_metric_by_scenario(self, metric: str, title: str, ylabel: str):
        fig, axes = self._subplot_grid(len(self._scenarios))
        for ax, scenario in zip(axes, self._scenarios):
            for model in self._models:
                arrays = self._episode_arrays(model, scenario, metric)
                if not arrays:
                    continue
                # pad/trim to common length then stack
                min_len = min(len(a) for a in arrays)
                mat = np.array([a[:min_len] for a in arrays])
                mean, ci = self._aggregate(mat)
                t = np.arange(min_len)
                c = self._colour(model)
                ax.plot(t, mean, label=model, color=c)
                ax.fill_between(t, mean - ci, mean + ci, alpha=0.2, color=c)
            ax.set_title(scenario)
            ax.set_ylabel(ylabel if ax == axes[0] else "")
            ax.grid(True, alpha=0.4)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(self._models), bbox_to_anchor=(0.5, 1.02))
        for ax in axes:
            ax.set_xlabel("Time (Monitor Interval)")
        fig.suptitle(title, y=1.06)
        self._save(f"{title.replace(' ', '_')}_by_scenario.png")

    # ------------------------------------------------------------------ #
    # Plot 5 — send rate oscillation |Δ send_rate|                       #
    # ------------------------------------------------------------------ #

    def _plot_oscillation(self):
        fig, axes = self._subplot_grid(len(self._scenarios))
        for ax, scenario in zip(axes, self._scenarios):
            for model in self._models:
                arrays = self._episode_arrays(model, scenario, "send_rate")
                if not arrays:
                    continue
                osc = [np.abs(np.diff(a)) for a in arrays]
                min_len = min(len(o) for o in osc)
                mat = np.array([o[:min_len] for o in osc])
                mean, ci = self._aggregate(mat)
                t = np.arange(min_len)
                c = self._colour(model)
                ax.plot(t, mean, label=model, color=c)
                ax.fill_between(t, mean - ci, mean + ci, alpha=0.15, color=c)
            ax.set_title(scenario)
            ax.set_ylabel("|Δ Send Rate|" if ax == axes[0] else "")
            ax.grid(True, alpha=0.4)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(self._models), bbox_to_anchor=(0.5, 1.02))
        for ax in axes:
            ax.set_xlabel("Time (Monitor Interval)")
        fig.suptitle("Send Rate Oscillation vs Time", y=1.06)
        self._save("Send_Rate_Oscillation_vs_Time.png")

    # ------------------------------------------------------------------ #
    # Plot 6–7 — Pareto scatter (throughput vs latency or loss)          #
    # ------------------------------------------------------------------ #

    def _plot_pareto(self, x_metric: str, x_label: str, filename: str):
        fig, axes = self._subplot_grid(len(self._scenarios), sharey=True)
        for ax, scenario in zip(axes, self._scenarios):
            for model in self._models:
                sub = self.df[(self.df["model"] == model) &
                              (self.df["scenario"] == scenario)]
                if sub.empty:
                    continue
                x    = sub[x_metric].values
                thpt = sub["throughput"].values
                c    = self._colour(model)
                ax.scatter(x, thpt, alpha=0.15, s=10, color=c, label=model)
                ax.scatter(x.mean(), thpt.mean(), s=120, marker="X",
                           color=c, label=f"{model} mean",
                           edgecolors="black", linewidths=0.5)
            ax.set_title(scenario)
            ax.set_ylabel("Throughput" if ax == axes[0] else "")
            ax.grid(True, alpha=0.4)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   ncol=4, bbox_to_anchor=(0.5, 1.02))
        for ax in axes:
            ax.set_xlabel(x_label)
        fig.suptitle(f"Throughput vs {x_label} (Pareto)", y=1.06)
        self._save(f"{filename}.png")

    # ------------------------------------------------------------------ #
    # Plot 8 — summary bar charts                                         #
    # ------------------------------------------------------------------ #

    def _plot_summary_bars(self):
        metrics = [
            ("throughput", "Mean Throughput (Mbps)", "Throughput (Mbps)"),
            ("latency",    "Mean Latency (s)",        "Latency (s)"),
            ("reward",     "Mean Reward",             "Reward"),
        ]
        for metric, title, ylabel in metrics:
            keys   = [(m, s) for m in self._models for s in self._scenarios]
            means, cis, labels = [], [], []
            for model, scenario in keys:
                sub = self.df[(self.df["model"] == model) &
                              (self.df["scenario"] == scenario)][metric]
                if sub.empty:
                    continue
                vals = sub.values
                means.append(vals.mean())
                cis.append(1.96 * vals.std() / np.sqrt(len(vals)))
                labels.append(f"{model}\n{scenario}")

            x = np.arange(len(means))
            fig, ax = plt.subplots(figsize=(max(8, len(means) * 1.4), 5))
            ax.bar(x, means, yerr=cis, capsize=4, alpha=0.75, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} — mean ± 95% CI")
            ax.grid(axis="y", alpha=0.4)
            self._save(f"bar_{metric}_summary.png")
