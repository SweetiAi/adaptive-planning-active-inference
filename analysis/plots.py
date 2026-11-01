"""
Publication-style static figures for the adaptive planning-depth extension.

Scientific basis:
Paul, A., Isomura, T., & Razi, A. (2024).
On Predictive Planning and Counterfactual Learning in Active Inference.
Entropy, 26(6), 484. https://doi.org/10.3390/e26060484

The underlying mixed CL/DPEFE architecture, DPEFE planner, Beta arbitration,
and mutating-grid task originate from Paul et al. This module visualizes only
the project-specific counterfactual planning-depth analyses and saved summary
tables.
"""


from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import PHASES, _bool, _validate_evaluation_setup


def _read(output: Path, name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(output / name)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if not frame.empty:
        _validate_evaluation_setup(frame)
    if "beta_frozen" in frame:
        frame["beta_frozen"] = frame.beta_frozen.map(_bool)
    return frame


def _primary(frame: pd.DataFrame, normal_beta: bool = True) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "utility" in frame:
        frame = frame[frame.utility == "capped_episode_length"]
    if normal_beta and "beta_frozen" in frame:
        frame = frame[frame.beta_frozen == False]
    return frame


def _finish(fig, output: Path, name: str, files: list[str]) -> None:
    fig.tight_layout()
    path = output / "figures" / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    files.append(str(path.resolve()))


def _representative_penalties(values: np.ndarray) -> list[float]:
    values = sorted(set(float(v) for v in values))
    if len(values) <= 3:
        return values
    return [values[0], values[len(values) // 2], values[-1]]


def _distribution_plot(distribution: pd.DataFrame, phases: list[str], penalties: list[float],
                       title: str):
    fig, axes = plt.subplots(1, len(penalties), figsize=(5 * len(penalties), 4), squeeze=False,
                             sharey=True)
    depths = sorted(distribution.planning_depth.unique())
    colours = plt.cm.viridis(np.linspace(.15, .85, len(depths)))
    for axis, penalty in zip(axes[0], penalties):
        data = distribution[(distribution["lambda"] == penalty) & distribution.phase.isin(phases)]
        table = data.pivot(index="phase", columns="planning_depth", values="probability").reindex(
            index=phases, columns=depths).fillna(0)
        bottom = np.zeros(len(phases))
        for depth, colour in zip(depths, colours):
            probability = table[depth].to_numpy()
            axis.bar(phases, probability, bottom=bottom, label=f"N={depth}", color=colour)
            bottom += probability
        axis.set_ylim(0, 1)
        axis.set_title(f"λ = {penalty:g}")
        axis.set_xlabel("Checkpoint phase")
    axes[0, 0].set_ylabel("Selection-chosen depth probability")
    axes[0, -1].legend(title="Planning depth", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(title)
    return fig


def make_plots(output_dir: str | Path) -> list[str]:
    """Generate the seven requested figures; all task outcomes come from TEST data."""
    output = Path(output_dir)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    fixed = _read(output, "fixed_depth_summary.csv")
    frontier = _primary(_read(output, "frontier_summary.csv"))
    mixtures = _primary(_read(output, "context_blind_mixture_frontier.csv"))
    distribution = _primary(_read(output, "optimal_depth_distribution.csv"))
    shuffled = _primary(_read(output, "shuffled_oracle_summary.csv"))
    runtime = _read(output, "runtime_scaling.csv")
    ablation = _read(output, "beta_frozen_summary.csv")
    files: list[str] = []
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "axes.titlesize": 11})

    if not frontier.empty:
        fig, axis = plt.subplots(figsize=(8, 5))
        normal_fixed = _primary(fixed)
        normal_fixed = normal_fixed[normal_fixed.phase == "all"].sort_values("planning_depth")
        axis.scatter(normal_fixed.planning_cost, -normal_fixed.mean_capped_episode_length,
                     color="black", s=45, label="Fixed depth", zorder=5)
        for row in normal_fixed.itertuples():
            axis.annotate(f"N={row.planning_depth}", (row.planning_cost, -row.mean_capped_episode_length),
                          xytext=(4, 5), textcoords="offset points", fontsize=8)
        mixture = mixtures.sort_values("planning_cost")
        axis.plot(mixture.planning_cost, -mixture.selection_fit_test_loss, "--", color="#5b8e38",
                  label="Context-blind mixture (selection fit)")
        axis.plot(mixture.planning_cost, -mixture.optimistic_test_hull_loss, ":", color="0.5",
                  label="Optimistic TEST convex envelope")
        curve = frontier.sort_values("oracle_cost")
        axis.plot(curve.oracle_cost, -curve.shuffled_test_loss_mean, "o-", color="#d39435",
                  markersize=4, label="Shuffled oracle (same compute)")
        axis.plot(curve.oracle_cost, -curve.oracle_test_loss, "o-", color="#3366a1",
                  markersize=5, label="Context-aware oracle")
        if curve.oracle_loss_ci_low.notna().any():
            axis.fill_between(curve.oracle_cost, -curve.oracle_loss_ci_high,
                              -curve.oracle_loss_ci_low, color="#3366a1", alpha=.12,
                              label="Oracle seed-bootstrap 95% interval")
        axis.set_xlabel("Mean theoretical planning cost (N)")
        axis.set_ylabel("Held-out navigation performance (−capped episode steps)")
        axis.set_title("Fresh-replanning depth: performance–computation frontier")
        axis.legend(fontsize=8)
        _finish(fig, output, "performance_cost_frontier.png", files)

    if not distribution.empty:
        penalties = _representative_penalties(distribution["lambda"].to_numpy())
        phases = [p for p in PHASES if p in set(distribution.phase)]
        fig = _distribution_plot(distribution, phases, penalties, "Oracle depth assignments by learning phase")
        _finish(fig, output, "optimal_depth_by_phase.png", files)
        hard_phases = [p for p in ["H1", "H3"] if p in phases]
        if hard_phases:
            fig = _distribution_plot(distribution, hard_phases, penalties,
                                     "Early versus late learning in the same hard maze")
            _finish(fig, output, "h1_vs_h3_depth_distribution.png", files)

    if not shuffled.empty:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        shuffled = shuffled.sort_values("lambda")
        axis.axhline(0, color="0.5", linewidth=1)
        axis.plot(shuffled["lambda"], shuffled.oracle_minus_shuffled, "o-", color="#3366a1",
                  label="Oracle − mean shuffled performance")
        axis.fill_between(shuffled["lambda"], shuffled.permutation_delta_p025,
                          shuffled.permutation_delta_p975, color="#d39435", alpha=.2,
                          label="Central 95% assignment-permutation range")
        if shuffled.seed_bootstrap_ci_low.notna().any():
            axis.fill_between(shuffled["lambda"], shuffled.seed_bootstrap_ci_low,
                              shuffled.seed_bootstrap_ci_high, color="#3366a1", alpha=.2,
                              label="Seed-bootstrap 95% interval for mean gap")
        positive = shuffled.loc[shuffled["lambda"] > 0, "lambda"]
        if len(positive):
            axis.set_xscale("symlog", linthresh=float(positive.min()))
        axis.set_xlabel("Compute penalty λ (steps per unit N)")
        axis.set_ylabel("Held-out performance advantage (episode steps)")
        axis.set_title("Allocation-specific headroom at exactly matched compute")
        axis.legend(fontsize=8)
        _finish(fig, output, "shuffled_oracle_comparison.png", files)

    if not ablation.empty:
        phases = [p for p in PHASES if p in set(ablation.phase)]
        fig, axes = plt.subplots(1, len(phases), figsize=(3.4 * len(phases), 4), squeeze=False)
        for axis, phase in zip(axes[0], phases):
            data = ablation[ablation.phase == phase].sort_values("planning_depth")
            axis.plot(data.planning_depth, data.normal_mean_capped_episode_length, "o-",
                      label="Normal beta dynamics", color="#3366a1")
            axis.plot(data.planning_depth, data.frozen_mean_capped_episode_length, "s--",
                      label="Beta fixed at checkpoint", color="#b96447")
            axis.set_title(phase)
            axis.set_xlabel("Planning depth N")
        axes[0, 0].set_ylabel("Mean held-out capped episode steps")
        axes[0, -1].legend(fontsize=8)
        fig.suptitle("Depth response with normal and frozen beta")
        _finish(fig, output, "beta_frozen_comparison.png", files)

    if not runtime.empty:
        fig, axis = plt.subplots(figsize=(6.5, 4.5))
        axis.plot(runtime.planning_depth, runtime["mean"], "o-", label="Mean fresh planning wall time")
        axis.plot(runtime.planning_depth, runtime["median"], "s--", label="Median fresh planning wall time")
        axis.set_xlabel("Planning depth N (primary theoretical cost)")
        axis.set_ylabel("Planning wall time (seconds)")
        axis.set_title("Engineering runtime scaling; scientific cost remains C(N)=N")
        axis.legend(fontsize=8)
        _finish(fig, output, "runtime_vs_depth.png", files)

    normal_fixed = _primary(fixed)
    if not normal_fixed.empty:
        phases = [p for p in PHASES if p in set(normal_fixed.phase)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for phase in phases:
            data = normal_fixed[normal_fixed.phase == phase].sort_values("planning_depth")
            axes[0].plot(data.planning_depth, data.mean_capped_episode_length, "o-", label=phase)
            axes[1].plot(data.planning_depth, data.mean_success, "o-", label=phase)
        axes[0].set_ylabel("Mean held-out capped episode steps")
        axes[1].set_ylabel("Held-out success rate")
        axes[1].set_ylim(-.02, 1.02)
        for axis in axes:
            axis.set_xlabel("Planning depth N")
            axis.legend(title="Phase", fontsize=8)
        fig.suptitle("Forced-fresh mixed-model depth response by phase")
        _finish(fig, output, "fixed_depth_by_phase.png", files)
    return files
