"""
Held-out analysis for the adaptive planning-depth extension of the mixed
Active Inference model introduced by Paul, Isomura, and Razi (2024).

Scientific basis:
Paul, A., Isomura, T., & Razi, A. (2024).
On Predictive Planning and Counterfactual Learning in Active Inference.
Entropy, 26(6), 484. https://doi.org/10.3390/e26060484

The original DPEFE planner, Counterfactual Learning pathway, mixed-model
architecture, Beta arbitration mechanism, and mutating-grid implementation
are contributions of Paul et al. and the associated upstream codebase.

This module implements project-specific held-out analysis for counterfactual
planning-depth interventions. It does not implement or replace the original
mixed-model agent.

Selection and TEST samples have distinct roles. Only selection losses choose
oracle depths, the fixed-depth comparator, and the primary matched-cost
mixture. The TEST convex envelope is reported only as an optimistic descriptive
benchmark. Rollouts are averaged within checkpoints; independent training seeds
are the resampling units. No analysis result is returned to agent decision code.

Experiment A compares depths only after one forced fresh replan in every
evaluation branch. The same raw treatment grid supplies every fixed-depth,
oracle, shuffled-oracle, and context-blind comparison.
"""


from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PHASES = ["E1", "E2", "H1", "H2", "H3"]
EXPERIMENT_A_SETUP = {"training_refresh": "original_schedule", "evaluation_replan": "forced_fresh"}
LOSSES = {"capped_episode_length": "capped_episode_length", "navigation_regret": "navigation_regret"}
METRICS = ["capped_episode_length", "episode_length", "action_count", "t_length",
           "legacy_final_step_index", "success", "navigation_regret",
           "path_efficiency", "planning_walltime_seconds", "planning_cost"]


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        if value.strip().lower() not in {"true", "false", "0", "1"}:
            raise ValueError(f"Invalid beta_frozen value: {value!r}")
        return value.strip().lower() in {"true", "1"}
    return bool(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def checkpoint_weights(metadata: pd.DataFrame) -> np.ndarray:
    """Equal seed, then generator trajectory, then phase, then checkpoint weights."""
    if metadata.empty:
        return np.array([], dtype=float)
    frame = metadata.reset_index(drop=True)
    seeds = frame.base_seed.unique()
    weights = np.zeros(len(frame))
    for seed in seeds:
        at_seed = frame.base_seed == seed
        generators = frame.loc[at_seed, "generator_N"].unique()
        for generator in generators:
            at_trajectory = at_seed & (frame.generator_N == generator)
            phases = frame.loc[at_trajectory, "phase"].unique()
            for phase in phases:
                ids = np.flatnonzero(at_trajectory & (frame.phase == phase))
                weights[ids] = 1 / (len(seeds) * len(generators) * len(phases) * len(ids))
    assert np.isclose(weights.sum(), 1)
    return weights


def matched_mixture(depths: np.ndarray, selection_losses: np.ndarray,
                    target_cost: float) -> dict[str, Any]:
    """Solve the context-blind matched-cost LP by enumerating singleton/pair vertices.

    Constraints are sum(w)=1 and sum(w*N)=target_cost.  An optimum needs at
    most two nonzero weights.  This function sees aggregate selection losses,
    never checkpoint features or TEST losses.  Exact ties use the first pair
    in increasing-depth order; all tied mixtures have the same target cost.
    """
    depths = np.asarray(depths, dtype=float)
    losses = np.asarray(selection_losses, dtype=float)
    if len(depths) != len(losses) or not np.all(np.isfinite(losses)):
        raise ValueError("Mixture requires a finite loss for each candidate depth")
    if target_cost < depths.min() - 1e-9 or target_cost > depths.max() + 1e-9:
        raise ValueError("Target compute lies outside the candidate-depth interval")
    best = None
    for lo in range(len(depths)):
        for hi in range(lo, len(depths)):
            if lo == hi:
                if not np.isclose(target_cost, depths[lo], atol=1e-10, rtol=0):
                    continue
                fraction = 0.0
            else:
                fraction = (target_cost - depths[lo]) / (depths[hi] - depths[lo])
                if fraction < -1e-10 or fraction > 1 + 1e-10:
                    continue
                fraction = float(np.clip(fraction, 0, 1))
            weights = np.zeros(len(depths))
            weights[lo] += 1 - fraction
            weights[hi] += fraction
            loss = float(weights @ losses)
            if best is None or loss < best["selection_loss"]:
                best = {"weights": weights, "selection_loss": loss,
                        "lower_N": int(depths[lo]), "upper_N": int(depths[hi]),
                        "upper_weight": fraction}
    if best is None:
        raise AssertionError("No feasible context-blind mixture")
    assert np.isclose(best["weights"] @ depths, target_cost)
    return best


def _seed_means(values: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    """Average checkpoint values using the design, retaining one row per seed."""
    values = np.asarray(values)
    rows = []
    for seed in metadata.base_seed.unique():
        ids = np.flatnonzero(metadata.base_seed.to_numpy() == seed)
        rows.append(np.tensordot(checkpoint_weights(metadata.iloc[ids]), values[ids], axes=1))
    return np.asarray(rows)


def _ci(values: np.ndarray, bootstrap_ids: np.ndarray | None) -> tuple[float, float]:
    if bootstrap_ids is None:
        return np.nan, np.nan
    draws = np.asarray(values)[bootstrap_ids].mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def _validate_evaluation_setup(frame: pd.DataFrame) -> None:
    """Reject stale-cache or unlabelled comparisons before computing any effects."""
    for column, expected in EXPERIMENT_A_SETUP.items():
        if column not in frame:
            raise ValueError(f"Experiment A metadata missing: {column}")
        if frame[column].isna().any() or not frame[column].eq(expected).all():
            raise ValueError(f"Experiment A requires {column}={expected!r} for every treatment")


def _write_table(frame: pd.DataFrame, output: Path, name: str) -> None:
    """Carry the common intervention definition into every derived CSV."""
    frame = frame.copy()
    for column, value in EXPERIMENT_A_SETUP.items():
        frame[column] = value
    frame.to_csv(output / name, index=False)


def _validate_raw(raw: pd.DataFrame, depths: list[int]) -> pd.DataFrame:
    required = {"checkpoint_id", "base_seed", "generator_N", "phase", "planning_depth",
                "beta_frozen", "rollout_split", "capped_episode_length", "episode_length",
                "action_count", "success", "planning_cost", "planning_walltime_seconds"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw rollout columns missing: {missing}")
    _validate_evaluation_setup(raw)
    raw = raw.copy()
    raw["beta_frozen"] = raw.beta_frozen.map(_bool)
    raw["planning_depth"] = raw.planning_depth.astype(int)
    if set(raw.planning_depth) != set(depths):
        raise ValueError("Raw data must contain exactly the configured candidate_N values")
    if not set(raw.rollout_split) <= {"selection", "test"}:
        raise ValueError("rollout_split must be selection or test")
    if not np.array_equal(raw.planning_cost.to_numpy(), raw.planning_depth.to_numpy()):
        raise AssertionError("The primary declared theoretical planning cost must equal N")
    if raw.capped_episode_length.isna().any():
        raise ValueError("Primary outcome contains missing capped episode lengths")
    for column, expected in [("mutation_threshold", 300), ("first_hard_episode", 301)]:
        if column in raw and (raw[column].isna().any() or not raw[column].eq(expected).all()):
            raise ValueError(f"Experiment A requires {column}={expected}")
    if "delta_from_mutation" in raw:
        if "episode_idx" not in raw or not raw.delta_from_mutation.eq(raw.episode_idx - 301).all():
            raise ValueError("delta_from_mutation must equal episode_idx - first_hard_episode (301)")
    for (_, checkpoint), group in raw.groupby(["beta_frozen", "checkpoint_id"], sort=False):
        observed = set(zip(group.rollout_split, group.planning_depth))
        expected = set(itertools.product(["selection", "test"], depths))
        if observed != expected:
            raise ValueError(f"Incomplete selection/TEST depth grid at {checkpoint}")
        for column in ["base_seed", "generator_N", "phase", "episode_idx", "start_state",
                       "environment_version", "optimal_path_length", "delta_from_mutation",
                       "mutation_threshold", "first_hard_episode"]:
            if column in group and group[column].nunique(dropna=False) != 1:
                raise ValueError(f"Checkpoint metadata varies across treatments: {checkpoint}/{column}")
        if "rollout_seed" in group:
            by_split = {split: set(g.rollout_seed) for split, g in group.groupby("rollout_split")}
            if by_split["selection"] & by_split["test"]:
                raise AssertionError("Selection and TEST rollout streams overlap")
    return raw


def _summarise_checkpoints(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["beta_frozen", "checkpoint_id", "planning_depth", "rollout_split"]
    metrics = [column for column in METRICS if column in raw]
    grouped = raw.groupby(keys, sort=True, observed=True)
    result = grouped[metrics].mean().rename(columns={m: f"mean_{m}" for m in metrics})
    result["median_episode_length"] = grouped.episode_length.median()
    result["median_capped_episode_length"] = grouped.capped_episode_length.median()
    result["n_rollouts"] = grouped.size()
    meta_cols = [c for c in ["base_seed", "generator_N", "phase", "episode_idx",
                            "delta_from_mutation", "environment_version", "start_state",
                            "optimal_path_length", "mutation_threshold", "first_hard_episode",
                            *EXPERIMENT_A_SETUP] if c in raw]
    return result.join(grouped[meta_cols].first()).reset_index()


def _make_panel(checkpoints: pd.DataFrame, frozen: bool, depths: np.ndarray,
                loss_column: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = checkpoints[checkpoints.beta_frozen == frozen]
    metadata_columns = [c for c in ["checkpoint_id", "base_seed", "generator_N", "phase",
                                   "episode_idx", "delta_from_mutation", "environment_version",
                                   "start_state", "optimal_path_length", "mutation_threshold",
                                   "first_hard_episode", *EXPERIMENT_A_SETUP] if c in frame]
    metadata = frame[metadata_columns].drop_duplicates("checkpoint_id").sort_values("checkpoint_id")
    metadata = metadata.reset_index(drop=True)
    matrices = []
    for split in ["selection", "test"]:
        matrix = frame[frame.rollout_split == split].pivot(
            index="checkpoint_id", columns="planning_depth", values=f"mean_{loss_column}")
        matrices.append(matrix.reindex(index=metadata.checkpoint_id, columns=depths).to_numpy(float))
    valid = np.isfinite(matrices[0]).all(axis=1) & np.isfinite(matrices[1]).all(axis=1)
    return metadata.loc[valid].reset_index(drop=True), matrices[0][valid], matrices[1][valid]


def _shuffle_assignments(selected: np.ndarray, metadata: pd.DataFrame,
                         rng: np.random.Generator) -> np.ndarray:
    """Shuffle within independent seeds and equal-design-weight strata.

    Balanced data reduce to within-seed permutations across all checkpoints.
    Weight strata additionally preserve exactly the weighted cost if interrupted
    runs supply unequal numbers of checkpoints.  The depth multiset is preserved.
    """
    weights = checkpoint_weights(metadata)
    strata = pd.DataFrame({"seed": metadata.base_seed.to_numpy(), "weight": weights.round(14)})
    shuffled = selected.copy()
    for ids in strata.groupby(["seed", "weight"], sort=False).indices.values():
        shuffled[ids] = rng.permutation(selected[ids])
    assert sorted(shuffled.tolist()) == sorted(selected.tolist())
    return shuffled


def _bootstrap_contrasts(selection: np.ndarray, test: np.ndarray, selected: np.ndarray,
                         metadata: pd.DataFrame, depths: np.ndarray, penalty: float,
                         null_seed_losses: np.ndarray, bootstrap_ids: np.ndarray | None) -> dict:
    if bootstrap_ids is None:
        return {f"{name}_{tail}": np.nan for name in ["oracle_loss", "oracle_minus_fixed_utility",
                "oracle_minus_mixture", "oracle_minus_test_hull", "oracle_minus_shuffled"]
                for tail in ["ci_low", "ci_high"]}
    rows = np.arange(len(selected))
    sel_seed = _seed_means(selection, metadata)
    test_seed = _seed_means(test, metadata)
    oracle_seed = _seed_means(test[rows, selected], metadata)
    cost_seed = _seed_means(depths[selected], metadata)
    records = []
    for sampled in bootstrap_ids:
        sel = sel_seed[sampled].mean(axis=0)
        tst = test_seed[sampled].mean(axis=0)
        oracle = float(oracle_seed[sampled].mean())
        cost = float(cost_seed[sampled].mean())
        fixed = int(np.argmin(sel + penalty * depths))
        mix = matched_mixture(depths, sel, cost)
        test_hull = matched_mixture(depths, tst, cost)
        null = float(null_seed_losses[:, sampled].mean())
        records.append([oracle, float(tst[fixed] - oracle + penalty * (depths[fixed] - cost)),
                        float(mix["weights"] @ tst - oracle),
                        float(test_hull["weights"] @ tst - oracle), null - oracle])
    names = ["oracle_loss", "oracle_minus_fixed_utility", "oracle_minus_mixture",
             "oracle_minus_test_hull", "oracle_minus_shuffled"]
    quantiles = np.quantile(records, [0.025, 0.975], axis=0)
    return {f"{name}_{tail}": float(quantiles[k, j]) for j, name in enumerate(names)
            for k, tail in enumerate(["ci_low", "ci_high"])}


def _fixed_summaries(checkpoints: pd.DataFrame, depths: np.ndarray,
                     bootstrap_reps: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for frozen in sorted(checkpoints.beta_frozen.unique()):
        group = checkpoints[checkpoints.beta_frozen == frozen]
        for phase in ["all", *[p for p in PHASES if p in set(group.phase)]]:
            phase_data = group if phase == "all" else group[group.phase == phase]
            for depth in depths:
                frame = phase_data[(phase_data.planning_depth == depth) &
                                   (phase_data.rollout_split == "test")].reset_index(drop=True)
                weights = checkpoint_weights(frame)
                seeds = frame.base_seed.nunique()
                bootstrap = (rng.integers(0, seeds, size=(bootstrap_reps, seeds))
                             if seeds >= 2 and bootstrap_reps > 0 else None)
                row = {"beta_frozen": bool(frozen), "phase": phase, "planning_depth": int(depth),
                       "planning_cost": int(depth), "n_checkpoints": len(frame), "n_seeds": seeds,
                       "median_checkpoint_mean_episode_length": float(frame.mean_episode_length.median())}
                for metric in METRICS:
                    column = f"mean_{metric}"
                    if column not in frame:
                        continue
                    valid = frame[column].notna().to_numpy()
                    row[f"mean_{metric}"] = (float(np.average(frame.loc[valid, column], weights=weights[valid]))
                                              if valid.any() else np.nan)
                    if valid.all():
                        low, high = _ci(_seed_means(frame[column].to_numpy(), frame), bootstrap)
                        row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = low, high
                rows.append(row)
    return pd.DataFrame(rows)


def _h1_h3(assignments: pd.DataFrame, checkpoints: pd.DataFrame,
            depths: np.ndarray, bootstrap_reps: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (frozen, utility, penalty), group in assignments.groupby(["beta_frozen", "utility", "lambda"]):
        early = group[group.phase == "H1"]
        late = group[group.phase == "H3"]
        common = sorted(set(early.base_seed) & set(late.base_seed))
        if not common:
            continue
        phase_stats = []
        for phase_frame in [early, late]:
            phase_frame = phase_frame[phase_frame.base_seed.isin(common)].sort_values("base_seed")
            chosen = phase_frame.selected_N.to_numpy()
            threshold = 25 if (depths >= 25).any() else float(depths.max())
            stat = np.column_stack([chosen, chosen >= threshold,
                                    *[chosen == depth for depth in depths]])
            phase_stats.append(_seed_means(stat, phase_frame))
        names = ["mean_selected_N", "probability_N_at_least_25" if threshold == 25 else "probability_max_N",
                 *[f"probability_N_{int(n)}" for n in depths]]
        bootstrap = (rng.integers(0, len(common), size=(bootstrap_reps, len(common)))
                     if len(common) >= 2 and bootstrap_reps > 0 else None)
        for idx, name in enumerate(names):
            diff = phase_stats[1][:, idx] - phase_stats[0][:, idx]
            low, high = _ci(diff, bootstrap)
            rows.append({"beta_frozen": bool(frozen), "utility": utility, "lambda": penalty,
                         "statistic": name, "H1": float(phase_stats[0][:, idx].mean()),
                         "H3": float(phase_stats[1][:, idx].mean()),
                         "H3_minus_H1": float(diff.mean()), "ci_low": low, "ci_high": high,
                         "n_paired_seeds": len(common)})
        # This response interaction holds maze regime fixed and does not depend on oracle selection.
        metric = LOSSES[utility]
        response = []
        for phase in ["H1", "H3"]:
            frame = checkpoints[(checkpoints.beta_frozen == frozen) & (checkpoints.phase == phase) &
                                (checkpoints.rollout_split == "test") & checkpoints.base_seed.isin(common)]
            meta, _, tst = _make_panel(checkpoints[checkpoints.checkpoint_id.isin(frame.checkpoint_id)],
                                       frozen, depths, metric)
            meta_order = np.argsort(meta.base_seed.to_numpy(), kind="stable")
            response.append(_seed_means((tst[:, -1] - tst[:, 0])[meta_order], meta.iloc[meta_order]))
        if len(response[0]) == len(response[1]):
            diff = response[1] - response[0]
            low, high = _ci(diff, bootstrap)
            rows.append({"beta_frozen": bool(frozen), "utility": utility, "lambda": penalty,
                         "statistic": "max_minus_min_depth_loss_interaction", "H1": float(response[0].mean()),
                         "H3": float(response[1].mean()), "H3_minus_H1": float(diff.mean()),
                         "ci_low": low, "ci_high": high, "n_paired_seeds": len(common)})
    return pd.DataFrame(rows)


def _beta_summaries(checkpoints: pd.DataFrame, depths: np.ndarray, bootstrap_reps: int,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Paired frozen-minus-normal TEST contrasts, including the depth-response interaction."""
    if set(checkpoints.beta_frozen) != {False, True}:
        return pd.DataFrame()
    test = checkpoints[checkpoints.rollout_split == "test"]
    rows = []
    for phase in ["all", *[p for p in PHASES if p in set(test.phase)]]:
        at_phase = test if phase == "all" else test[test.phase == phase]
        paired_by_depth = {}
        for depth in depths:
            frame = at_phase[at_phase.planning_depth == depth]
            normal = frame[frame.beta_frozen == False].set_index("checkpoint_id")
            frozen = frame[frame.beta_frozen == True].set_index("checkpoint_id")
            shared = normal.index.intersection(frozen.index).sort_values()
            if not len(shared):
                continue
            normal, frozen = normal.loc[shared], frozen.loc[shared]
            meta = normal.reset_index()
            weights = checkpoint_weights(meta)
            difference = frozen.mean_capped_episode_length.to_numpy() - normal.mean_capped_episode_length.to_numpy()
            seed_differences = _seed_means(difference, meta)
            count = len(seed_differences)
            bootstrap = (rng.integers(0, count, size=(bootstrap_reps, count))
                         if count >= 2 and bootstrap_reps > 0 else None)
            low, high = _ci(seed_differences, bootstrap)
            rows.append({"phase": phase, "planning_depth": int(depth),
                "normal_mean_capped_episode_length": float(weights @ normal.mean_capped_episode_length),
                "frozen_mean_capped_episode_length": float(weights @ frozen.mean_capped_episode_length),
                "frozen_minus_normal_length": float(weights @ difference),
                "frozen_minus_normal_ci_low": low, "frozen_minus_normal_ci_high": high,
                "normal_success_rate": float(weights @ normal.mean_success),
                "frozen_success_rate": float(weights @ frozen.mean_success),
                "n_paired_checkpoints": len(shared), "n_paired_seeds": count})
            paired_by_depth[int(depth)] = (meta, difference)
        # A separate interaction distinguishes a general freezing shift from a changed depth effect.
        if int(depths.min()) in paired_by_depth and int(depths.max()) in paired_by_depth:
            low_meta, low_delta = paired_by_depth[int(depths.min())]
            high_meta, high_delta = paired_by_depth[int(depths.max())]
            low_frame = low_meta[["checkpoint_id", "base_seed", "generator_N", "phase"]].copy()
            low_frame["low_delta"] = low_delta
            high_frame = high_meta[["checkpoint_id"]].copy()
            high_frame["high_delta"] = high_delta
            paired = low_frame.merge(high_frame, on="checkpoint_id", validate="one_to_one")
            values = paired.high_delta.to_numpy() - paired.low_delta.to_numpy()
            per_seed = _seed_means(values, paired)
            count = len(per_seed)
            bootstrap = (rng.integers(0, count, size=(bootstrap_reps, count))
                         if count >= 2 and bootstrap_reps > 0 else None)
            low, high = _ci(per_seed, bootstrap)
            interaction = float(checkpoint_weights(paired) @ values)
            for row in rows:
                if row["phase"] == phase:
                    row.update({"frozen_minus_normal_max_minus_min_depth_interaction": interaction,
                                "depth_interaction_ci_low": low, "depth_interaction_ci_high": high})
    return pd.DataFrame(rows)


def analyse(raw_path: str | Path, output_dir: str | Path, config: dict) -> dict:
    """Write held-out scientific tables and return serializable analysis metadata.

    Positive oracle-minus-comparator differences mean better navigation utility.
    Primary uncertainty uses a training-seed cluster bootstrap; a one-seed run
    has unavailable confidence intervals, rather than misleading zero-width bars.
    Assignment permutations retain seed membership and exact weighted compute.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    depths = np.asarray(sorted(set(config.get("candidate_N", [5, 10, 25, 50]))), dtype=int)
    penalties = sorted(set(float(v) for v in config.get("lambda_grid", [0, .1, 1, 10, 100])))
    if not penalties or penalties[0] != 0 or any(v < 0 or not np.isfinite(v) for v in penalties):
        raise ValueError("lambda_grid must contain zero and finite nonnegative penalties")
    permutations = int(config.get("shuffled_permutations", 500))
    bootstrap_reps = int(config.get("bootstrap_reps", 500))
    if permutations < 1 or bootstrap_reps < 0:
        raise ValueError("Use at least one shuffle and a nonnegative bootstrap count")
    rng = np.random.default_rng(int(config.get("analysis_seed", 982451653)))
    raw = _validate_raw(pd.read_csv(raw_path), depths.tolist())
    if "smoke_only" in raw and raw.smoke_only.map(_bool).any():
        raise ValueError("Smoke-only fixture rows cannot be used for scientific analysis")
    checkpoints = _summarise_checkpoints(raw)
    _write_table(checkpoints, output, "checkpoint_summary.csv")
    geometry_columns = [c for c in ["phase", "environment_version", "start_state", "optimal_path_length"]
                        if c in raw]
    geometry = raw.drop_duplicates("checkpoint_id")
    geometry = geometry.groupby(geometry_columns, dropna=False).agg(
        n_checkpoints=("checkpoint_id", "size"), n_seeds=("base_seed", "nunique")
    ).reset_index()
    _write_table(geometry, output, "geometry_summary.csv")
    fixed = _fixed_summaries(checkpoints, depths, bootstrap_reps, rng)
    _write_table(fixed, output, "fixed_depth_summary.csv")
    assignment_rows, frontier_rows, shuffle_rows, null_rows, mixture_rows, distribution_rows = [], [], [], [], [], []
    exclusions = []

    for frozen in sorted(checkpoints.beta_frozen.unique()):
        for utility, loss_column in LOSSES.items():
            if f"mean_{loss_column}" not in checkpoints:
                continue
            meta, selection, test = _make_panel(checkpoints, frozen, depths, loss_column)
            excluded = checkpoints[checkpoints.beta_frozen == frozen].checkpoint_id.nunique() - len(meta)
            exclusions.append({"beta_frozen": bool(frozen), "utility": utility, "excluded_checkpoints": int(excluded)})
            if len(meta) == 0:
                continue
            weights = checkpoint_weights(meta)
            n_seeds = meta.base_seed.nunique()
            bootstrap = (rng.integers(0, n_seeds, size=(bootstrap_reps, n_seeds))
                         if n_seeds >= 2 and bootstrap_reps > 0 else None)
            selection_mean = weights @ selection
            test_mean = weights @ test
            row_ids = np.arange(len(meta))
            for target in np.unique(np.r_[depths, np.linspace(depths.min(), depths.max(), 91)]):
                mixture = matched_mixture(depths, selection_mean, float(target))
                test_hull = matched_mixture(depths, test_mean, float(target))
                mixture_rows.append({"beta_frozen": bool(frozen), "utility": utility,
                    "planning_cost": float(target), "selection_fit_test_loss": float(mixture["weights"] @ test_mean),
                    "selection_fit_selection_loss": mixture["selection_loss"],
                    "lower_N": mixture["lower_N"], "upper_N": mixture["upper_N"],
                    "upper_weight": mixture["upper_weight"],
                    "optimistic_test_hull_loss": float(test_hull["weights"] @ test_mean)})

            for penalty in penalties:
                selected = np.argmin(selection + penalty * depths[None, :], axis=1)
                selected_depth = depths[selected]
                oracle_values = test[row_ids, selected]
                oracle_loss = float(weights @ oracle_values)
                oracle_cost = float(weights @ selected_depth)
                fixed_index = int(np.argmin(selection_mean + penalty * depths))
                mixture = matched_mixture(depths, selection_mean, oracle_cost)
                mixture_loss = float(mixture["weights"] @ test_mean)
                test_hull = matched_mixture(depths, test_mean, oracle_cost)
                hull_loss = float(test_hull["weights"] @ test_mean)
                null_seed_losses = []
                null_losses = []
                for permutation in range(permutations):
                    shuffled = _shuffle_assignments(selected, meta, rng)
                    shuffled_cost = float(weights @ depths[shuffled])
                    assert np.isclose(shuffled_cost, oracle_cost, atol=1e-10, rtol=0)
                    losses = test[row_ids, shuffled]
                    loss = float(weights @ losses)
                    null_losses.append(loss)
                    null_seed_losses.append(_seed_means(losses, meta))
                    null_rows.append({"beta_frozen": bool(frozen), "utility": utility, "lambda": penalty,
                                      "permutation": permutation, "planning_cost": shuffled_cost,
                                      "test_loss": loss, "oracle_minus_shuffled": loss - oracle_loss})
                null_losses = np.asarray(null_losses)
                null_seed_losses = np.asarray(null_seed_losses)
                delta_null = null_losses - oracle_loss
                confidence = _bootstrap_contrasts(selection, test, selected, meta, depths, penalty,
                                                   null_seed_losses, bootstrap)
                row = {"beta_frozen": bool(frozen), "utility": utility, "lambda": penalty,
                       "n_checkpoints": len(meta), "n_seeds": n_seeds,
                       "oracle_cost": oracle_cost, "oracle_test_loss": oracle_loss,
                       "oracle_test_performance": -oracle_loss,
                       "oracle_test_utility": -oracle_loss - penalty * oracle_cost,
                       "selection_selected_fixed_N": int(depths[fixed_index]),
                       "fixed_test_loss": float(test_mean[fixed_index]),
                       "fixed_test_utility": float(-test_mean[fixed_index] - penalty * depths[fixed_index]),
                       "oracle_minus_fixed_performance": float(test_mean[fixed_index] - oracle_loss),
                       "oracle_minus_fixed_utility": float(test_mean[fixed_index] - oracle_loss +
                                                            penalty * (depths[fixed_index] - oracle_cost)),
                       "oracle_minus_best_test_fixed_performance_descriptive": float(test_mean.min() - oracle_loss),
                       "mixture_test_loss": mixture_loss, "mixture_lower_N": mixture["lower_N"],
                       "mixture_upper_N": mixture["upper_N"], "mixture_upper_weight": mixture["upper_weight"],
                       "oracle_minus_mixture": mixture_loss - oracle_loss,
                       "optimistic_test_hull_loss_descriptive": hull_loss,
                       "oracle_minus_test_hull": hull_loss - oracle_loss,
                       "shuffled_test_loss_mean": float(null_losses.mean()),
                       "oracle_minus_shuffled": float(delta_null.mean()),
                       "selection_jensen_headroom_diagnostic": float(
                           np.min(selection_mean + penalty * depths) -
                           weights @ np.min(selection + penalty * depths[None, :], axis=1)),
                       **confidence}
                frontier_rows.append(row)
                low, high = np.quantile(delta_null, [.025, .975])
                shuffle_rows.append({"beta_frozen": bool(frozen), "utility": utility, "lambda": penalty,
                    "planning_cost": oracle_cost, "oracle_test_loss": oracle_loss,
                    "shuffled_test_loss_mean": float(null_losses.mean()),
                    "shuffled_test_loss_sd": float(null_losses.std(ddof=1)) if permutations > 1 else np.nan,
                    "oracle_minus_shuffled": float(delta_null.mean()),
                    "permutation_delta_p025": float(low), "permutation_delta_p975": float(high),
                    "permutation_fraction_at_least_oracle": float((1 + np.count_nonzero(null_losses <= oracle_loss)) /
                                                                   (permutations + 1)),
                    "seed_bootstrap_ci_low": confidence["oracle_minus_shuffled_ci_low"],
                    "seed_bootstrap_ci_high": confidence["oracle_minus_shuffled_ci_high"],
                    "n_permutations": permutations, "n_seeds": n_seeds})
                for idx, context in meta.iterrows():
                    assignment_rows.append({**context.to_dict(), "beta_frozen": bool(frozen), "utility": utility,
                        "lambda": penalty, "selected_N": int(selected_depth[idx]),
                        "checkpoint_weight": float(weights[idx]),
                        "selection_loss": float(selection[idx, selected[idx]]),
                        "selection_utility": float(-selection[idx, selected[idx]] - penalty * selected_depth[idx]),
                        "test_loss": float(oracle_values[idx]), "test_performance": float(-oracle_values[idx]),
                        "test_utility": float(-oracle_values[idx] - penalty * selected_depth[idx])})
                for phase in ["all", *[p for p in PHASES if p in set(meta.phase)]]:
                    ids = np.arange(len(meta)) if phase == "all" else np.flatnonzero(meta.phase.to_numpy() == phase)
                    phase_weights = checkpoint_weights(meta.iloc[ids])
                    for depth in depths:
                        matches = selected_depth[ids] == depth
                        distribution_rows.append({"beta_frozen": bool(frozen), "utility": utility, "lambda": penalty,
                            "phase": phase, "planning_depth": int(depth), "probability": float(phase_weights @ matches),
                            "checkpoint_count": int(matches.sum()), "n_checkpoints": len(ids),
                            "n_seeds": meta.iloc[ids].base_seed.nunique()})

    assignments = pd.DataFrame(assignment_rows)
    frontier = pd.DataFrame(frontier_rows)
    _write_table(assignments, output, "oracle_assignments.csv")
    _write_table(frontier, output, "frontier_summary.csv")
    _write_table(pd.DataFrame(shuffle_rows), output, "shuffled_oracle_summary.csv")
    _write_table(pd.DataFrame(null_rows), output, "shuffled_oracle_permutations.csv")
    _write_table(pd.DataFrame(mixture_rows), output, "context_blind_mixture_frontier.csv")
    _write_table(pd.DataFrame(distribution_rows), output, "optimal_depth_distribution.csv")
    _write_table(_h1_h3(assignments, checkpoints, depths, bootstrap_reps, rng), output, "h1_h3_summary.csv")

    _write_table(_beta_summaries(checkpoints, depths, bootstrap_reps, rng), output, "beta_frozen_summary.csv")
    timing = raw.groupby("planning_depth").planning_walltime_seconds.agg(["mean", "median", "std", "count"])
    _write_table(timing.reset_index(), output, "runtime_scaling.csv")
    timing_slope = float(np.polyfit(timing.index, timing["mean"], 1)[0]) if len(timing) >= 2 else np.nan
    primary = frontier[(frontier.utility == "capped_episode_length") & (frontier.beta_frozen == False)]
    summary = {
        **EXPERIMENT_A_SETUP,
        "experiment": "A_fresh_replanning_depth",
        "mutation_threshold": 300, "first_hard_episode": 301,
        "mutation_relative_index": "episode_idx - first_hard_episode",
        "comparison_setup": "All fixed-depth, oracle, shuffled-oracle, and context-blind outcomes use the same forced-fresh evaluation grid; this experiment does not compare cache reuse versus replanning.",
        "n_raw_rollouts": len(raw), "n_base_seeds": raw.base_seed.nunique(),
        "n_checkpoints": raw.checkpoint_id.nunique(), "candidate_N": depths.tolist(),
        "lambda_grid": penalties, "bootstrap_reps": bootstrap_reps,
        "shuffled_permutations": permutations, "analysis_seed": config.get("analysis_seed", 982451653),
        "weighting": "Equal seed; equal generator trajectory within seed; equal phase within trajectory; equal checkpoint within phase; rollout means first.",
        "uncertainty": "Training-seed cluster percentile bootstrap; CI unavailable with fewer than two seeds. Selection-fitted baselines are refitted on bootstrap seeds.",
        "shuffle": "Within base_seed and equal checkpoint-weight strata, across phases; preserves selected-depth multiset and exact weighted mean compute.",
        "tie_break": "Exact utility ties select lower N (candidates sorted ascending). Mixture ties use first feasible increasing-depth pair.",
        "selection_test_separation": "Oracle assignments, fixed comparator, and primary matched-cost mixture use selection losses only. All reported task performance uses independent TEST means.",
        "test_hull": "Optimistic descriptive benchmark fitted to aggregate TEST losses, not an independently evaluated learned baseline.",
        "normalized_utility": "-mean(navigation_regret)-lambda*N; lambda units differ from primary capped-steps utility.",
        "selection_jensen_headroom": "In-sample diagnostic only; never interpreted as held-out oracle performance.",
        "planning_runtime_linear_slope_seconds_per_N": timing_slope,
        "excluded_nonfinite_utility_checkpoints": exclusions,
        "primary_frontier": primary.to_dict(orient="records"),
        "provisional_interpretation": "Assess held-out gains, same-hard-maze H1/H3 differences, and seed reproducibility jointly; this oracle experiment alone does not establish metacognition.",
    }
    summary = _json_safe(summary)
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
