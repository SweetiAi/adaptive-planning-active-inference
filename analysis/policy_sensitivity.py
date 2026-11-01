"""
Policy-sensitivity analysis for the adaptive planning-depth extension.

Scientific basis:
Paul, A., Isomura, T., & Razi, A. (2024).
On Predictive Planning and Counterfactual Learning in Active Inference.
Entropy, 26(6), 484. https://doi.org/10.3390/e26060484

The DPEFE planner and mixed CL/DPEFE architecture are upstream contributions
of Paul et al. This module implements only project-specific descriptive
comparisons across counterfactual fresh-planning-depth interventions.

Inputs are action-by-state probability arrays only. All modeled state labels
receive equal weight in global summaries, including labels at drawn wall cells.
This module accepts no agent/model references and uses no random numbers.
"""


from __future__ import annotations

from itertools import combinations
from numbers import Integral
from typing import Mapping

import numpy as np

DEPTHS = (5, 10, 25, 50)
NORMALIZATION_ATOL = 1e-12


def _validated_policies(policies: Mapping[int, np.ndarray], start_state: int):
    if not isinstance(policies, Mapping):
        raise ValueError("policies must map each of depths 5, 10, 25, 50 to an array")
    if any(isinstance(depth, (bool, np.bool_)) or not isinstance(depth, Integral) for depth in policies):
        raise ValueError("Policy depth keys must be integers")
    if set(policies) != set(DEPTHS):
        raise ValueError("Policies must contain exactly depths 5, 10, 25, 50")
    if isinstance(start_state, (bool, np.bool_)) or not isinstance(start_state, Integral):
        raise ValueError("start_state must be an integer")
    start_state = int(start_state)
    arrays = {}
    shape = None
    for depth in DEPTHS:
        value = policies[depth]
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
            raise ValueError(f"Depth {depth} policy must be a real numeric NumPy array")
        if value.ndim != 2 or value.shape[0] != 4 or value.shape[1] < 1:
            raise ValueError(f"Depth {depth} policy must have shape (4, S) with S >= 1")
        if shape is not None and value.shape != shape:
            raise ValueError("All depth policies must have the same state dimension")
        shape = value.shape
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"Depth {depth} policy must be finite and nonnegative")
        # Work on independent float64 copies. Validation never normalizes or
        # changes the input probabilities, even when they are writable views.
        array = np.array(value, dtype=float, copy=True)
        if not np.isfinite(array).all() or not np.allclose(
                array.sum(axis=0), 1.0, atol=NORMALIZATION_ATOL, rtol=0):
            raise ValueError(f"Depth {depth} policy columns must sum to one within 1e-12")
        arrays[depth] = array
    if not 0 <= start_state < shape[1]:
        raise ValueError("start_state lies outside the policy state dimension")
    return arrays, start_state


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities > 0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


def _jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    # KL(p || (p+q)/2) = sum p*log(2p/(p+q)). Masking zeros avoids log(0),
    # and this form avoids underflow when halving a positive subnormal mass.
    combined = first + second
    positive_first, positive_second = first > 0, second > 0
    first_kl = np.sum(first[positive_first] * np.log(
        (2 * first[positive_first]) / combined[positive_first]))
    second_kl = np.sum(second[positive_second] * np.log(
        (2 * second[positive_second]) / combined[positive_second]))
    # Roundoff can produce a negative value near zero; JS is nonnegative.
    return max(0.0, float((first_kl + second_kl) / 2))


def policy_entropy_start(policies: Mapping[int, np.ndarray], start_state: int) -> dict[int, float]:
    """Return per-depth Shannon entropy at the start state, in natural-log nats."""
    arrays, start = _validated_policies(policies, start_state)
    return {depth: _entropy(arrays[depth][:, start]) for depth in DEPTHS}


def compare_depth_policies(policies: Mapping[int, np.ndarray], start_state: int) -> list[dict]:
    """Return six increasing-depth pair summaries of the original 4-by-S policies.

    TV is half the action-wise L1 distance. JS is Jensen-Shannon divergence in
    nats. Global TV and argmax statistics use every modeled state column, with
    no visitation, reachability or wall filtering. NumPy's first-maximum rule
    resolves exact argmax ties to the lowest action index.
    """
    arrays, start = _validated_policies(policies, start_state)
    entropies = {depth: _entropy(arrays[depth][:, start]) for depth in DEPTHS}
    rows = []
    for depth_a, depth_b in combinations(DEPTHS, 2):
        first, second = arrays[depth_a], arrays[depth_b]
        state_tv = np.abs(first - second).sum(axis=0) / 2
        argmax_diff = np.argmax(first, axis=0) != np.argmax(second, axis=0)
        rows.append({
            "depth_a": depth_a, "depth_b": depth_b,
            "start_state_TV": float(state_tv[start]),
            "start_state_JS": _jensen_shannon(first[:, start], second[:, start]),
            "mean_state_TV": float(state_tv.mean()),
            "max_state_TV": float(state_tv.max()),
            "fraction_states_argmax_diff": float(argmax_diff.mean()),
            "start_state_argmax_diff": bool(argmax_diff[start]),
            "H_start_a": entropies[depth_a], "H_start_b": entropies[depth_b],
        })
    return rows
