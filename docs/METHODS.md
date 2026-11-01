# Methods

## Scientific basis

This project extends the mixed Active Inference architecture introduced by:

**Paul, A., Isomura, T., & Razi, A. (2024). _On Predictive Planning and Counterfactual Learning in Active Inference_. Entropy, 26(6), 484. https://doi.org/10.3390/e26060484**

The original DPEFE planner, Counterfactual Learning pathway, Beta arbitration mechanism, mutating-grid task, and core mixed-model implementation are upstream contributions of Paul et al.

The present project does not claim those components as original contributions.

## Research question

The project asks:

> When, and how much, should an Active Inference agent plan?

The original mixed model already combines predictive planning (DPEFE) with experience-based Counterfactual Learning (CL). This project moves one level upward and studies whether fresh planning effort itself should be adapted to context and computational cost.

## Experiment A: counterfactual planning-depth intervention

Candidate fresh planning depths were:

```text
N ∈ {5, 10, 25, 50}
```

At each selected pre-planning checkpoint:

1. the learned agent state was snapshotted,
2. the checkpoint was cloned across counterfactual branches,
3. fresh DPEFE planning was recomputed independently for each candidate depth,
4. the post-planning policy state was evaluated with matched stochastic conditions where possible,
5. branch outcomes were kept separate from the base training trajectory.

The intended causal contrast is therefore planning depth under an otherwise matched learned state.

## Training and evaluation semantics

The base training trajectory preserves the upstream refresh schedule.

Evaluation branches use a **forced fresh** DPEFE replan at the candidate depth.

The experiment therefore studies **fresh-replanning depth**, not cache reuse versus replanning.

## Candidate phases

Pilot 0 sampled five learning phases:

- `E1`: early easy,
- `E2`: late easy,
- `H1`: early hard,
- `H2`: intermediate hard,
- `H3`: late hard.

`H1` and `H3` use the same hard maze and therefore provide a descriptive same-geometry comparison across different learning contexts.

## Pilot 0 screening design

Pilot 0 used:

- 2 screening base seeds,
- 3 generator trajectories per seed,
- 1 checkpoint per phase per trajectory,
- 30 checkpoints total,
- 4 candidate planning depths,
- 3 selection and 3 TEST rollouts per candidate depth/checkpoint condition,
- 15,000-action episode cap.

The dataset was explicitly designated as **screening** and is not treated as confirmatory evidence.

## Selection and TEST separation

Selection and TEST rollout streams serve different roles.

Selection data may be used to:

- choose the checkpoint-level oracle depth,
- select the global fixed-depth comparator,
- fit the primary matched-cost context-blind mixture.

TEST data are reserved for held-out behavioural evaluation of those choices.

This separation is enforced in `analysis/analysis.py`.

## Primary outcome and censoring

The main behavioural outcome is capped episode length.

Episodes reaching 15,000 actions are right-censored by the experimental cap. Because hard-regime censoring is substantial, success rate, cap rate, and diagnostic goal progress are interpreted alongside capped episode length.

This censoring is one of the main reasons Pilot 0 is reported conservatively.

## Policy sensitivity

Fresh DPEFE policies are compared across planning depths using descriptive metrics including:

- start-state total variation distance,
- start-state Jensen–Shannon divergence,
- mean state-wise total variation,
- maximum state-wise total variation,
- fraction of modeled states with different argmax actions.

These diagnostics test the cheap falsification question:

> Does planning depth actually change the generated policy?

If policies had remained effectively invariant, a larger behavioural study would not have been justified.

## Cost-sensitive analysis

For planning depth `N`, the primary theoretical planning cost is:

```text
C(N) = N
```

A cost-sensitive loss can therefore be written schematically as:

```text
loss = behavioural loss + λN
```

where `λ` controls the trade-off between behavioural performance and planning effort.

Measured wall-clock planning time is retained as an engineering diagnostic rather than substituted for the primary theoretical cost.

## Oracle and matched-compute comparisons

The screening analysis compares:

- fixed planning depths,
- a checkpoint-aware oracle selected from selection data,
- context-blind matched-cost mixtures,
- shuffled-oracle assignments preserving compute allocation.

The oracle is a diagnostic upper-bound style construct, not a deployable controller.

Its role is to ask whether context-dependent allocation could in principle matter before asking whether the agent can infer that allocation from its own internal state.

## Meta-control tests

The project subsequently tested whether cheap pre-planning internal signals could predict when deeper planning was worth its cost.

Candidate signals included policy uncertainty, disagreement between experience-based and cached planning distributions, transition-model confidence, cache age, and richer full-state similarity diagnostics.

Under the strictest grouped out-of-sample evaluation, the tested internal representations did not reliably outperform a coarse phase/regime baseline.

The result is therefore reported as a **negative result for the tested checkpoint-level internal meta-control hypothesis**.

## Interpretation rule

The central scientific distinction is:

```text
context-dependent value of computation ≠ demonstrated metacognitive control
```

The current results show the former but do not establish the latter.

## Data and artifact policy

This companion repository includes only:

- reviewed analysis code,
- curated summary tables,
- curated figures,
- provenance and methods documentation.

Large raw rollout dumps, serialized checkpoints, recovery state, local environments, and exploratory artifacts are intentionally excluded.

Implementation-level reproduction of the upstream mixed-model agent remains tied to the official fork of `aswinpaul/aimmppcl_2023`.
