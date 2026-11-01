# Pilot 0 Screening Figures

These figures summarize **screening-stage results** from the adaptive planning-depth extension of the mixed Active Inference model introduced by:

Paul, A., Isomura, T., & Razi, A. (2024).
*On Predictive Planning and Counterfactual Learning in Active Inference.*
Entropy, 26(6), 484. https://doi.org/10.3390/e26060484

The original DPEFE planner, Counterfactual Learning pathway, Beta arbitration mechanism, and mutating-grid task are upstream contributions of Paul et al.

These plots visualize project-specific counterfactual interventions on fresh planning depth:

`N ∈ {5, 10, 25, 50}`

They are based on **Pilot 0 screening data** and should not be interpreted as confirmatory evidence.

## Figures

### `policy_TV_by_phase.png`

Pairwise policy sensitivity across planning depths.

The figure shows that fresh planning depth has little effect in the easy regime but can substantially alter the generated policy in the hard regime.

### `success_by_phase_depth.png`

Success rate across learning phases and planning depths.

The hard regime shows clear depth-dependent behavioural differences, particularly during early and intermediate adaptation.

### `censoring_by_phase.png`

Fraction of rollouts reaching the 15,000-action cap.

This figure documents a major limitation of Pilot 0: substantial censoring in the hard regime, especially at shallower planning depths.

### `H1_vs_H3_screening.png`

Descriptive comparison of early-hard (`H1`) and late-hard (`H3`) checkpoints in the same hard maze.

This comparison helps separate changes associated with learning context from simple maze-geometry differences.

## Interpretation boundary

Together, these figures support the screening-level conclusions that:

- planning depth materially changes policy in the hard regime,
- behavioural consequences of planning depth depend on learning context,
- deeper planning can reduce failure/censoring in some hard-regime conditions,
- the effect is not explained solely by maze geometry.

They do **not** establish that:

- the agent possesses metacognition,
- the agent can infer the optimal planning depth from its internal state,
- an adaptive meta-controller generalizes out of sample,
- or Pilot 0 provides confirmatory evidence.

See `../summary/README.md` and the repository-level documentation for the full methodological and attribution context.
