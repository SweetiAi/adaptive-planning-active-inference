# Adaptive Planning in Active Inference

A research extension exploring **when and how much an Active Inference agent should plan**.

## Project status

This repository is currently a **private research workspace**. It is being prepared cautiously before any public release.

The project builds on the mixed Active Inference model introduced by Aswin Paul and collaborators, which combines:

- **Counterfactual Learning (CL)**
- **Dynamic Programming Expected Free Energy (DPEFE)**

The present work studies a higher-level question: whether planning depth should adapt to context, computational cost, and the agent's internal state.

## Main findings so far

The screening study found that:

- planning depth can substantially change policy,
- deeper planning can improve behaviour in some hard-regime checkpoints,
- the value of deeper planning is context-dependent,
- however, the tested pre-planning internal-state representations did not reliably predict when deeper planning was worth its computational cost beyond a coarse phase/regime baseline.

This is therefore treated as a **negative result for the current meta-control hypothesis**, while the experimental framework remains reusable for future work on value of computation, replanning, and explicit meta-control.

## Repository policy

This repository will contain only reviewed project-specific code, tests, documentation, compact summary results, and figures.

Large raw outputs, serialized checkpoints, local environments, temporary files, and exploratory artifacts are intentionally excluded.

## Attribution

The original mixed-model implementation is maintained separately. This repository is intended to contain the independent adaptive-planning research extension and supporting analysis, with clear attribution to the original work.
