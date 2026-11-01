# Adaptive Planning in Active Inference

A research extension exploring **when and how much an Active Inference agent should plan**.

## Project status

This repository is currently a **private research workspace**. It is being prepared cautiously before any public release.

## Upstream scientific basis

This project is explicitly based on the mixed Active Inference architecture introduced by:

> **Paul, A., Isomura, T., & Razi, A. (2024). _On Predictive Planning and Counterfactual Learning in Active Inference_. Entropy, 26(6), 484. https://doi.org/10.3390/e26060484**

Original implementation:

> **Aswin Paul — `aswinpaul/aimmppcl_2023`**

The original paper and code introduced and implemented the key ingredients on which this project depends, including:

- **Dynamic Programming Expected Free Energy (DPEFE)** as the predictive-planning pathway,
- **Counterfactual Learning (CL)** as the experience-based pathway,
- the **mixed-model arbitration mechanism** between DPEFE and CL,
- the state-dependent arbitration quantity **Beta**,
- the adaptive / mutating grid-world used to study the balance between planning and learning from experience.

These ideas and the original mixed-model implementation are **not contributions of this repository**.

## What this project adds

The present project began from a question motivated by Paul et al.'s mixed architecture:

> If an agent already has a mechanism for choosing how much to rely on predictive planning versus experience, can it also determine **when additional planning is worth computing, and how deep that planning should be?**

The extension therefore shifts the problem from **action arbitration** to **computational arbitration / meta-control**.

In particular, this project studies counterfactual interventions on planning depth:

```text
N ∈ {5, 10, 25, 50}
```

while holding the learned agent state fixed at a checkpoint, and asks whether the expected behavioural benefit of deeper planning justifies its additional computational cost.

This value-of-computation / adaptive-planning question, the counterfactual planning-depth evaluation framework, and the associated screening analyses are the independent extension developed in this project.

## Main findings so far

The screening study found that:

- planning depth can substantially change policy,
- deeper planning can improve behaviour in some hard-regime checkpoints,
- the value of deeper planning is context-dependent,
- however, the tested pre-planning internal-state representations did not reliably predict when deeper planning was worth its computational cost beyond a coarse phase/regime baseline.

This is therefore treated as a **negative result for the current meta-control hypothesis**, while the experimental framework remains reusable for future work on value of computation, replanning, and explicit meta-control.

## Repository policy

This repository will contain only reviewed project-specific code, tests, documentation, compact summary results, and figures.

The original Paul et al. mixed-model source is **not being copied wholesale into this repository**. Where the extension depends on the original implementation, that dependency will be stated explicitly.

Large raw outputs, serialized checkpoints, local environments, temporary files, and exploratory artifacts are intentionally excluded.

## Attribution and citation

Please cite the upstream work when discussing the mixed model, DPEFE/CL architecture, Beta arbitration, or the original mutating-grid implementation:

**Paul, A., Isomura, T., & Razi, A. (2024). _On Predictive Planning and Counterfactual Learning in Active Inference_. Entropy, 26(6), 484. https://doi.org/10.3390/e26060484**

See `CITATION.md` and `docs/UPSTREAM_AND_SCOPE.md` for the exact attribution boundary between the upstream model and this extension.
