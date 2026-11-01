# Citation and attribution

## Upstream model

This project builds directly on the mixed Active Inference architecture introduced in:

**Paul, A., Isomura, T., & Razi, A. (2024). _On Predictive Planning and Counterfactual Learning in Active Inference_. Entropy, 26(6), 484. https://doi.org/10.3390/e26060484**

Original implementation: **Aswin Paul — `aswinpaul/aimmppcl_2023`**.

The following scientific and implementation components are attributed to Paul et al. and the upstream codebase:

- the DPEFE predictive-planning pathway,
- the Counterfactual Learning (CL) pathway,
- the mixed-model arbitration between DPEFE and CL,
- the state-dependent `Beta` arbitration quantity,
- the mutating/adaptive grid-world task used by the original model,
- the original training and planning implementation on which this extension was evaluated.

These are not claimed as contributions of this repository.

## This extension

This repository studies a separate meta-control question derived from the upstream architecture:

> Given an agent that can already arbitrate between planning and experience, can it also determine when additional planning is worth computing, and how deep that planning should be?

The independent contributions developed in this project are:

- treating planning depth as a counterfactual computational intervention,
- evaluating candidate planning depths from identical pre-planning checkpoints,
- separating selection and test Monte Carlo rollouts,
- cost-sensitive / value-of-computation analysis,
- policy-sensitivity analyses across planning depths,
- grouped out-of-sample tests of whether pre-planning internal state predicts the value of deeper planning,
- the resulting negative finding for the current checkpoint-level meta-control hypothesis.

## Citation practice for this repository

Any document, figure, analysis note, or code module that discusses or depends on the upstream mixed-model architecture should cite Paul et al. (2024) explicitly.

Where code depends on the original implementation, that dependency should be stated in the file header or accompanying documentation rather than obscured.
