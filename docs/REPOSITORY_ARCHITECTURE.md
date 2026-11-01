# Repository Architecture and Provenance

This project deliberately uses **two repositories with different roles**.

## 1. Upstream fork: implementation and reproducibility

Repository:

`SweetiAi/aimmppcl_2023`

This repository is an official GitHub fork of:

`aswinpaul/aimmppcl_2023`

The upstream repository implements the mixed Active Inference model introduced by:

**Paul, A., Isomura, T., & Razi, A. (2024). _On Predictive Planning and Counterfactual Learning in Active Inference_. Entropy, 26(6), 484. https://doi.org/10.3390/e26060484**

The upstream model provides the original:

- DPEFE predictive-planning pathway,
- Counterfactual Learning pathway,
- mixed-model arbitration mechanism,
- state-dependent Beta arbitration variable,
- mutating grid-world environment,
- core model and planner implementation.

These are upstream contributions and are not claimed as original contributions of the adaptive-planning project.

The fork is retained as the **execution environment** for experiments that require the original model. Project-specific experimental code should live on a dedicated branch rather than modifying the preserved upstream `main` branch.

## 2. Companion repository: research contribution and presentation

Repository:

`SweetiAi/adaptive-planning-active-inference`

This repository is the **companion research repository**. Its role is to present the scientific question, independent analyses, curated results, figures, methodological documentation, and the exact contribution boundary of the adaptive-planning extension.

It should not contain a wholesale copy of the original Paul et al. implementation.

## Contribution boundary

### Upstream contribution

Paul et al. introduced the mixed CL/DPEFE architecture and the original mechanism for balancing predictive planning and experience-based action selection.

### This project's extension

This project asks a different higher-level question:

> Given the Paul et al. mixed-model agent, when is additional predictive planning worth computing, and how deep should that planning be?

The main extension is therefore from **action arbitration** to **computational arbitration / meta-control**.

Project-specific contributions include:

- counterfactual intervention on planning depth `N`,
- checkpoint-level cloning of identical learned states,
- paired evaluation of `N ∈ {5, 10, 25, 50}`,
- common-random-number evaluation across planning-depth branches,
- independent selection/test Monte Carlo splits,
- cost-sensitive value-of-computation analysis,
- policy-sensitivity diagnostics,
- grouped out-of-sample tests of candidate internal meta-control signals,
- falsification of the tested checkpoint-level meta-control hypothesis in the screening study.

## Safety policy for repository maintenance

To preserve provenance and avoid accidentally altering or obscuring the upstream model:

1. The fork's `main` branch is treated as preserved upstream code.
2. Adaptive-planning implementation work belongs on a dedicated project branch.
3. No exploratory outputs, local checkpoints, virtual environments, raw Monte Carlo dumps, or personal filesystem paths should be committed.
4. Files copied into the companion repository must be independently reviewed for provenance and dependency assumptions.
5. Any file derived from or tightly coupled to the upstream implementation must include explicit attribution.
6. The companion repository should remain understandable even to a reader who does not inspect the execution fork.

## Citation rule

Whenever the mixed model, CL, DPEFE, Beta arbitration, or the original mutating-grid implementation is described, the upstream paper should be cited explicitly:

**Paul, A., Isomura, T., & Razi, A. (2024). _On Predictive Planning and Counterfactual Learning in Active Inference_. Entropy, 26(6), 484. https://doi.org/10.3390/e26060484**
