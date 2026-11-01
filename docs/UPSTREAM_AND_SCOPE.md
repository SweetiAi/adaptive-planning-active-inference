# Upstream provenance and project scope

## Why this project exists

The starting point for this work is the mixed Active Inference model of Paul, Isomura, and Razi (2024), which combines a predictive-planning route (DPEFE) with a Counterfactual Learning route and uses a state-dependent arbitration mechanism to balance them.

That architecture motivated a new question:

> If an agent can arbitrate between planning and experience once both action distributions are available, can it also control the computation itself — deciding when fresh planning is worth performing and how deep that planning should be?

This repository is dedicated to that extension.

## Upstream work

The following are treated as upstream contributions and are not claimed as original contributions here:

- DPEFE as implemented in the original mixed model,
- Counterfactual Learning in the original mixed model,
- the mixed CL/DPEFE action-selection architecture,
- `Beta` as the state-dependent arbitration quantity,
- the mutating grid-world task and canonical transition setup,
- the original agent training and DPEFE planning code.

Primary reference:

**Paul, A., Isomura, T., & Razi, A. (2024). _On Predictive Planning and Counterfactual Learning in Active Inference_. Entropy, 26(6), 484. https://doi.org/10.3390/e26060484**

Original implementation: **Aswin Paul — `aswinpaul/aimmppcl_2023`**.

## Independent extension developed here

The independent research contribution begins at the computational-control layer above the upstream mixed model.

The project asks how planning depth should be selected under an explicit computational cost. The main experimental intervention compares fresh planning depths

```text
N = 5, 10, 25, 50
```

from the same pre-planning checkpoint, so that the learned agent state is held fixed while planning depth is changed counterfactually.

The project-specific methodology includes:

- exact pre-planning checkpointing and cloning,
- depth-only counterfactual interventions,
- common random numbers across branches,
- independent selection/test Monte Carlo splits,
- memoization of deterministic fresh planning across rollouts,
- cost-sensitive Oracle analysis,
- policy-distance and behavioural-sensitivity analyses,
- grouped generalization tests for internal meta-control signals,
- falsification criteria and explicit stopping rules.

## Current scientific conclusion

The screening study supports the claim that planning depth can materially alter policy and behaviour and that its value is context-dependent.

However, the tested pre-planning internal-state representations did not reliably predict when deeper planning was worth its cost beyond a coarse phase/regime baseline under the strictest generalization test.

The current checkpoint-level meta-control hypothesis is therefore treated as unsupported in this dataset.

## Repository boundary

This repository should not be interpreted as a reimplementation or replacement of the Paul et al. model.

The original model is maintained separately. Code placed here should either:

1. be independent analysis/documentation developed for this extension, or
2. clearly declare any dependency on the upstream model.

Raw copies of the original mixed-model source are intentionally not included by default.
