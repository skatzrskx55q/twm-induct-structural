# Model card — INDUCT — Structural Induction Pilot

## Intended use
Controlled synthetic world-dynamics research, reproduction, ablation and extensions. Not validated for real-world forecasts, production automation, robotics, financial or safety decisions.

## Design
1,515 learned node parameters, 24 soft conditional trees with leave-one-out selection over eight transitions; frozen E36 edge solver.

## Evaluation
Open L2/L3 H16: conditioned 76/240 and 102/240 vs static 65/240 and 82/240, vs E36 84/240 and 108/240.

## Known limitations
No sealed test labels in package; no superiority over E36.

## Data and weights
Conditioned and static tree checkpoints and the E36 edge-solver checkpoint set are included.

The original protocols define split, features and seeds. See the [release validation](../validation/linux-cpu.md).

## Technical entry points
[code/structural_induction.py](../research/TWM_INDUCT_STRUCTURAL_INDUCTION_2026-11-04/code/structural_induction.py) | [code/structural_rollout.py](../research/TWM_INDUCT_STRUCTURAL_INDUCTION_2026-11-04/code/structural_rollout.py) | [checkpoints/](../research/TWM_INDUCT_STRUCTURAL_INDUCTION_2026-11-04/checkpoints/) | [reference/](../research/TWM_INDUCT_STRUCTURAL_INDUCTION_2026-11-04/reference/) | [results/STATISTICAL_AND_MECHANISM_AUDIT.json](../research/TWM_INDUCT_STRUCTURAL_INDUCTION_2026-11-04/results/STATISTICAL_AND_MECHANISM_AUDIT.json) | [docs/README_REPRODUCTION_RU.md](../research/TWM_INDUCT_STRUCTURAL_INDUCTION_2026-11-04/docs/README_REPRODUCTION_RU.md)

## Evidence status
Historical status: Positive ablation; weaker than E36. Fresh selected check: PASS. See `../validation/current_check.log`. Only the documented research environment is evaluated; there is no warranty of generalization.
