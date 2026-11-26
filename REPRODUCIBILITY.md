# Reproducibility

1. Install the pinned Python dependencies using [QUICKSTART.md](QUICKSTART.md).
2. Verify the source SHA-256 manifest with `python verify_integrity.py`.
3. Run the frozen native check. The historical command is `python -m pytest -q -p no:cacheprovider tests/test_structural.py`.
4. Run `python run.py --keep ../twm-induct-structural-repro demo` to retain outputs and compare them against [archived results](research/TWM_INDUCT_STRUCTURAL_INDUCTION_2026-11-04/results/STAGE_SUMMARY.json). The direct native demo command is `python code/structural_rollout.py --seeds 17,29,43 --modes conditioned,static,ablate,E36,random`.

## Training or calibration

Train the conditioned and static node modules using the frozen E36 edge solver.

```bash
python run.py --keep ../twm-induct-structural-training native -- code/structural_experiment.py --steps 850 --seeds 17,29,43 --modes conditioned,static
```

[original guide](research/TWM_INDUCT_STRUCTURAL_INDUCTION_2026-11-04/docs/README_REPRODUCTION_RU.md) · [Original release guide](docs/REPRODUCTION.md) · [Release validation](validation/linux-cpu.md)
