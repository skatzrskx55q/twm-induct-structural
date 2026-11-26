# Quick start

Use a Linux x86-64 CPU environment and Python 3.13 (verified with 3.13.5). Allow at least 1 GiB of working space for the extracted repository and run copies, plus Python and dependency space. The commands run without service credentials.

```bash
git clone https://github.com/skatzrskx55q/twm-induct-structural.git
cd twm-induct-structural
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python verify_integrity.py
python run.py check
python run.py --keep ../twm-induct-structural-demo demo
```

Use a fresh path in `--keep` to preserve results. You can also extract the separately attached [research ZIP](https://github.com/skatzrskx55q/twm-induct-structural/releases/tag/v0.1.0-research) and start at virtual environment setup.

Recomputes synthetic graph rollouts for the conditioned, static, ablation, E36 and random variants.

Selected checks exercise the documented native entry points. See [the validation record](validation/linux-cpu.md) and [original guide](research/TWM_INDUCT_STRUCTURAL_INDUCTION_2026-11-04/docs/README_REPRODUCTION_RU.md).
