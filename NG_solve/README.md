# NG_solve

NGSolve/Netgen implementation of Sobolev-gradient experiments for the stationary Gross-Pitaevskii equation.

This subfolder mirrors the Firedrake workflow in the repository and is mainly used for:
- validating behavior in an alternative FEM stack,
- comparing computational cost trends,
- generating CSV results for method comparisons.

## Files

- `gradients_NG.py`
  - Core NGSolve gradient classes.
  - Base class: `GradientsNG`.
  - Implemented methods: `Gradient_L2`, `Gradient_az`.

- `define_budget_NG.py`
  - Budget/scaling benchmark script.
  - Measures assembly and step time for `L2` and `az` methods.

- `test_NG_1.py`
  - Harmonic potential experiment.
  - Sweeps `beta` and `tau` and writes results to `../results/test_NG_1.csv`.

- `test_NG_2.py`
  - Harmonic + optical lattice experiment.
  - Sweeps separate `tau` sets for `L2` and `az`, writes to `../results/test_NG_2.csv`.

## Requirements

- `ngsolve`
- `netgen`
- `numpy`
- `matplotlib`

Run scripts from the repository root (`tesi/`) so relative output paths remain valid.

## Usage

### 1) Computational budget points

Run one mesh level:

```bash
python3 NG_solve/define_budget_NG.py --log2h -6
```

Plot budget scaling:

```bash
python3 NG_solve/define_budget_NG.py --plot
```

### 2) Test campaigns

Harmonic potential:

```bash
python3 NG_solve/test_NG_1.py
```

Optical lattice potential:

```bash
python3 NG_solve/test_NG_2.py
```

## Output files

Scripts append rows to CSV files under `../results/`:
- `Budget_definition_NG.csv`
- `test_NG_1.csv`
- `test_NG_2.csv`

If you need clean runs, remove or archive old CSV files before launching a new campaign.

## Notes

- The scripts enforce an L2-normalized iterate at each optimization step.
- `E_ref` values are hard-coded and should be updated if mesh or model settings change.
- `Gradient_az` uses a nonlinear Riesz projection each iteration, which can have different cost/scaling from `Gradient_L2`.
