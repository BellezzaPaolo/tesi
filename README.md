# Sobolev-Gradient and ParaflowS Experiments for the Stationary GPE

This repository contains numerical experiments for minimizing the Gross-Pitaevskii energy functional with Firedrake.

It compares:
- classical Sobolev-gradient descent variants,
- a multiscale ParaflowS strategy that combines fine and coarse updates,
- different external potentials and parameter regimes.

The code is organized as research scripts (not a packaged library), with CSV/log/plot outputs for post-processing.

## Problem Setting

The scripts solve stationary minimization problems of the form:

$$
E(u) = \frac{1}{2}\int_{\Omega}\left(\frac{1}{2}|\nabla u|^2 + V(x)u^2 + \frac{\beta}{2}|u|^4\right)\,dx,
\qquad \|u\|_{L^2(\Omega)} = 1.
$$

where $\Omega = [-6,6]^2$ in most experiments.

## Main Components

- `gradients.py`: Sobolev-gradient step operators (`L2`, `L2_P`, `H1`, `a0`, `az`, `az_ada`, explicit `L2`).
- `optimizer.py`:
	- `Gradient_Descent`: baseline iterative minimization,
	- `ParaflowS`: fine/coarse correction strategy.
- `main_opt.py`, `main_opt_test_2.py`: main experiment drivers comparing GD and ParaflowS.
- `test_1.py`, `test_2.py`, `test_3.py`: targeted experiment scripts.
- `generate_ground_truth_1.py`, `generate_ground_truth_2.py`, `generate_ground_truth_3.py`: reference runs for selected potentials.
- `define_budget.py` + `Run_budget.sh`: timing/budget studies over mesh resolutions.
- `extract_data.py`, `make_table.py`, `results/make_table.py`, `test/make_conv_table.py`: analysis and table generation from CSV outputs.

## Requirements

Core dependency is Firedrake.

Python packages used across scripts:
- `numpy`
- `matplotlib`
- `pandas`
- `scipy`

Most scripts assume you are running inside a Firedrake-enabled Python environment.

## Setup

1. Activate Firedrake environment (example used in this repo):

```bash
source ~/venv-firedrake/bin/activate
```

2. Install auxiliary Python packages if needed:

```bash
pip install numpy matplotlib pandas scipy
```

## Quick Start

Run a full GD + ParaflowS comparison (harmonic potential case):

```bash
python3 main_opt.py
```

Run alternative configuration used for comparison tables/plots:

```bash
python3 main_opt_test_2.py
```

Generate ground-truth style reference runs:

```bash
python3 generate_ground_truth_1.py
python3 generate_ground_truth_2.py
python3 generate_ground_truth_3.py
```

Run budget/timing campaign over multiple mesh sizes:

```bash
bash Run_budget.sh
```

## Alternative Backend: NGSolve

This repository also contains an NGSolve/Netgen implementation in `NG_solve/`.

Use it for:
- cross-checking convergence behavior with a second FEM stack,
- computational budget/scaling comparisons,
- reproducing the NGSolve-based CSV tables in `results/`.

Entry point documentation:
- `NG_solve/README.md`

Typical commands:

```bash
python3 NG_solve/define_budget_NG.py --log2h -6
python3 NG_solve/test_NG_1.py
python3 NG_solve/test_NG_2.py
```

## Outputs

Depending on script flags, runs produce:
- CSV summaries (for example in `incontro*/`, `results/`, `test/case_test*/`),
- per-run logs (`.log`),
- convergence plots (`.png`),
- LaTeX tables (for example with `make_table.py`).

Many scripts append to existing CSV files, so you may want to back up or clear old outputs before launching a new campaign.

## Notes on Reproducibility

- Mesh and potential definitions are script-specific.
- Several scripts hard-code parameter grids (`beta`, `tau`, `Nf`, `Ng`, tolerances, iteration limits).
- Some random-potential experiments use fixed seeds for reproducibility.

For exact reproducibility, use the same environment, same script parameters, and clean output CSV files before re-running.

## Suggested Workflow

1. Pick an experiment driver (`main_opt.py`, `main_opt_test_2.py`, or `test_*.py`).
2. Edit parameter vectors (mesh size, $\beta$, step sizes, methods, tolerances).
3. Run experiments.
4. Post-process with `extract_data.py` / `make_table.py` / plotting utilities in `test/`.

## Project Structure (High Level)

- `NG_solve/`: alternative timing/solver scripts.
- `old/`: previous implementations kept for reference.
- `results/`: aggregated CSV results and LaTeX tables.
- `test/`: plotting and comparison utilities with case subfolders.
- `graphs*/`, `incontro*/`: experiment outputs and comparative datasets.

## Citation / Academic Use

If you use this code in a thesis or publication, cite your Firedrake version and include script-level parameter settings used in the experiments.