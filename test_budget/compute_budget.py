import csv
import time
import argparse
import sys
from pathlib import Path

# Parse command-line arguments before importing Firedrake so PETSc does not see `--log2h`.
parser = argparse.ArgumentParser(description='Benchmark gradient assembly and step timings', add_help=True)
parser.add_argument('--log2h', type=int, required=True,
                    help='integer value of log2(h). Example: -3 -> h=1/8')
args, remaining_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining_args]
log2h = args.log2h

import numpy as np
import firedrake as fd
import gradients as gradients

h = 2**log2h
beta = 10
tau = 0.2
# Set the output path for the results CSV file.
filename_results = Path('~/Desktop/tesi/tesi/test_budget/result.csv').expanduser()
filename_results.parent.mkdir(parents=True, exist_ok=True)

# Write the header only when the file is empty or does not exist yet.
if not filename_results.exists() or filename_results.stat().st_size == 0:
    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['method', 'adaptivity', 'h', 'N', 'time_assemble', 'time_step'])

# Build test problem on the unit square.
mesh = fd.UnitSquareMesh(int(1/h), int(1/h))
W = fd.FunctionSpace(mesh, "CG", 1)
bcs = [fd.DirichletBC(W, fd.Constant(0.0), (1, 2, 3, 4))]

x = fd.SpatialCoordinate(mesh)
v = 0.5 * (x[0]**2 + x[1]**2)
N = W.dim()

# Random initial condition used consistently across method timings.
pcg = fd.PCG64()#seed=123456789)
rg = fd.RandomGenerator(pcg)
# Draw from a beta distribution to avoid a too-regular initial state.
u0 = rg.beta(W, 1.0, 2.0)

# List of benchmarks to run: 
# (method name that will be written to the CSV, class of the solver that will be used, time step)
benchmarks = [
    ('L2_e', gradients.Gradient_L2_explicit, tau),
    ('L2_s', gradients.Gradient_L2_semimplicit, tau),
    ('H1_e', gradients.Gradient_H1_explicit, tau),
    ('a0_e', gradients.Gradient_a0_explicit, tau),
    ('a0_e', gradients.Gradient_a0_explicit, None), # adaptive time step
    ('az_e', gradients.Gradient_az_explicit, tau),
    ('az_e', gradients.Gradient_az_explicit, None), # adaptive time step
    ('az_s', gradients.Gradient_az_semimplicit, tau),
    ('az_s', gradients.Gradient_az_semimplicit, None), # adaptive time step
]


for method_name, solver_class, tau_value in benchmarks:
    problem = solver_class(W, bcs, beta, v)

    # Warm-up assembly, then time the second assembly and one step.
    problem.assemble_problem(tau_value)
    t0 = time.perf_counter()
    problem.assemble_problem(tau_value)
    time_assemble = time.perf_counter() - t0

    problem.step(u0)
    t1 = time.perf_counter()
    problem.step(u0)
    time_step = time.perf_counter() - t1

    print(f'{method_name}: h: {h}, N: {N}, time assemble: {time_assemble}, time step: {time_step}')

    # Save results to CSV file.
    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        if tau_value is None:
            adaptivity = 1
        else:            
            adaptivity = 0
        writer.writerow([method_name, adaptivity, h, N, time_assemble, time_step])
