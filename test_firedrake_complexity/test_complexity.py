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

h = 2 ** log2h

beta = fd.Constant(10.0)

# Set the output path for the results CSV file.
filename_results = Path('~/Desktop/tesi/tesi/test_firedrake_complexity/times.csv').expanduser()
filename_results.parent.mkdir(parents=True, exist_ok=True)

# Write the header only when the file is empty or does not exist yet.
if not filename_results.exists() or filename_results.stat().st_size == 0:
    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['h', 'N', 'assemble_mass', 'assemble_laplacian', 'assemble_system',
                                    'solve_mass', 'solve_laplacian', 'solve_system',
                                    'definition_solver_mass', 'definition_solver_laplacian', 'definition_solver_system'])

# Build test problem on the unit square.
mesh = fd.UnitSquareMesh(int(1/h), int(1/h))
W = fd.FunctionSpace(mesh, "CG", 1)
bcs = [fd.DirichletBC(W, fd.Constant(0.0), (1, 2, 3, 4))]

N = W.dim()

u = fd.TrialFunction(W)
v = fd.TestFunction(W)
uh = fd.Function(W)

non_linear = fd.Constant(1.0)


a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
m = u * v * fd.dx
s = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx + u * v * fd.dx + non_linear * u * v * fd.dx
rhs = v * fd.dx

t2 = time.time()
A = fd.assemble(a, bcs = bcs)
t3 = time.time()
M = fd.assemble(m, bcs = bcs)
t4 = time.time()
S = fd.assemble(s, bcs = bcs)
t5 = time.time()
RHS = fd.assemble(rhs)
t6 = time.time()
time_assemble_mass = t4 - t3
time_assemble_laplacian = t3 - t2
time_assemble_system = t5 - t4
time_assemble_rhs = t6 - t5

t7 = time.time()
solver_laplacian = fd.LinearSolver(A)
t8 = time.time()
solver_mass = fd.LinearSolver(M)
t9 = time.time()
solver_system = fd.LinearSolver(S)
t10 = time.time()
time_definition_solver_laplacian = t8 - t7
time_definition_solver_mass = t9 - t8
time_definition_solver_system = t10 - t9

t11 = time.time()
solver_laplacian.solve(uh, RHS)
t12 = time.time()
solver_mass.solve(uh, RHS)
t13 = time.time()
solver_system.solve(uh, RHS)
t14 = time.time()
time_solve_laplacian = t12 - t11
time_solve_mass = t13 - t12
time_solve_system = t14 - t13

# Save results to CSV file.
with open(filename_results, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([h, N, time_assemble_mass, time_assemble_laplacian, time_assemble_system,
                     time_solve_mass, time_solve_laplacian, time_solve_system,
                     time_definition_solver_mass, time_definition_solver_laplacian, time_definition_solver_system])
