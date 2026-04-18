"""Benchmark assembly and step costs for NGSolve gradient implementations.

Use:
- `--log2h <k>` to run one benchmark point,
- `--plot` to visualize scaling from the aggregated CSV output.
"""

import csv
import time
import argparse
import sys

def compute_times(log2h):
    # Convert log2-level to the mesh size convention used in this submodule.
    hmax = 2**(log2h+3)
    order = 1
    dirichlet_bcs = 'outer'
    beta = 10
    potential = 'Harmonic'
    tau = 1
    filename_results = './results/Budget_definition_NG.csv'
    initial_guess = 'random'

    from gradients_NG import Gradient_L2, Gradient_az
    import ngsolve as ng
    from netgen.geom2d import SplineGeometry

    # Build square domain and corresponding NGSolve mesh.
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-6, -6), p2=(6, 6), bc="outer")

    ngmesh = geo.GenerateMesh(maxh=hmax)
    mesh = ng.Mesh(ngmesh)

    grad_L2 = Gradient_L2(beta, potential, hmax, mesh, order, dirichlet_bcs)
    grad_az = Gradient_az(beta, potential, hmax, mesh, order, dirichlet_bcs)

    # Number of degrees of freedom for scaling plots.
    N = grad_az.fes.ndof


    # L2 timing: warm-up assemble/step once, then measure assemble and one step.
    grad_L2.assemble_problem(initial_guess, tau, 0.0)
    t0 = time.perf_counter()
    grad_L2.assemble_problem(initial_guess, tau, 0.0)
    time_assemble = time.perf_counter() - t0

    grad_L2.step()
    t1 = time.perf_counter()
    grad_L2.step()
    time_step = time.perf_counter() - t1

    print(f'h: {hmax}, N: {N}, time assemble: {time_assemble}, time step: {time_step}')

    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['L2',hmax, N, time_assemble, time_step])

    # az timing with the same protocol.
    grad_az.assemble_problem(initial_guess,tau, 0.0)
    t0 = time.time()
    grad_az.assemble_problem(initial_guess, tau, 0.0)
    time_assemble = time.time() - t0

    grad_az.step()
    t1 = time.time()
    grad_az.step()
    time_step = time.time() - t1

    print(f'h: {hmax}, N: {N}, time assemble: {time_assemble}, time step: {time_step}')

    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['az',hmax, N, time_assemble, time_step])

def plotting():
    # Plot benchmark scaling from the CSV file.
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv('../results/Budget_definition_NG.csv',
                     dtype={"name_opt": str, "h": float, "N": int, "time_assemble": float, "time_step": float, "time_step2": float, "time_step3": float})

    # Optionally skip coarsest points if needed for readability.
    discard_first_N = 0

    h = (df["h"].unique())[discard_first_N:]
    methods = df['name_opt'].unique()
    N = (df['N'].unique())[discard_first_N:]
    
    T_assemble = []
    T_step = []

    for method in methods:
        Ta_m = []
        Ts_m = []
        for hi in h:
            df_filtered = df[(df['name_opt'] == method) &
                             (df['h'] == hi)]
            Ta_m.append(df_filtered["time_assemble"].mean())
            Ts_m.append(df_filtered["time_step"].mean())

        T_assemble.append((method,Ta_m))
        T_step.append((method,Ts_m))

    fig, ax = plt.subplots(2,1, figsize=(6,12))

    ax[0].set_title('Assemble Time')
    ax[0].set_yscale('log', base = 2)
    ax[0].set_xscale('log', base = 2)
    ax[0].plot(N, N/N[0], 'k--', label='O(N)')
    ax[0].plot(N, (N/N[0])**2, 'k:', label='O(N^2)')
    for method, Ta_m in T_assemble:
        ax[0].plot(N, Ta_m/Ta_m[0], 'o-', label=method)
    # ax[0].plot(N,(N/N[0])**1.5, 'k*',label = 'O(N^1.5)')
    # ax[0].plot(N,(N/N[0])**0.5, 'k.',label = 'O(N^0.5)')
    ax[0].set_xlabel('N')
    ax[0].set_ylabel('Time assemble [s]')
    ax[0].legend()
    ax[0].grid()

    ax[1].set_title('Time Step')
    ax[1].set_yscale('log', base = 2)
    ax[1].set_xscale('log', base = 2)
    ax[1].plot(N, (N/N[0]), 'k--', label='O(N)')
    ax[1].plot(N, (N/N[0])**2, 'k:', label='O(N^2)')
    for method, Ts1_m in T_step:
        ax[1].plot(N, Ts1_m/Ts1_m[0], 'o-', label=method)
    # ax[1].plot(N,(N/N[0])**0.75, 'k*',label = 'O(N^0.75)')
    ax[1].set_xlabel('N')
    ax[1].set_ylabel('Time step [s]')
    ax[1].legend()
    ax[1].grid()

    plt.show()

parser = argparse.ArgumentParser()
# Exactly one mode is expected: benchmarking OR plotting.
parser.add_argument("--log2h", type=float, help="value of the space discretization parameter h", default=0)
parser.add_argument("--plot", action="store_true", help="whether to plot the solution or not", default=False)

args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

if (not(args.plot) and args.log2h == 0)  or (args.plot and args.log2h != 0):
    print("Error: choose exactly one mode between --plot and --log2h. Exiting.")
    sys.exit(0)
elif args.plot:
    plotting()
else:
    compute_times(args.log2h)
