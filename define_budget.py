"""Benchmark assembly and step costs for Sobolev-gradient methods.

This script is used to estimate computational budget across mesh refinements.
It can run one benchmark point (via --log2h) or plot aggregated results
from an existing CSV file (via --plot).
"""

import csv
import time
import argparse
import sys
import numpy as np

def compute_times(log2h):
    # Convert log2(h) to actual mesh size h.
    h = 2**log2h
    beta = 10
    tau = 0.2
    filename_results = './results/Budget_definition_pointwise_all_default.csv'

    import firedrake as fd
    import gradients

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

    # L2 gradient: assemble once as warm-up, then time assemble and one step.
    problem_L2 = gradients.gradient_L2(beta, v, W, bcs, h)


    problem_L2.assemble_problem(u0, tau)
    t0 = time.perf_counter()
    problem_L2.assemble_problem(u0, tau)#, u_ex)
    time_assemble = time.perf_counter() - t0

    problem_L2.step()
    t1 = time.perf_counter()
    problem_L2.step()
    time_step = time.perf_counter() - t1

    print(f'h: {h}, N: {N}, time assemble: {time_assemble}, time step: {time_step}')

    # Append one benchmark row per method.
    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['L2',h, N, time_assemble, time_step])

    # H1 gradient benchmark.
    problem_H1 = gradients.gradient_H1(beta, v, W, bcs, h)

    problem_H1.assemble_problem(u0, tau)
    t0 = time.time()
    problem_H1.assemble_problem(u0, tau)#, u_ex)
    time_assemble = time.time() - t0

    problem_H1.step()
    t1 = time.time()
    problem_H1.step()
    time_step = time.time() - t1
    print(f'h: {h}, N: {N}, time assemble: {time_assemble}, time step: {time_step}')

    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['H1',h, N, time_assemble, time_step])

    # a0 gradient benchmark.
    problem_a0 = gradients.gradient_a0(beta, v, W, bcs, h)
    problem_a0.assemble_problem(u0, tau)
    t0 = time.time()
    problem_a0.assemble_problem(u0, tau)
    time_assemble = time.time() - t0

    problem_a0.step()
    t1 = time.time()
    problem_a0.step()
    time_step = time.time() - t1

    print(f'h: {h}, N: {N}, time assemble: {time_assemble}, time step: {time_step}')

    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['a0',h, N, time_assemble, time_step])

    # az gradient benchmark.
    problem_az = gradients.gradient_az(beta, v, W, bcs, h)
    problem_az.assemble_problem(u0,tau)
    t0 = time.time()
    problem_az.assemble_problem(u0, tau)
    time_assemble = time.time() - t0

    problem_az.step()
    t1 = time.time()
    problem_az.step()
    time_step = time.time() - t1

    print(f'h: {h}, N: {N}, time assemble: {time_assemble}, time step: {time_step}')

    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['az',h, N, time_assemble, time_step])

def plotting():
    # Import plotting dependencies only when needed.
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv('./results/Budget_definition_NG.csv',#'./results/Budget_definition_pointwise.csv',
                     dtype={"name_opt": str, "h": float, "N": int, "time_assemble": float, "time_step": float, "time_step2": float, "time_step3": float})

    # Discard smallest meshes to improve readability in log-log scaling plots.
    discard_first_N = 2

    h = (df["h"].unique())[discard_first_N:]
    methods = df['name_opt'].unique()
    N = (df['N'].unique())[discard_first_N:]
    #N = np.sqrt(N) # since N is the number of dofs, we take the square root to have the number of refinements
    
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

    fig, ax = plt.subplots(1,1, figsize=(14,10))

    # ax[0].set_title('Assemble Time')
    # ax[0].set_yscale('log', base = 2)
    # ax[0].set_xscale('log', base = 2)
    # ax[0].plot(N, N/N[0], 'k--', label='O(N)')
    # ax[0].plot(N, (N/N[0])**2, 'k:', label='O(N^2)')
    # for method, Ta_m in T_assemble:
    #     ax[0].plot(N, Ta_m/Ta_m[0], 'o-', label=method)
    # # ax[0].plot(N,(N/N[0])**1.5, 'k*',label = 'O(N^1.5)')
    # # ax[0].plot(N,(N/N[0])**0.5, 'k.',label = 'O(N^0.5)')
    # ax[0].set_xlabel('N')
    # ax[0].set_ylabel('Time assemble [s]')
    # ax[0].legend()
    # ax[0].grid()

    # Plot step-cost scaling and theoretical O(N), O(N^2) references.
    ax.set_yscale('log', base = 2)
    ax.set_xscale('log', base = 2)
    ax.plot(N, (N/N[0]), 'k--', linewidth=3, markersize=10, label='O(N)')
    ax.plot(N, (N/N[0])**2, 'k:', linewidth=3, markersize=10, label='O(N^2)')
    for method, Ts1_m in T_step:
        if method == 'L2-Sobolev Gradient' or method == 'L2':
            ax.plot(N, Ts1_m/Ts1_m[0], 's-', linewidth=4, markersize=12, label='$L^2$-SGF')
        if method == 'au-Sobolev Gradient' or method == 'az':
            ax.plot(N, Ts1_m/Ts1_m[0], 'o-', linewidth=3, markersize=10, label='$a_u$-SGF')
    # ax[0].plot(N,(N/N[0])**0.75, 'k*',label = 'O(N^0.75)')
    ax.set_xlabel('Number of refinements', fontsize=28, fontweight='bold')
    ax.set_ylabel('Computational time [s]', fontsize=28, fontweight='bold')
    ax.set_title('Computational time per iteration (NGSolve)', fontsize=30, fontweight='bold')
    ax.legend(fontsize=28)#, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.grid()

    plt.show()

parser = argparse.ArgumentParser()
# Exactly one mode is expected:
# - benchmark mode: --log2h <value>
# - plotting mode:  --plot
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
