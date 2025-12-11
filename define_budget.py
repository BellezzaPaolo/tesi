import csv
import time
import argparse
import sys

def compute_times(log2h):
    h = 2**log2h
    beta = 10
    tau = 0.2
    filename_results = './results/Budget_defintion.csv'

    import firedrake as fd
    import gradients

    mesh = fd.UnitSquareMesh(int(1/h), int(1/h))
    W = fd.FunctionSpace(mesh, "CG", 1)
    bcs = [fd.DirichletBC(W, fd.Constant(0.0), (1, 2, 3, 4))]

    x = fd.SpatialCoordinate(mesh)
    v = 0.5 * (x[0]**2 + x[1]**2)
    N = W.dim()

    # PCG64 random number generator
    pcg = fd.PCG64()#seed=123456789)
    rg = fd.RandomGenerator(pcg)
    # beta distribution
    u0 = rg.beta(W, 1.0, 2.0)

    dummy_problem =gradients.dummy_gradient(beta, v, W, bcs, h)
    dummy_problem.assemble_problem(u0, tau)
    dummy_problem.step()

    problem_L2 = gradients.gradient_L2(beta, v, W, bcs, h)
    # L2 gradient
    t0 = time.time()
    problem_L2.assemble_problem(u0, tau)#, u_ex)
    time_assemble = time.time() - t0

    t0 = time.time()
    problem_L2.step()
    time_step = time.time() - t0

    t0 = time.time()
    problem_L2.step()
    time_step2 = time.time() - t0
    t0 = time.time()
    problem_L2.step()
    time_step3 = time.time() - t0
    print(f'h: {h}, N: {N}, time assemble: {time_assemble}, time step: {time_step}, time step2: {time_step2}, time step3: {time_step3}')

    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['L2',h, N, time_assemble, time_step, time_step2, time_step3])

    problem_H1 = gradients.gradient_H1(beta, v, W, bcs, h)
    # H1 gradient
    t0 = time.time()
    problem_H1.assemble_problem(u0, tau)#, u_ex)
    time_assemble = time.time() - t0

    t0 = time.time()
    problem_H1.step()
    time_step = time.time() - t0
    t0 = time.time()
    problem_H1.step()
    time_step2 = time.time() - t0
    t0 = time.time()
    problem_H1.step()
    time_step3 = time.time() - t0
    print(f'h: {h}, N: {N}, time assemble: {time_assemble}, time step: {time_step}, time step2: {time_step2}, time step3: {time_step3}')

    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['H1',h, N, time_assemble, time_step, time_step2, time_step3])

    # a_0 gradient
    problem_a0 = gradients.gradient_a0(beta, v, W, bcs, h)
    t0 = time.time()
    problem_a0.assemble_problem(u0, tau)#, u_ex)
    time_assemble = time.time() - t0

    t0 = time.time()
    problem_a0.step()
    time_step = time.time() - t0

    t0 = time.time()
    problem_a0.step()
    time_step2 = time.time() - t0


    t0 = time.time()
    problem_a0.step()
    time_step3 = time.time() - t0

    print(f'h: {h}, N: {N}, time assemble: {time_assemble}, time step: {time_step}, time step2: {time_step2}, time step3: {time_step3}')

    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['a0',h, N, time_assemble, time_step, time_step2, time_step3])


    # az gradient
    problem_az = gradients.gradient_az(beta, v, W, bcs, h)
    t0 = time.time()
    problem_az.assemble_problem(u0, tau)#, u_ex)
    time_assemble = time.time() - t0

    t0 = time.time()
    problem_az.step()
    time_step = time.time() - t0

    t0 = time.time()
    problem_az.step()
    time_step2 = time.time() - t0

    t0 = time.time()
    problem_az.step()
    time_step3 = time.time() - t0

    print(f'h: {h}, N: {N}, time assemble: {time_assemble}, time step: {time_step}, time step2: {time_step2}, time step3: {time_step3}')

    with open(filename_results, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['az',h, N, time_assemble, time_step, time_step2, time_step3])

def plotting():
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv('./results/Budget_defintion.csv',
                     dtype={"name_opt": str, "h": float, "N": int, "time_assemble": float, "time_step": float, "time_step2": float, "time_step3": float})

    h = df["h"].unique()
    methods = df['name_opt'].unique()
    N = df['N'].unique()
    
    T_assemble = []#{'L2':[], 'H1':[], 'a0':[], 'az':[]}
    T_step1 = []#{'L2':[], 'H1':[], 'a0':[], 'az':[]}
    T_step2 = []#{'L2':[], 'H1':[], 'a0':[], 'az':[]}
    T_step3 = []#{'L2':[], 'H1':[], 'a0':[], 'az':[]}

    for method in methods:
        Ta_m = []
        Ts1_m = []
        Ts2_m = []
        Ts3_m = []
        for hi in h:
            df_filtered = df[(df['name_opt'] == method) &
                             (df['h'] == hi)]
            Ta_m.append(df_filtered["time_assemble"].mean())
            Ts1_m.append(df_filtered["time_step"].mean())
            Ts2_m.append(df_filtered["time_step2"].mean())
            Ts3_m.append(df_filtered["time_step3"].mean())

        print(f'Method: {method}, Assemble times: {Ts1_m}')
        print(f'Method: {method}, Step1 times: {Ts2_m}')
        print(f'Method: {method}, Step2 times: {Ts3_m}')

        T_assemble.append((method,Ta_m))
        T_step1.append((method,Ts1_m))
        T_step2.append((method,Ts2_m))
        T_step3.append((method,Ts3_m))
    
    fig, ax = plt.subplots(1,2, figsize=(24,12))

    ax[0].set_title('Assemble Time')
    ax[0].set_yscale('log', base = 2)
    ax[0].set_xscale('log', base = 2)
    ax[0].plot(h, h**-1*h[0], 'k--', label='O(h^-1)')
    ax[0].plot(h, h**-2*h[0]**2, 'k-.', label='O(h^-2)')
    ax[0].plot(h, h**-3*h[0]**3, 'k:', label='O(h^-3)')
    for method, Ta_m in T_assemble:
        ax[0].plot(h, Ta_m/Ta_m[0], 'o-', label=method)
    ax[0].set_xlabel('h')
    ax[0].set_ylabel('Time assemble [s]')
    ax[0].legend()
    ax[0].grid()

    ax[1].set_title('Time Step')
    ax[1].set_yscale('log', base = 2)
    ax[1].set_xscale('log', base = 2)
    ax[1].plot(h, h**-1*h[0], 'k--', label='O(h^-1)')
    ax[1].plot(h, h**-2*h[0]**2, 'k-.', label='O(h^-2)')
    #ax[1].plot(h, h**-3, 'k:', label='O(N^-3)')
    for method, Ts1_m in T_step1:
        ax[1].plot(h, Ts1_m/Ts1_m[0], 'o-', label=method)
    ax[1].set_xlabel('h')
    ax[1].set_ylabel('Time step [s]')
    ax[1].legend()
    ax[1].grid()

    plt.show()


# Parse lightweight CLI args before importing Firedrake/PETSc so
# PETSc's option parser does not see application-specific flags like `--h`.
parser = argparse.ArgumentParser()
parser.add_argument("--log2h", type=float, help="value of the space discretization parameter h", default=0)
parser.add_argument("--plot", action="store_true", help="whether to plot the solution or not", default=False)

args, unknown = parser.parse_known_args()
# Remove parsed arguments from sys.argv so PETSc / firedrake won't report them
sys.argv = [sys.argv[0]] + unknown

if (not(args.plot) and args.log2h == 0)  or (args.plot and args.log2h != 0):
    print("Error, both  or none not possible. Exiting.")
    sys.exit(0)
elif args.plot:
    plotting()
else:
    compute_times(args.log2h)
