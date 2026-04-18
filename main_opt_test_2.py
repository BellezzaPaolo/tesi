"""Main experiment driver for case 2 (augmented oscillatory potential).

Compares GD and ParaflowS across time steps and correction budgets, storing
CSV/log/plot outputs in the selected experiment folder.
"""

import firedrake as fd
import sys
import csv
import time
from optimizer import Gradient_Descent, ParaflowS

is_save_CSV = True
is_save_log = True
is_save_plot = True

folder = './incontro21/'

filename_results_GD = folder + 'GD.csv'
filename_results_PF = folder + 'PF.csv'

nx = 256
beta = 1000
# tau_az = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
# tau_L2 = [0.1, 0.5, 1., 1.5, 2., 2.5, 3., 5., 10., 100., 1000.]
tau_v = [1.5, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025] #[1, 0.5]
MaxIter = 1000
toll = 1e-5
# Candidate gradient operators for fine and coarse phases.
methods_coarse = ['L2_P', 'az']
methods_fine = ['L2_P', 'az']
# ParaflowS controls: number of fine and coarse correction steps.
Nf_v = [2, 5, 10]
Ng_v = [2, 5, 10, 20, 100]

# PDE setup for case 2 potential.
mesh = fd.RectangleMesh(nx, nx, 6, 6, -6, -6, diagonal = 'left')
W = fd.FunctionSpace(mesh, 'CG',1)

# Data and boundary conditions.
x = fd.SpatialCoordinate(mesh)
v = 0.5 * (x[0]**2 + x[1]**2) + fd.Constant(20) + fd.Constant(20) * fd.sin( 2 * fd.pi * x[0]) * fd.sin(2 * fd.pi * x[1])
bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]

mu_TF = fd.Constant(fd.sqrt(beta / fd.pi))
u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF - v)/beta), 0.0)
E_ref = {1000: 15.204825}

if is_save_CSV:
    # Create/append result files using a fixed schema.
    with open(filename_results_GD, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time'])

    with open(filename_results_PF, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['fine_operator', 'coarse_operator', 'h', 'beta', 'N_fine', 'N_coarse', 'tau_fine', 'tau_coarse', 'energy', 'lambda', 'iterate_coarse', 'iterate_fine', 'iterate', 'error', 'total_time', 'mean_time'])

orig_stdout = sys.stdout
t_start = time.time()

only_for_print_time = 0


optim = ParaflowS(beta,v,W, bcs, 12 * 2**(-8))
optim_GD = Gradient_Descent(beta,v,W, bcs, 12 * 2**(-8))

for tau in tau_v:
    for name_fine in methods_fine:
        # Baseline GD run for the current (tau, gradient) pair.
        optim_GD.compile(u0, tau, E_ref[beta],grad_type = name_fine)

        filename = 'GD'+ name_fine + '_tau' + str(tau)
        if is_save_log:
            f = open(folder + filename + '.log', 'w')
            sys.stdout = f
        res = optim_GD.minimize(MaxIter, toll)
        if is_save_plot:
            optim_GD.plot_history(filesave = folder +filename + '.png')

        if is_save_CSV:
            optim_GD.save_data(filename_results_GD, res)

        if is_save_log:
            f.close()

        # Current experiment ties coarse operator to fine operator.
        #for name_coarse in methods_coarse:
        name_coarse = name_fine
        for Nf in Nf_v:
            for Ng in Ng_v:

                # ParaflowS run with tau_f=tau and tau_c=tau*Nf.
                optim.compile(u0, tau, tau * Nf, E_ref[beta],grad_type_coarse=name_coarse, grad_type_fine = name_fine, Nf = Nf, Ng = Ng)

                filename = 'PF'+ name_fine + '_' + name_coarse + '_tau' + str(tau) + '_Nf'+ str(Nf)+ '_Ng' + str(Ng)
                if is_save_log:
                    f = open(folder + filename + '.log', 'w')
                    sys.stdout = f
                res = optim.minimize(MaxIter, toll)
                if is_save_plot:
                    optim.plot_history(filesave = folder + filename + '.png')
                if is_save_CSV:
                    optim.save_data(filename_results_PF, res)
                if is_save_log:
                    f.close()
            
        sys.stdout = orig_stdout

            # Console progress estimate for long parameter sweeps.
        only_for_print_time += 1
        print(f'Done {name_fine} minimization for tau = {tau} and beta = {beta} in {time.time() - t_start} seconds. Missing time: {(time.time() - t_start)*(len(tau_v)*len(methods_fine) - only_for_print_time)/ only_for_print_time}')