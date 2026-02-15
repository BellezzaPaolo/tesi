import firedrake as fd
import sys
import csv
import time
from optimizer import Gradient_Descent, ParaflowS

is_save_CSV = True
is_save_log = True
is_save_plot = True

folder = './incontro1_alpha_min/'

filename_results_GD = folder + 'GD.csv'
filename_results_PF = folder + 'PF.csv'

nx = 256
beta_v = [1000]#,100,1000]
tau_v = [1, 0.5, 0.25, 0.1, 0.05, 0.025] #[1, 0.5]
MaxIter = 1000
toll = 1e-5
E_ref = {10: 0.79620688, 100: 1.97298868, 1000: 5.99303235}
methods_coarse = ['L2_P', 'az']
methods_fine = ['L2_P', 'az']
Nf_v = [2, 4, 5, 6]#, 10]
Ng_v = [2, 5, 100]# 10, 20, 100]

mesh = fd.RectangleMesh(nx, nx, 6, 6, -6, -6, diagonal = 'left')
W = fd.FunctionSpace(mesh,'CG',1)

x,y = fd.SpatialCoordinate(mesh)
v = 0.5 * (x**2 + y**2)
bcs = [fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4))]
u0 = 1/fd.pi**(0.5) * fd.exp(-(x**2 + y**2)/2)

if is_save_CSV:
    with open(filename_results_GD, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time'])

    with open(filename_results_PF, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['fine_operator', 'coarse_operator', 'h', 'beta', 'N_fine', 'N_coarse', 'tau_fine', 'tau_coarse', 'energy', 'lambda', 'iterate_coarse', 'iterate_fine', 'iterate', 'error', 'total_time', 'mean_time'])

orig_stdout = sys.stdout
t_start = time.time()

only_for_print_time = 0

for beta in beta_v:
    optim = ParaflowS(beta,v,W, bcs, 12 * 2**(-8))
    optim_GD = Gradient_Descent(beta,v,W, bcs, 12 * 2**(-8))

    for tau in tau_v:
        for name_fine in methods_fine:
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

            #for name_coarse in methods_coarse:
            name_coarse = name_fine
            for Nf in Nf_v:
                for Ng in Ng_v:

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

            only_for_print_time += 1
            print(f'Done {name_fine} minimization for tau = {tau} and beta = {beta} in {time.time() - t_start} seconds. Missing time: {(time.time() - t_start)*(len(tau_v)*len(beta_v)*len(methods_fine) - only_for_print_time)/ only_for_print_time}')