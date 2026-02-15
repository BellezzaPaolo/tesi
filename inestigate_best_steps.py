import firedrake as fd
import sys
import csv
import time
from tqdm import tqdm
from optimizer import Gradient_Descent, ParaflowS

is_save_CSV = True

filename_results_PF = './graphs/PF.csv'

nx = 256
beta_v = [1000]#,100,1000]
tau_v = [0.6, 0.75, 0.375, 1, 0.5, 0.25, 0.1]

func = [lambda ta, Nf: tau/8, lambda tau, Nf: tau /4, lambda tau, Nf: tau /2, lambda tau, Nf: tau , lambda tau, Nf: tau * 2,lambda tau, Nf: tau * 4, lambda tau, Nf: tau * Nf, lambda tau, Nf: tau / Nf]

        
MaxIter = 200
toll = 1e-5
E_ref = {10: 0.79620688, 100: 1.97298868, 1000: 5.99303235}
methods_coarse = ['L2_P', 'az']
methods_fine = ['az']#['L2_P', 'az']
Nf_v = [1]#[2, 4, 5, 6]#, 10]
Ng_v = [2, 5, 10, 20, 100]

set_values_tau = set()

for tau in tau_v:
    for Nf in Nf_v:
        for f in func:
            set_values_tau.add((tau,f(tau, Nf)))

set_values_tau.add((0.44,0.6))
set_values_tau.add((0.75,1))
set_values_tau.add((0.75,2))
set_values_tau.add((0.75,5))
set_values_tau.add((0.6,5))
set_values_tau.add((0.5,4))

print(f"number of points: {len(set_values_tau)}")

mesh = fd.RectangleMesh(nx, nx, 6, 6, -6, -6, diagonal = 'left')
W = fd.FunctionSpace(mesh,'CG',1)

x,y = fd.SpatialCoordinate(mesh)
v = 0.5 * (x**2 + y**2)
bcs = [fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4))]
u0 = 1/fd.pi**(0.5) * fd.exp(-(x**2 + y**2)/2)

if is_save_CSV:
    with open(filename_results_PF, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['fine_operator', 'coarse_operator', 'h', 'beta', 'N_fine', 'N_coarse', 'tau_fine', 'tau_coarse', 'energy', 'lambda', 'iterate_coarse', 'iterate_fine', 'iterate', 'error', 'total_time', 'mean_time'])

#done = False

for beta in beta_v:
    optim = ParaflowS(beta,v,W, bcs, 12 * 2**(-8))

    for tau, tau_g in tqdm(set_values_tau, desc='Processing tau values'):
            print(f"Starting minimization for tau = {tau} and tau_g = {tau_g}")
            
            #if not done:
            for name_fine in methods_fine:

                #for name_coarse in methods_coarse:
                name_coarse = name_fine
                for Nf in Nf_v:
                    for Ng in Ng_v:

                        optim.compile(u0, tau, tau_g, E_ref[beta],grad_type_coarse=name_coarse, grad_type_fine = name_fine, Nf = Nf, Ng = Ng)

                        filename = 'PF'+ name_fine + '_' + name_coarse + '_tau' + str(tau) + '_Nf'+ str(Nf)+ '_Ng' + str(Ng)

                        res = optim.minimize(MaxIter, toll, verbose=False)

                        if is_save_CSV:
                            optim.save_data(filename_results_PF, res)

            # if tau == 0.6 and tau_g == 0.6:
            #     done = True
            