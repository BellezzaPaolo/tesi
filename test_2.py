import firedrake as fd
# import numpy as np
import csv
import matplotlib.pyplot as plt
import utils
from optimizer import Gradient_Descent
# import time

xmin, ymin = -6., -6.
xmax, ymax = 6., 6.

# the real h is 12 * 2**(-8), the other ones are for the convergence analysis
# h_v = [12 * 2**(-4), 12 * 2**(-6),12 * 2**(-7), 12 * 2**(-8),12 * 2**(-9)]
h_v = [12 * 2**(-8)]
beta = 1000
tau_az = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
tau_L2 = [0.1, 0.5, 1., 1.5, 2., 2.5, 3., 5., 10., 100., 1000.]

MaxIter = 100
toll = 1e-5

filename_results = './results/test_2_pointwise_itself.csv'


with open(filename_results, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time'])

for h in h_v:
    nx = int((xmax-xmin)/h)
    filename_ref = './Ground_Truth_2/U_GS_b'+str(beta)+'_N'+str(nx)+'.h5'
    mesh, u_ex = utils.load_ground_truth(filename_ref)
    W = fd.FunctionSpace(mesh, 'CG',1)

    # Data and boundary conditions
    x = fd.SpatialCoordinate(mesh)
    v = 0.5 * (x[0]**2 + x[1]**2) + fd.Constant(20) + fd.Constant(20) * fd.sin( 2 * fd.pi * x[0]) * fd.sin(2 * fd.pi * x[1])
    bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]

    mu_TF = fd.Constant(fd.sqrt(beta / fd.pi))

    u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF - v)/beta), 0.0)

    E_ref = 0.5 * fd.assemble(0.5 * fd.dot(fd.grad(u_ex), fd.grad(u_ex)) * fd.dx + v * u_ex **2 * fd.dx + beta * 0.5 * abs(u_ex)**4 * fd.dx)
    # L2 gradient
    optim_GD = Gradient_Descent(beta,v,W, bcs, h)
    for tau in tau_L2:
        optim_GD.compile(u0, tau, E_ref, grad_type = 'L2')

        res = optim_GD.minimize(MaxIter, toll)

        optim_GD.save_data(filename_results, res)

        # problem_L2.plot_history()

        if res["converged"]:
            print()
            print(f'L2 minization with h: {h}, beta: {beta}, tau:{tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
            print()
        else:
            print()
            print(f'L2 minization with h: {h}, beta: {beta}, tau:{tau} did NOT converged in iterate: {res["iterate"]}')
            print()
        
    # a_z gradient
    for tau in tau_az:
        optim_GD.compile(u0, tau, E_ref, grad_type = 'az')

        res = optim_GD.minimize(MaxIter, toll)

        optim_GD.save_data(filename_results, res)

        # problem_az.plot_history('az')

        if res["converged"]:
            print()
            print(f'a_z minization with h: {h}, beta: {beta}, tau:{tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
            print()
        else:
            print()
            print(f'a_z minization with h: {h}, beta: {beta}, tau:{tau} did NOT converged in iterate: {res["iterate"]}')
            print()

    plt.show()