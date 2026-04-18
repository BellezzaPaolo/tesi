import firedrake as fd
import numpy as np
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
tau_az = list(np.linspace(0.5, 1.7,20))#[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
tau_L2 = np.sort(list(np.logspace(-2, 3,10, base = 20)) + list(np.linspace(0.1,100, 20))) #[0.1, 0.5, 1., 1.5, 2., 2.5, 3., 5., 10., 100., 1000.]
Iter_L2 = []

Iter_az = []

MaxIter = 100
toll = 1e-5

# filename_results = './results/test_2_pointwise_itself.csv'


# with open(filename_results, "a", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time'])

for h in h_v:
    nx = int((xmax-xmin)/h)
    mesh = fd.RectangleMesh(nx, nx, xmax, ymax, originX = xmin, originY = ymin, diagonal = 'left')

    W = fd.FunctionSpace(mesh, 'CG',1)

    # Data and boundary conditions
    x = fd.SpatialCoordinate(mesh)
    v = 0.5 * (x[0]**2 + x[1]**2) + fd.Constant(20) + fd.Constant(20) * fd.sin( 2 * fd.pi * x[0]) * fd.sin(2 * fd.pi * x[1])
    bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]

    mu_TF = fd.Constant(fd.sqrt(beta / fd.pi))

    u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF - v)/beta), 0.0)

    E_ref = 15.204825
    optim_GD = Gradient_Descent(beta,v,W, bcs, h)

    # L2 gradient
    for tau in tau_L2:
        optim_GD.compile(u0, E_ref, grad_type = 'L2', tau = tau)

        res = optim_GD.minimize(MaxIter, toll, False)
        Iter_L2.append(res["iterate"])

        # optim_GD.save_data(filename_results, res)

        # optim_GD.plot_history(show = True)

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
        optim_GD.compile(u0, E_ref, grad_type = 'az', tau = tau)

        res = optim_GD.minimize(MaxIter, toll, False)
        Iter_az.append(res["iterate"])

        # optim_GD.save_data(filename_results, res)

        # optim_GD.plot_history(show = True)

        if res["converged"]:
            print()
            print(f'a_z minization with h: {h}, beta: {beta}, tau:{tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
            print()
        else:
            print()
            print(f'a_z minization with h: {h}, beta: {beta}, tau:{tau} did NOT converged in iterate: {res["iterate"]}')
            print()

    # az_ada gradient
    optim_GD.compile(u0, E_ref, grad_type = 'az_ada')

    res = optim_GD.minimize(MaxIter, toll, False)

    # optim_GD.save_data(filename_results, res)

    optim_GD.plot_history(show = True)

    if res["converged"]:
        print()
        print(f'a_z adaptive minization with h: {h}, beta: {beta}, converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
        print()
    else:
        print()
        print(f'a_z adaptive minization with h: {h}, beta: {beta}, did NOT converged in iterate: {res["iterate"]}')
        print()


    # plt.figure()
    # plt.loglog(tau_L2, Iter_L2, label = 'L2', marker = 'o')
    # plt.legend()
    # plt.xlabel('Step size (tau)')
    # plt.ylabel('Number of iterations to converge')
    # plt.show()

    # plt.figure()
    # plt.loglog(tau_az, Iter_az, label = 'a_z', marker = 'x')
    # plt.legend()
    # plt.xlabel('Step size (tau)')
    # plt.ylabel('Number of iterations to converge')
    # plt.show()