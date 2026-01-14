import firedrake as fd
# import numpy as np
import csv
import matplotlib.pyplot as plt
from optimizer import Gradient_Descent
# import time

xmin, ymin = -6., -6.
xmax, ymax = 6., 6.

# the used h are below, this is only for the convergence test
# h_v = [12 * 2**(-4), 12 * 2**(-6),12 * 2**(-7), 12 * 2**(-8),12 * 2**(-9)]
h_v = [12 * 2**(-8)]#[12 * 2**(-6),12 * 2**(-8)]
beta_v = [10, 100, 1000]
# tau_v = [0.01, 0.005, 0.001]
tau_v = [1, 0.5]
methods = ['L2', 'H1', 'a0', 'az']

MaxIter = 100
toll = 1e-5

filename_results = './results/test_1_prove.csv'


with open(filename_results, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time'])

E_ref = {10: 0.79620688, 100: 1.97298868, 1000: 5.99303235} # value for h = 12 * 2 **(-8)
#        10: 0.79620688, 100: 1.97298868, 1000: 5.99303235

for h in h_v:
    for beta in beta_v:
        nx = int((xmax-xmin)/h)
        ny = int((ymax-ymin)/h)
        mesh = fd.RectangleMesh(nx, ny, xmax, ymax, originX = xmin, originY = ymin, diagonal = 'left')
        W = fd.FunctionSpace(mesh, 'CG',1)

        # Data and boundary conditions
        x, y = fd.SpatialCoordinate(mesh)
        v = 0.5 * (x**2 + y**2)
        bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]
        u0 = 1/fd.pi**(0.5) * fd.exp(-(x**2 + y**2) / 2)

        optim_GD = Gradient_Descent(beta,v,W, bcs, h)

        for tau in tau_v:
            for name in methods:
                optim_GD.compile(u0, tau, E_ref[beta], grad_type = name)

                res = optim_GD.minimize(MaxIter, toll,verbose = False)

                optim_GD.save_data(filename_results, res)

                # optim_GD.plot_history()

                if res["converged"]:
                    print()
                    print(f'{name} minization with h: {h}, beta: {beta}, tau:{tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
                    print()
                else:
                    print()
                    print(f'{name} minization with h: {h}, beta: {beta}, tau:{tau} did NOT converged in iterate: {res["iterate"]}')
                    print()
plt.show()