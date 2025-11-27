import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
import utils
import gradients
import time

xmin, ymin = -6., -6.
xmax, ymax = 6., 6.

h_v = [12 * 2**(-6)]#,12 * 2**(-8)]
beta_v = [10, 100, 1000]
tau_v = [1, 0.5]

MaxIter = 50
toll = 1e-5

for h in h_v:
    for beta in beta_v:
        for tau in tau_v:
            nx = int((xmax-xmin)/h)
            filename = './Ground_Truth/U_GS_b'+str(beta)+'_N'+str(nx)+'.h5'
            mesh, u_ex = utils.load_ground_truth(filename)
            W = fd.FunctionSpace(mesh, 'CG',1)

            # Data and boundary conditions
            x = fd.SpatialCoordinate(mesh)
            v = 0.5 * (x[0]**2 + x[1]**2)
            bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]

            problem = gradients.gradient_a0(beta, v, W, bcs, h)

            u0 = 1/np.pi**(0.5) * fd.exp(-(x[0]**2 + x[1]**2) / 2)

            problem.assemble_problem(u0, tau, u_ex)

            res = problem.minimize(MaxIter, toll)

            if res["converged"]:
                print(f'Minization with h: {h}, beta: {beta}, tau:{tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
                print(f'Final error: {res["error"]} and norm of the solution: {res["norm"]}')
            else:
                print(f'Minization with h: {h}, beta: {beta}, tau:{tau} did not converged to energy: {res["energy"]:.4f} with lambda: {res["lam"]:.4f} at the iterate: {res["iterate"]}')
                print(f'Final error: {res["error"]} and norm of the solution: {res["norm"]}')