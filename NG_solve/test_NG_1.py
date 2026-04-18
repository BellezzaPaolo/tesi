"""NGSolve experiment: harmonic potential, multiple beta and tau values.

Compares L2 and az gradient variants and writes convergence statistics to CSV.
"""

from gradients_NG import *
import ngsolve as ng
from netgen.geom2d import SplineGeometry
import csv


# Parameter sweep for test case 1.
hmax_v = [12 * 2**(-8)] #0.046875
order = 1
dirichlet_bcs = 'outer'
beta_v = [10, 100, 1000]
potential = 'Harmonic'
tau_v = [1, 0.5]
MaxIter = 100
toll = 1e-5
E_ref = {1000: 5.9930293858250465, 100: 1.972982094492599, 10: 0.7961881636763722}
#        1000: 5.99303235          100: 1.97298868         10: 0.79620688
initial_guess = 'normalized gaussian'

filename = '../results/test_NG_1.csv'

# Create/append CSV with fixed schema.
with open(filename, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time'])


for hmax in hmax_v:
    for beta in beta_v:
        # Build domain and mesh once per (h, beta) configuration.
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-6, -6), p2=(6, 6), bc="outer")

        ngmesh = geo.GenerateMesh(maxh=hmax)
        mesh = ng.Mesh(ngmesh)

        grad_L2 = Gradient_L2(beta, potential, hmax, mesh, order, dirichlet_bcs)
        grad_az = Gradient_az(beta, potential, hmax, mesh, order, dirichlet_bcs)

        for tau in tau_v:
            # L2 benchmark run.
            grad_L2.assemble_problem(initial_guess, tau, E_ref[beta])
            res = grad_L2.minimize(MaxIter, toll)

            grad_L2.save_data(filename, 'L2', res)

            if res["converged"]:
                print(f'L2 minization with h: {hmax}, beta: {beta}, tau: {tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
            else:
                print(f'L2 minization with h: {hmax}, beta: {beta}, tau: {tau} did NOT converged in iterate: {res["iterate"]}')

            # az benchmark run.
            grad_az.assemble_problem(initial_guess, tau, E_ref[beta])
            res = grad_az.minimize(MaxIter, toll)

            grad_az.save_data(filename, 'az', res)

            if res["converged"]:
                print(f'az minization with h: {hmax}, beta: {beta}, tau: {tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
            else:
                print(f'az minization with h: {hmax}, beta: {beta}, tau: {tau} did NOT converged in iterate: {res["iterate"]}')
