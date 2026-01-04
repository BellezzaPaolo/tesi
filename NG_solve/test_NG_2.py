from gradients_NG import *
import ngsolve as ng
from netgen.geom2d import SplineGeometry
import csv

hmax = 12 * 2**(-8)
order = 1
dirichlet_bcs = 'outer'
beta = 1000
potential = 'Harmonic and optical lattice'
tau_az = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
tau_L2 = [0.1, 0.5, 1., 1.5, 2., 2.5, 3., 5., 10., 100., 1000.]
MaxIter = 100
toll = 1e-5
E_ref = {1000: 15.19933842472953419933842}
#        1000: 15.204825
initial_guess = 'Thomas-Fermi density'

filename = '../results/test_NG_2.csv'

with open(filename, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time'])

geo = SplineGeometry()
geo.AddRectangle(p1=(-6, -6), p2=(6, 6), bc="outer")

ngmesh = geo.GenerateMesh(maxh=hmax)
mesh = ng.Mesh(ngmesh)

# L2 gradient
grad_L2 = Gradient_L2(beta, potential, hmax, mesh, order, dirichlet_bcs)

for tau in tau_L2:
    grad_L2.assemble_problem(initial_guess, tau, E_ref[beta])
    res = grad_L2.minimize(MaxIter, toll)

    grad_L2.save_data(filename, 'L2', res)

    if res["converged"]:
        print(f'L2 minization with h: {hmax}, beta: {beta}, tau: {tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
    else:
        print(f'L2 minization with h: {hmax}, beta: {beta}, tau: {tau} did NOT converged in iterate: {res["iterate"]}')

# az gradient
grad_az = Gradient_az(beta, potential, hmax, mesh, order, dirichlet_bcs)

for tau in tau_az:
    grad_az.assemble_problem(initial_guess, tau, E_ref[beta])
    res = grad_az.minimize(MaxIter, toll)

    grad_az.save_data(filename, 'az', res)

    if res["converged"]:
        print(f'az minization with h: {hmax}, beta: {beta}, tau: {tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
    else:
        print(f'az minization with h: {hmax}, beta: {beta}, tau: {tau} did NOT converged in iterate: {res["iterate"]}')