import firedrake as fd
import numpy as np
import csv
import matplotlib.pyplot as plt
from optimizer import Gradient_Descent
from test.pot_3 import RandomDisorderPotential
# import time

xmin, xmax = -6., 6.
ymin, ymax = -6., 6.

nx = 256
h = (xmax - xmin) / nx
# beta_v = [1000]#,100,1000]
MaxIter = 1000
toll = 1e-5

tau_v = {'az':list(np.linspace(0.5, 2.0, 15)), 'L2_P': [0.01, 0.05, 0.1, 0.5, 0.8, 1, 1.5, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100]} #L2_P
# [0.05,0.1]
beta = 10
# tau_v = list(np.linspace(0.5, 2.0, 15)) #az
E_ref = {10: 4.602621438437267}

mesh = fd.RectangleMesh(nx, nx, 6, 6, -6, -6, diagonal = 'left')
W = fd.FunctionSpace(mesh,'CG',1)

x,y = fd.SpatialCoordinate(mesh)

epsilon = 0.03

nxV = int((xmax - xmin) / (epsilon))
nyV = int((ymax - ymin) / (epsilon))

meshV = fd.RectangleMesh(nxV, nyV, xmax, ymax, originX=xmin, originY=ymin, quadrilateral=True)
# Function space for a cellwise-constant potential
Vdg = fd.FunctionSpace(meshV, "DG", 0)   # piecewise-constant per cell

# Create random disorder potential using the RandomDisorderPotential class
# Using seed=21 for consistency with the original rng seed
potential_obj = RandomDisorderPotential(number_of_cells=400, domain_size=12.0, seed=21)

# Create the DG0 function for the potential
V_random = fd.Function(Vdg)

# Get cell coordinates from meshV and evaluate the potential
coord_fn_V = fd.Function(fd.VectorFunctionSpace(meshV, 'DG', 0))
coord_fn_V.interpolate(fd.SpatialCoordinate(meshV))
coords_V = coord_fn_V.dat.data_ro

# Evaluate potential at cell coordinates and assign to V_random
V_random.dat.data[:] = potential_obj.evaluate(coords_V[:, 0], coords_V[:, 1])

Wv = fd.FunctionSpace(mesh, 'DG', 0)

v = fd.Function(Wv)

# Get cell midpoints from the triangular mesh to evaluate V_random
# For a DG0 function, we need one value per cell
# Use the cell centroid coordinates
x_dg = fd.SpatialCoordinate(mesh)
coord_fn = fd.Function(fd.VectorFunctionSpace(mesh, 'DG', 0))
coord_fn.interpolate(x_dg)
coords = coord_fn.dat.data_ro

# Step 2: Evaluate V_random at those coordinates
vals = np.array([V_random.at((float(px), float(py)), tolerance=1e-10)
                for px, py in coords])

# Step 3: assign to the DG0 field on the triangular mesh
v.dat.data[:] = vals

mu_TF = fd.Constant(fd.sqrt(beta / fd.pi))

u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF - v)/beta), 0.0)

bcs = [fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4))]

# fig, ax = plt.subplots(2,1, figsize=(5,10))
# col = fd.tripcolor(v, axes=ax[0], cmap='BrBG')
# plt.colorbar(col)
# ax[0].set_title('Random Potential v(x)')
# ax[0].axis('equal')
# col = fd.tripcolor(u_ex, axes = ax[1], cmap='coolwarm')
# plt.colorbar(col)
# ax[1].set_title('Ground Truth u_ex(x)')
# ax[1].axis('equal')
# plt.show()

filename_results = './results/test_3.csv'


with open(filename_results, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time'])


# L2 gradient
optim_GD = Gradient_Descent(beta,v,W, bcs, h)

# L2 gradient
for tau in tau_v['L2_P']:
    optim_GD.compile(u0, tau, E_ref[beta], grad_type = 'L2')

    res = optim_GD.minimize(MaxIter, toll, False)

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
for tau in tau_v['az']:
    optim_GD.compile(u0, tau, E_ref[beta], grad_type = 'az')

    res = optim_GD.minimize(MaxIter, toll, False)

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
optim_GD.compile(u0, E_ref[beta], grad_type = 'az_ada')

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