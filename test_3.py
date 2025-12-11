import firedrake as fd
import numpy as np
import csv
import matplotlib.pyplot as plt
import utils
import gradients
# import time

# reproducibility
rng = np.random.default_rng(21)

xmin, ymin = -6., -6.
xmax, ymax = 6., 6.

epsilon = 0.03

nxV = int((xmax - xmin) / (epsilon))
nyV = int((ymax - ymin) / (epsilon))

meshV = fd.RectangleMesh(nxV, nyV, xmax, ymax, originX=xmin, originY=ymin, quadrilateral=True)

# Function space for a cellwise-constant potential
Wdg = fd.FunctionSpace(meshV, "DG", 0)   # piecewise-constant per cell

# Values to choose between
val_low = 1.0
val_high = (epsilon**-2)   # epsilon^{-2}

# sample Nx * Ny random choices in row-major order
choices = rng.choice([val_low, val_high], size=(nyV, nxV), p=[0.5, 0.5])

# Firedrake cell ordering for a RectangleMesh is compatible with flattening rows,
# but to be safe we flatten in the same natural order (row-major)
flat = choices.ravel(order="C")  # length = number of cells

# create the DG0 function and assign cell values
V_random = fd.Function(Wdg)
# The DG0 Function stores one degree of freedom per cell. We can write into the
# underlying vector directly. For many Firedrake versions:
V_random.dat.data[:] = flat

h = 12 * 2**(-8)
beta = 100

nx = int((xmax - xmin) / h)
ny = int((ymax - ymin) / h)

filename_ref = './Ground_Truth_3/U_GS_b'+str(beta)+'_N'+str(nx)+'.h5'
mesh, u_ex = utils.load_ground_truth(filename_ref)
W = fd.FunctionSpace(mesh, 'CG',1)

v = fd.Function(W)
v.interpolate(V_random)

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


bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]

mu_TF = fd.Constant(fd.sqrt(beta / fd.pi))

u0 = fd.conditional(v < mu_TF, fd.sqrt((mu_TF- v)/beta), 0.0)

MaxIter = 1000
toll = 1e-5

tau_L2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10., 100., 1000.]
tau_az = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

filename_results = './results/test_3.csv'


with open(filename_results, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['optimizer_name', 'h', 'beta', 'tau', 'energy', 'lambda', 'iterate', 'error', 'total_time', 'mean_time'])


# L2 gradient
problem_L2 = gradients.gradient_L2(beta, v, W, bcs, h)
for tau in tau_L2:
    problem_L2.assemble_problem(u0, tau, u_ex)

    res = problem_L2.minimize(MaxIter, toll)

    problem_L2.save_data(filename_results, 'L2',res)

    # problem_L2.plot_history('L2')

    if res["converged"]:
        print(f'L2 minization with h: {h}, beta: {beta}, tau:{tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
    else:
        print(f'L2 minization with h: {h}, beta: {beta}, tau:{tau} did NOT converged in iterate: {res["iterate"]}')
    
# a_z gradient
problem_az = gradients.gradient_az(beta, v, W, bcs, h)
for tau in tau_az:
    problem_az.assemble_problem(u0, tau, u_ex)

    res = problem_az.minimize(MaxIter, toll)

    problem_az.save_data(filename_results, 'az',res)

    # problem_az.plot_history('az')

    if res["converged"]:
        print(f'a_z minization with h: {h}, beta: {beta}, tau:{tau} converged to energy: {res["energy"]} with lambda: {res["lam"]} at the iterate: {res["iterate"]}')
    else:
        print(f'a_z minization with h: {h}, beta: {beta}, tau:{tau} did NOT converged in iterate: {res["iterate"]}')
