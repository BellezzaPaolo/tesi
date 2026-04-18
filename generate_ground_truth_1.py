"""Ground-truth run for case 1: harmonic potential with multiple beta values.

This script computes highly converged reference solutions by running a
normalized fixed-point/gradient-flow style iteration with a direct LU solve.
"""

import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
import old.utils as utils
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def assemble_forms(u, w, v, tau, u_old, beta):
    # Keep numeric parameters as UFL constants to preserve symbolic assembly.
    tau_c = fd.Constant(tau)
    beta_c = fd.Constant(beta)

    a = u * w * fd.dx \
        + tau_c * 0.5 * fd.dot(fd.grad(u),fd.grad(w)) * fd.dx \
        + tau_c * (v * u) * w * fd.dx \
        + tau_c * beta_c * (u_old **2 * u) * w * fd.dx

    rhs = u_old * w * fd.dx
    return a, rhs

xmin, ymin = -6., -6.
xmax, ymax = 6., 6.

# Spatial resolution(s) and interaction strengths used for reference energies.
h_v = [12* 2**(-8)]#[12 * 2**(-4), 12 * 2**(-6),12 * 2**(-7), 12 * 2**(-8),12 * 2**(-9), 12 * 2**(-10)]
beta_v = [10, 100, 1000]

for h in h_v:
    nx = int((xmax-xmin)/h)
    ny = int((ymax-ymin)/h)

    mesh = fd.RectangleMesh(nx, ny, xmax, ymax, originX = xmin, originY = ymin, diagonal = 'right')

    # Plot it
    # fig, ax = plt.subplots()
    # fd.triplot(mesh, axes=ax)
    # ax.legend()
    # ax.axis('equal')
    # plt.title('Mesh')

    # function spaces
    W = fd.FunctionSpace(mesh, 'CG', 1)


    # Harmonic trap and homogeneous Dirichlet boundary conditions.
    x = fd.SpatialCoordinate(mesh)
    v = 0.5 * (x[0]**2 + x[1]**2)
    bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]

    tau = 1 #[1., 0.5]

    # Trial/test functions and normalized Gaussian initial condition.
    u = fd.TrialFunction(W)
    w = fd.TestFunction(W)

    u0 = 1/fd.pi**(0.5) * fd.exp(-(x[0]**2 + x[1]**2) / 2)
    # u0 = fd.Constant(1.)

    u_old = fd.Function(W)
    u_old.interpolate(u0)

    # Plot the potential once for visual verification.
    v_func = fd.Function(W)
    v_func.interpolate(v)
    div_theme = LinearSegmentedColormap.from_list(
        "div_theme",
        [
            "#2c3e50",  # dark blue
            "#4a90e2",  # light blue
            "#ffffff",  # center (zero)
            "#f5b041",  # light orange
            "#e67e22"   # strong accent
        ]
    )
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    col = fd.tripcolor(v_func, axes=ax, cmap=div_theme)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.margins(0)
    ax.set_xticks([])
    ax.set_yticks([])
    cax = inset_axes(
        ax,
        width="100%",   # exact same width as plotting axis
        height="4%",
        loc="lower left",
        bbox_to_anchor=(0.0, -0.08, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    fig.colorbar(col, cax=cax, orientation='horizontal')
    # ax.axis('off')
    # plt.title(r'Potential: $V(x,y) = \frac{1}{2}(x^2 + y^2)$')
    plt.show()

    # a, rhs = assemble_forms(u, w, v, tau[0], u_old, beta[0])

    uh = fd.Function(W)

    # problem = fd.LinearVariationalProblem(a, rhs, uh, bcs)
    # param = {'ksp_type': 'gmres', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu',
    #          'ksp_monitor':None}
    # Direct linear solve at every iteration (robust for reference runs).
    param = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
    # solver =  fd.LinearVariationalSolver(problem, solver_parameters=param)

    MaxIter = 1000
    toll = 1e-10

    for beta in beta_v:

        # Energy associated with the current beta.
        def energy(uh, v = v, beta = beta):
            return 0.5 * fd.assemble(( 0.5 * fd.dot(fd.grad(uh), fd.grad(uh)) + v * uh**2 + beta/2 * abs(uh) **4) * fd.dx)

        for i in range(MaxIter):
            # One implicit step, then project back to the L2 unit sphere.
            a, rhs = assemble_forms(u, w, v, tau, u_old, beta)
            problem = fd.LinearVariationalProblem(a, rhs, uh, bcs)
            solver =  fd.LinearVariationalSolver(problem, solver_parameters=param)
            solver.solve()

            uh.assign( uh / fd.norm(uh, 'L2'))

            error = fd.errornorm(uh, u_old,'L2')/fd.norm(uh,'L2')

            print(f'\rIter {i}, Error: {error:.2e}, Energy: {energy(uh):.10f}', end="", flush=True)

            
            u_old.assign(uh)

            if error < toll:
                break

        # Report the converged energy and chemical potential estimate.
        e_gs = energy(uh)
        lamb_gs = 2 * e_gs + beta/2 * fd.norm(uh,'L4')**4

        print()
        print()
        print(f'Final energy estimate: {e_gs} and associated lambda: {lamb_gs} with h: {h} and beta: {beta}') #0.79620688 2.06380
        # filename = './Ground_Truth_1/U_GS_b'+str(beta)+'_N'+str(nx)+'.h5'
        # utils.save_uh(mesh, uh, filename)

        # Plot the converged state for this beta.
        fig, ax = plt.subplots()
        col = fd.tripcolor(uh, axes=ax, cmap='coolwarm')
        plt.colorbar(col)
        ax.axis('equal')
        plt.title(f'Final Solution ($h = 12 \\times 2^{{{int(np.log2(h/12))}}}$, $\\beta = {beta}$)')
        plt.show()
