import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
import utils

def compute_riesz(R_u,R_u2u, u, w, v, u_old, beta, bcs, param = None):
    a = 0.5 * fd.dot(fd.grad(u), fd.grad(w)) * fd.dx\
        + v * u * w * fd.dx
    rhs = u_old * w * fd.dx

    problem_Ru = fd.LinearVariationalProblem(a, rhs, R_u, bcs)
    solver_Ru = fd.LinearVariationalSolver(problem_Ru, solver_parameters = param)
    solver_Ru.solve()

    beta_c = fd.Constant(beta)
    rhs2 = beta_c * abs(u_old)**2 * u_old * w * fd.dx 

    problem_Ru2u = fd.LinearVariationalProblem(a, rhs2, R_u2u, bcs)
    solver_Ru2u = fd.LinearVariationalSolver(problem_Ru2u, solver_parameters = param)
    solver_Ru2u.solve()    

    return


def assemble_forms(u, w, tau, u_old, R_u,R_u2u):
    # ensure numeric parameters are UFL Constants to avoid premature python-side
    # evaluation that can produce plain Python numbers instead of UFL expressions
    tau_c = fd.Constant(tau)

    m = u * w * fd.dx

    int_E = fd.assemble((u_old + R_u2u) * u_old * fd.dx)
    int_R = fd.assemble(R_u * u_old * fd.dx)

    rhs = u_old * w *fd.dx\
        - tau_c * (u_old + R_u2u) *w * fd.dx \
        + tau_c * int_E/int_R * R_u * w * fd.dx 

    return m, rhs

xmin, ymin = -6., -6.
xmax, ymax = 6., 6.

h_v = [12 * 2**(-6),12 * 2**(-8)]
beta_v = [10, 100, 1000]
tau_v = [1,0.5]


MaxIter = 100
toll = 1e-5

for h in h_v:
    for beta in beta_v:
        for tau in tau_v:
            nx = int((xmax-xmin)/h)

            filename = './Ground_Truth/U_GS_b'+str(beta)+'_N'+str(nx)+'.h5'
            mesh, u_ex = utils.load_ground_truth(filename)
            # function spaces
            W = fd.FunctionSpace(mesh, 'CG', 1)


            # Data and boundary conditions
            x = fd.SpatialCoordinate(mesh)
            v = 0.5 * (x[0]**2 + x[1]**2)
            bcs = [ fd.DirichletBC(W, fd.Constant(0.0), (1,2,3,4)) ]

            # define the variational problem
            u = fd.TrialFunction(W)
            w = fd.TestFunction(W)

            u0 = 1/np.pi**(0.5) * fd.exp(-(x[0]**2 + x[1]**2) / 2)

            u_old = fd.Function(W)
            u_old.interpolate(u0)

            uh = fd.Function(W)
            R_u = fd.Function(W)
            R_u2u = fd.Function(W)

            def energy(uh, v = v, beta = beta):
                return 0.5 * fd.assemble(( 0.5 * fd.dot(fd.grad(uh), fd.grad(uh)) + v * uh**2 + beta/2 * abs(uh) **4) * fd.dx)


            # param = {'ksp_type': 'gmres', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu',
            #          'ksp_monitor':None}
            # Use the following parameters if, instead, you want to solve the problem by a direct method.
            param = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}

            energy_ref = energy(u_ex)

            for i in range(MaxIter):

                compute_riesz(R_u,R_u2u, u, w, v, u_old, beta, bcs)
                
                a, rhs = assemble_forms(u, w, tau, u_old, R_u, R_u2u)
                problem = fd.LinearVariationalProblem(a, rhs, uh, bcs)
                solver =  fd.LinearVariationalSolver(problem, solver_parameters=param)
                solver.solve()

                uh.assign( uh / fd.norm(uh, 'L2'))

                energy_u = energy(uh)

                error = abs(energy_u - energy_ref)/energy_ref

                print(f'\rIter {i}, Error: {error:.2e}, Energy: {energy_u:.10f}', end="", flush=True)

                
                u_old.assign(uh)

                if error < toll:
                    lamb_gs = 2 * energy_u + beta/2 * fd.norm(uh,'L4')**4
                    print('\r', end="", flush= True)
                    print(f'Final energy estimate: {energy_u} and associated lambda: {lamb_gs} with h: {h} and beta: {beta}, Iter: {i}')
                    break
