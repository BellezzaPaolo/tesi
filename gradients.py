"""Sobolev-gradient operators used by GD and ParaflowS drivers.

Each class implements:
- `assemble_problem(...)` to build fixed forms/solvers,
- `step(u_old)` to compute one constrained update from the current iterate.
"""

import abc
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt


class gradient(abc.ABC):
    """
    General class that implements Sobolev Gradient to solve the Gross-Pitaevski equation
    
    Args:
        - W: function space of the solution
        - bcs: list of boundary conditions
        - u: Trial function
        - w: Test function
        - uh: solution
        - h: step of spatial discretization
    """
    def __init__(self, W, bcs, h, beta, v, name):
        '''
        Constructor of the class
        
        :param W (fd space): function space of the solution
        :param bcs (list): list of boundary conditions
        :param h (float): step of spatial discretization
        '''
        self.name = name

        # solution space
        self.W = W
        self.u = fd.TrialFunction(self.W)
        self.w = fd.TestFunction(self.W)
        self.uh = fd.Function(self.W)

        # boundary conditions
        self.bcs = bcs

        # discretization parameters
        self.h = h

        # Parameters of the nonlinear GPE problem.
        self.beta = beta
        self.v = v

    @abc.abstractmethod
    def assemble_problem(self):
        '''
        Inizialize and assembles all forms related to the minimization
        '''
        pass

    @abc.abstractmethod
    def step(self, u_old):
        '''
        Step of the minimization that starts from u_old

        :param u_old (fd.Function): solution at the previous time step
        '''
        pass

class gradient_L2_fully_expli(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v,'L2 explicit')

    def assemble_problem(self, tau, lump = False):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        :param lump (bool): decides if the system is solved with lumping or not
        '''
        if tau is None:
            raise ValueError("The time step tau must be provided for the L2 explicit gradient")
        
        self.tau = fd.Constant(tau)

        # Assemble mass matrix (optionally mass-lumped) used at each explicit step.
        if lump:
            self.M = fd.assemble(self.u * self.w * fd.dx, bcs = self.bcs, form_compiler_parameters={"quadrature_rule": "KMV","quadrature_degree": self.W.ufl_element().degree()})

            self.solver_Mass = fd.LinearSolver(self.M)
        else:
            self.M = fd.assemble(self.u * self.w * fd.dx, bcs = self.bcs)
            # NOTE: the default solver of firedrake is lu so this parameter are useless
            self.solver_Mass = fd.LinearSolver(self.M) # solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        # one solution of the linear system used only to compute the LU factorization
        self.solver_Mass.solve(self.uh, fd.assemble( self.w * fd.dx))

        
    def step(self, u_old):
        '''
        Implements the step of L2 gradient
        '''
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        # Evaluate energy-like scaling terms used by the explicit update.
        # intE = ∫ 0.5 ∇u∇u + Vu^2 + β |u|^2 u u dx
        intE = fd.assemble(0.5 * fd.dot(fd.grad(u_old), fd.grad(u_old)) * fd.dx + self.v * u_old * u_old * fd.dx + self.beta * abs(u_old)**2 * u_old * u_old * fd.dx)
        # intR = ∫ u^2 dx
        intR = fd.assemble(u_old * u_old * fd.dx)

        rhs = fd.assemble( u_old * self.w * fd.dx \
            - self.tau * 0.5 * fd.dot(fd.grad(u_old), fd.grad(self.w)) * fd.dx \
            - self.tau * self.v * u_old * self.w * fd.dx \
            - self.tau * self.beta * (u_old ** 2 * u_old) * self.w * fd.dx \
            + self.tau * intE/intR * u_old * self.w * fd.dx)
        
        # M u^n+1 = Mu^n - τ ( 0.5 * A + v M + β N(u^2) )u^n + τ intE/intR * Mu^n
        self.solver_Mass.solve(self.uh, rhs)


class gradient_L2(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v, 'L2')

    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''

        if tau is None:
            raise ValueError("The time step tau must be provided for the L2 gradient")
        
        self.tau = fd.Constant(tau)

        # Linear part fixed across iterations; nonlinear term is added in step().
        self.a = self.u * self.w * fd.dx \
            + self.tau * 0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
            + self.tau * self.v * self.u * self.w * fd.dx
        
    def step(self, u_old):
        '''
        Implements the step of L2 gradient
        '''
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        rhs = u_old * self.w * fd.dx 
        
        problem = fd.LinearVariationalProblem(self.a + self.tau * self.beta * (u_old **2 * self.u) * self.w * fd.dx,
                                                rhs,
                                                self.uh,
                                                self.bcs)
        solver =  fd.LinearVariationalSolver(problem)#, solver_parameters=param)
        solver.solve()


class gradient_L2_P(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v, 'L2_P')

    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        if tau is None:
            raise ValueError("The time step tau must be provided for the L2_P gradient")
        
        self.tau = fd.Constant(tau)

        self.a = self.u * self.w * fd.dx \
            + self.tau * 0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
            + self.tau * self.v * self.u * self.w * fd.dx
        
    def step(self, u_old):
        '''
        Implements the step of L2 gradient
        '''
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        # Dynamic scalar shift to improve stability/monotonicity of L2 updates.
        shift = (fd.assemble((0.5 * fd.dot(fd.grad(u_old), fd.grad(u_old)) + self.v * u_old * u_old + self.beta * abs(u_old)**2 * u_old * u_old)* fd.dx))/fd.assemble(u_old * u_old * fd.dx)
        rhs = (1 + self.tau * shift) * u_old * self.w * fd.dx
        
        problem = fd.LinearVariationalProblem(self.a + self.tau * self.beta * (u_old **2 * self.u) * self.w * fd.dx,
                                                rhs,
                                                self.uh,
                                                self.bcs)
        solver =  fd.LinearVariationalSolver(problem)#, solver_parameters=param)
        solver.solve()


class gradient_H1(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v,'H1')

    def assemble_problem(self,tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        if tau is None:
            raise ValueError("The time step tau must be provided for the H1 gradient")
        
        self.tau = fd.Constant(tau)

        self.A = fd.assemble(0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx, bcs = self.bcs)

        # Pre-factorized stiffness solve for repeated Riesz projections.
        self.R_u = fd.Function(self.W)
        self.gradE = fd.Function(self.W)
        self.solver_Stiffnes = fd.LinearSolver(self.A)#, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        # used only to compute the LU factorization
        self.solver_Stiffnes.solve(self.R_u, fd.assemble( self.w * fd.dx))

    def step(self, u_old):
        '''
        Implements a step of the gradient H1
        '''
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        # Riesz map of u_old.
        rhs_R = fd.assemble(u_old * self.w * fd.dx)

        self.solver_Stiffnes.solve(self.R_u, rhs_R)
        
        # Riesz map of the energy gradient.
        rhs_E = fd.assemble(0.5 * fd.dot(fd.grad(u_old), fd.grad(self.w)) * fd.dx \
                + self.v * u_old * self.w * fd.dx \
                + self.beta * abs(u_old) **2 * u_old * self.w * fd.dx)

        self.solver_Stiffnes.solve(self.gradE, rhs_E)

        # Constrained descent update in H1 geometry.

        intE = fd.assemble(self.gradE * u_old * fd.dx)
        intR = fd.assemble(self.R_u * u_old * fd.dx)

        self.uh.assign(u_old - self.tau * self.gradE + self.tau * intE/intR * self.R_u)


class gradient_a0(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v,'a0')

    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        if tau is None:
            raise ValueError("The time step tau must be provided for the a_0 gradient")
        
        self.tau = fd.Constant(tau)

        # Solver used by both Riesz projections in the a0 metric.
        self.R_u = fd.Function(self.W)
        self.R_u2u = fd.Function(self.W)

        self.A = fd.assemble(0.5 * fd.dot(fd.grad(self.u), fd.grad(self.w)) * fd.dx\
                                + self.v * self.u * self.w * fd.dx, bcs = self.bcs)

        self.solver_Stiffness = fd.LinearSolver(self.A)#, solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly', 'ksp_rtol': 1e-05, 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps', 'pc_factor_mat_mumps_icntl_14': 200, "ksp_view": None})

        # used only to compute the LU factorization
        self.solver_Stiffness.solve(self.R_u, fd.assemble( self.w * fd.dx))



    def step(self, u_old):
        '''
        Impelements one step of the a_0 gradient 
        '''
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        # Compute Riesz projections entering the closed-form update.
        rhs_Ru = fd.assemble(u_old * self.w * fd.dx)
        self.solver_Stiffness.solve(self.R_u, rhs_Ru)

        rhs_Ru2u = fd.assemble(self.beta * abs(u_old) ** 2 * u_old * self.w * fd.dx)

        self.solver_Stiffness.solve(self.R_u2u,rhs_Ru2u)

        # Constrained update in the a0 metric.
        intE = fd.assemble((u_old + self.R_u2u) * u_old * fd.dx)
        intR = fd.assemble(self.R_u * u_old * fd.dx)

        self.uh.assign((1 - self.tau) * u_old  - self.tau * self.R_u2u + self.tau * intE/intR * self.R_u)


class gradient_az(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v,'az')

    def assemble_problem(self, tau = None):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param tau (float): time step
        :param u_ref (fd.Function): reference solution
        '''
        if tau is None:
            raise ValueError("The time step tau must be provided for the a_z gradient")
        
        self.tau = fd.Constant(tau)

        # Build the linear part used in the nonlinear Riesz solve at each step.
        self.R_u = fd.Function(self.W)

        self.a = 0.5 * fd.inner(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
                + self.v * self.u * self.w * fd.dx
        
    
    def step(self, u_old):
        '''
        Implements one step of the a_z gradient
        '''
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        # Solve nonlinear Riesz system associated with current iterate.
        rhs_R = fd.inner(u_old , self.w) * fd.dx

        problem_R = fd.LinearVariationalProblem(self.a + self.beta * abs(u_old)**2 * self.u * self.w * fd.dx,
                                                rhs_R,
                                                self.R_u,
                                                self.bcs)
        solver_R = fd.LinearVariationalSolver(problem_R)#, solver_parameters={"ksp_view": None})
        solver_R.solve()

        # Update as a convex combination of u_old and normalized Riesz direction.
        # intE = fd.assemble(u_old * u_old * fd.dx) # should be 1 in exact aritmetic because it's simply the norm of hte previous uh
        intR = fd.assemble(self.R_u * u_old * fd.dx)

        self.uh.assign((1 - self.tau) * u_old + self.tau * 1/intR * self.R_u)

class gradient_az_ada(gradient):
    def __init__(self, W, bcs, h, beta, v):
        super().__init__(W, bcs, h, beta, v, 'az_ada')
    
    def assemble_problem(self, tau = None):
        '''
        Allocate and assembles forms and minimization quantities

        :param u0 (fd.Function): initial guess
        :param u_ref (fd.Function): reference solution
        '''
        # Build common forms; tau is selected adaptively inside step().
        self.R_u = fd.Function(self.W)

        self.a = 0.5 * fd.inner(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
                + self.v * self.u * self.w * fd.dx

        self.tau_history = []

    def golden_search(self, func, a = 0.01, b = 2.0, tol = 1e-5):
        """
        Golden-section search
        to find the minimum of f on [a,b]

        * f: a strictly unimodal function on [a,b]

        Example:
        >>> def f(x): return (x - 2) ** 2
        >>> x = gss(f, 1, 5)
        >>> print(f"{x:.5f}")
        2.00000

        """
        # Golden-section search over scalar step size interval [a, b].
        invphi = (fd.sqrt(5) - 1) / 2  # 1 / phi

        # x = np.linspace(0, 10, 100)
        # fig, ax = plt.subplots()
        # ax.plot(x, func(x))
        # plt.legend()
        # plt.show()

        while b - a > tol:
            c = b - (b - a) * invphi
            d = a + (b - a) * invphi
            if func(c) < func(d):
                b = d
            else:  # func(c) > func(d) to find the maximum
                a = c
        return (b + a) / 2

    def step(self, u_old):
        '''
        Implements one step of the a_z gradient with adaptive step size
        '''
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")

        # Compute nonlinear Riesz direction first.
        rhs_R = fd.inner(u_old , self.w) * fd.dx

        problem_R = fd.LinearVariationalProblem(self.a + self.beta * abs(u_old)**2 * self.u * self.w * fd.dx,
                                                rhs_R,
                                                self.R_u,
                                                self.bcs)
        solver_R = fd.LinearVariationalSolver(problem_R)#, solver_parameters={"ksp_view": None})
        solver_R.solve()


        # Build polynomial/rational energy model along the update direction.
        # intE = fd.assemble(u_old * u_old * fd.dx) # should be 1 in exact aritmetic because it's simply the norm of hte previous uh
        intR = fd.assemble(self.R_u * u_old * fd.dx)
        
        alpha0 = fd.assemble(0.5 * fd.inner(fd.grad(u_old), fd.grad(u_old)) * fd.dx \
                            + self.v * u_old * u_old * fd.dx)
        alpha1 = 2 / intR * fd.assemble(0.5 * fd.inner(fd.grad(self.R_u), fd.grad(u_old)) * fd.dx \
                            + self.v * self.R_u * u_old * fd.dx)
        alpha2 = 1/intR**2 * fd.assemble(0.5 * fd.inner(fd.grad(self.R_u), fd.grad(self.R_u)) * fd.dx \
                            + self.v * self.R_u * self.R_u * fd.dx)
        beta0 = fd.assemble(self.beta * 0.5 * (u_old)**4 * fd.dx) 
        beta1 = fd.assemble(self.beta * 2 * (u_old)**3 * self.R_u /intR * fd.dx)
        beta2 = fd.assemble(self.beta * 3 * (u_old)**2 * self.R_u**2 /intR**2 * fd.dx)
        beta3 = fd.assemble(self.beta * 2 * u_old * self.R_u**3 /intR**3 * fd.dx)
        beta4 = fd.assemble(self.beta * 0.5 * self.R_u**4 /intR**4 * fd.dx)
        gamma0 = fd.assemble(u_old**2 * fd.dx)
        gamma1 = fd.assemble(2 * u_old * self.R_u /intR * fd.dx)
        gamma2 = fd.assemble(self.R_u**2 /intR**2 * fd.dx)

        def den(x):
            return ((1- x)**2 * gamma0 + (1-x)* x * gamma1 + x**2 * gamma2)**0.5

        def f(x):
            d2 = den(x)**2
            d4 = den(x)**4
            return (alpha0 / d2 * (1-x)**2
                    + alpha1 / d2 * (1-x)*x
                    + alpha2 / d2 * x**2
                    + beta0 / d4 * (1-x)**4 
                    + beta1 / d4 * (1-x)**3 * x 
                    + beta2 / d4 * (1-x)**2 * x**2 
                    + beta3 / d4 * (1-x) * x**3 
                    + beta4 / d4 * x**4)

        # Choose tau adaptively by 1D minimization of the model energy.
        self.tau = fd.Constant(self.golden_search(f))

        self.uh.assign((1 - self.tau) * u_old + self.tau * 1/intR * self.R_u)

        self.tau_history.append(float(self.tau))