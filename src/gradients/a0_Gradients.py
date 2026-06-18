'''
This module implements the a0 gradient of the cost functional.
'''

import firedrake as fd
from .Gradients import Gradient
import time

class Gradient_a0(Gradient):
    '''
    This class implements the a0 gradient of the cost functional.

    X = H0^1(Omega) and inner product a_0(.,.) = ∫ 1/2 ∇. ∇. + V . . dx, so:

    The Riesz representative is given by the solution of the following problem:
    a_0(R_{a_0}(z),w) = ∫ 1/2 ∇ R_{a_0}(z) ∇ w + V R_{a_0}(z) w dx = (z,w)_{L^2}

    The gradient is given by the solution of this linear system:
    a_0(∇E_{a_0}(z),v) = ∫ 1/2 ∇E_{a_0}(z) ∇ v + V ∇E_{a_0}(z) v dx = ∫ 1/2 ∇ v ∇ w + V v w + β |v|^2 v w dx 
    '''
    
    def __init__(self, W, bcs, beta, v, type_discr):
        super().__init__(W, bcs, beta, v, 'a0', type_discr)

    def compute_Reiz_representative(self, u_old):
        '''
        For the a0 gradient, the Riesz representative is computed at each step solving the linear variational problem.
        '''
        
        rhs_Ru = fd.assemble(u_old * self.w * fd.dx)
        self.solver_Stiffness.solve(self.R_u, rhs_Ru)

        return self.R_u
    
    def compute_gradient(self, u_old):
        '''
        For the a0 gradient, the gradient is computed at each step solving the linear variational problem.
        To be rigorous here we compute the Reiss representative of the nonlinear term, and the gradient is given by this term plus the previous solution.
        '''
        rhs_Ru2u = fd.assemble(self.beta * abs(u_old) ** 2 * u_old * self.w * fd.dx)

        self.solver_Stiffness.solve(self.R_u2u,rhs_Ru2u)

        return u_old

    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        if tau is None:
            self.adaptivity = True
            self.tau = fd.Constant(1.0) # dummy value, the actual time step will be computed at each step.
            self.tau_history = []
            self.proj_term = fd.Function(self.W)
        else:
            self.adaptivity = False
            self.tau = fd.Constant(tau)

        self.shift = fd.Constant(0.0)

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
        Perform a step of the gradient flow.

        :param u_old (Function): the current iterate
        '''
        # t0 = time.time()
        self.compute_Reiz_representative(u_old)
        self.compute_gradient(u_old)
        # t1 = time.time()
        # print(f"Time to compute Riesz representative and gradient: {t1-t0:.2f} seconds")

        self.alpha0 = fd.assemble(0.5 * fd.inner(fd.grad(u_old), fd.grad(u_old)) * fd.dx \
                            + self.v * u_old * u_old * fd.dx)
        self.beta0 = fd.assemble(self.beta * 0.5 * (u_old)**4 * fd.dx)
        self.gamma0 = fd.assemble(u_old**2 * fd.dx)

        E_old = 0.5 * (self.alpha0/self.gamma0**2 + self.beta0/self.gamma0**4)

        self.shift = fd.assemble((u_old + self.R_u2u) * u_old * fd.dx) / fd.assemble(self.R_u * u_old * fd.dx)

        if self.adaptivity:
            self.proj_term.assign(-self.R_u2u + self.shift * self.R_u)

            self.alpha1 = fd.assemble(2 * 0.5 * fd.inner(fd.grad(u_old), fd.grad(self.proj_term)) * fd.dx + \
                                      2 * self.v * u_old * (self.proj_term) * fd.dx)
            self.alpha2 = fd.assemble(0.5 * fd.inner(fd.grad(self.proj_term), fd.grad(self.proj_term)) * fd.dx \
                                + self.v * (self.proj_term) * (self.proj_term) * fd.dx)
            self.beta1 = fd.assemble(self.beta * 2 * u_old**3 * (self.proj_term) * fd.dx)
            self.beta2 = fd.assemble(self.beta * 3 * u_old**2 * (self.proj_term)**2 * fd.dx)
            self.beta3 = fd.assemble(self.beta * 2 * u_old * (self.proj_term)**3 * fd.dx)
            self.beta4 = fd.assemble(self.beta * 0.5 * (self.proj_term)**4 * fd.dx)
            self.gamma1 = fd.assemble(2 * u_old * (self.proj_term) * fd.dx)
            self.gamma2 = fd.assemble((self.proj_term)**2 * fd.dx)
        # t2 = time.time()

        # print(f"Time to compute energy and coefficients for the golden search: {t2-t1:.2f} seconds")

        return E_old


    
class Gradient_a0_explicit(Gradient_a0):
    '''
    This class implements the a0 gradient by discretizing the gradient flow with Forward Euler:
    
    z^{n+1} =z^n - τ z^n - τ R_{a_0}(β |z|^2z) + τ (z^n + R_{a_0}(β |z^n|^2 z^n), z^n)_{L^2}/(R_{a_0}(z^n),z^n)_{L^2} R_{a_0}(z^n)
    '''
    def __init__(self, W, bcs, beta, v):
        super().__init__(W, bcs, beta, v, 'explicit')
            
    def step(self, u_old):
        '''
        Perform a step of the gradient flow.

        :param u_old (Function): the current iterate
        '''
        # t3 = time.time()
        E_old = super().step(u_old)
        # t4 = time.time()
        # print(f"Time to compute the step: {t4-t3:.2f} seconds")

        if self.adaptivity:
            def den(x):
                return ((1- x)**2 * self.gamma0 + (1-x)* x * self.gamma1 + x**2 * self.gamma2)**0.5

            def f(x):
                d2 = den(x)**2
                d4 = den(x)**4
                return (self.alpha0 / d2 * (1-x)**2
                        + self.alpha1 / d2 * (1-x)*x
                        + self.alpha2 / d2 * x**2
                        + self.beta0 / d4 * (1-x)**4 
                        + self.beta1 / d4 * (1-x)**3 * x 
                        + self.beta2 / d4 * (1-x)**2 * x**2 
                        + self.beta3 / d4 * (1-x) * x**3 
                        + self.beta4 / d4 * x**4) 
            # t6 = time.time()
            # Choose tau adaptively by 1D minimization of the model energy.
            self.tau = fd.Constant(self.golden_search(f, a= 0.01, b = 1.0))
            # t7 = time.time()
            # print(f"Time to compute the golden search: {t7-t6} seconds")
            self.tau_history.append(self.tau.values()[0])

        # t8 = time.time()
        self.uh.assign((1 - self.tau) * u_old  - self.tau * self.R_u2u + self.tau * self.shift * self.R_u)
        # t9 = time.time()
        # print(f"Time to update the solution: {t9-t8} seconds")

        return E_old