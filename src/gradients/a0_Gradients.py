'''
This module implements the a0 gradient of the cost functional.
'''

import firedrake as fd
from .Gradients import Gradient
import time

try:
    from line_profiler import profile
except NameError:
    def profile(function):
        return function


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

    @profile
    def compute_Reiz_representative(self, u_old):
        '''
        For the a0 gradient, the Riesz representative is computed at each step solving the linear variational problem.
        '''
        
        fd.assemble(u_old * self.w * fd.dx, tensor = self.rhs_Ru)
        self.solver_Stiffness.solve(self.R_u, self.rhs_Ru)

        return self.R_u
    
    @profile
    def compute_gradient(self, u_old):
        '''
        For the a0 gradient, the gradient is computed at each step solving the linear variational problem.
        To be rigorous here we compute the Reiss representative of the nonlinear term, and the gradient is given by this term plus the previous solution.
        '''
        fd.assemble(self.beta * u_old**2 * u_old * self.w * fd.dx, tensor = self.rhs_Ru2u)

        self.solver_Stiffness.solve(self.R_u2u,self.rhs_Ru2u)

        self.gradE.dat.data[:] = u_old.dat.data_ro[:] + self.R_u2u.dat.data_ro[:]

        return self.gradE

    @profile
    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        # Solver used by both Riesz projections in the a0 metric.
        self.R_u = fd.Function(self.W)
        self.R_u2u = fd.Function(self.W)
        self.gradE = fd.Function(self.W)

        self.a = 0.5 * fd.inner(fd.grad(self.u), fd.grad(self.w)) * fd.dx\
                                + self.v * self.u * self.w * fd.dx

        self.A = fd.assemble(self.a, bcs = self.bcs)

        self.solver_Stiffness = fd.LinearSolver(self.A)#, solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly', 'ksp_rtol': 1e-05, 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps', 'pc_factor_mat_mumps_icntl_14': 200, "ksp_view": None})

        self.rhs_Ru = fd.assemble( self.w * fd.dx)
        self.rhs_Ru2u = fd.assemble( self.w * fd.dx)
        
        # used only to compute the LU factorization
        self.solver_Stiffness.solve(self.R_u, self.rhs_Ru)

        if tau is None:
            self.adaptivity = True
            self.tau = 1.0 # dummy value, the actual time step will be computed at each step.
            self.tau_history = []
        else:
            self.adaptivity = False
            self.tau = tau

        self.proj_term = fd.Function(self.W)

        super().assemble_problem(compute_intE = True, compute_energy = False, adaptivity = self.adaptivity)

        # warm up with dummy inputs
        _ = self.compute_integrals(self.R_u.dat.data_ro, self.R_u.dat.data_ro, self.gradE.dat.data_ro)
       
        if self.adaptivity:
            _ = self.compute_integral_adaptivity(self.R_u.dat.data_ro, self.proj_term.dat.data_ro)
        
    @profile
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

        # E_old = 0.5 * (self.alpha0/self.gamma0**2 + self.beta0/self.gamma0**4)

        self.compute_integrals(u=u_old.dat.data_ro, R=self.R_u.dat.data_ro, gradE=self.gradE.dat.data_ro)

        self.intE_intR = self.integrals[1] / self.integrals[0]

        self.proj_term.dat.data[:] = - self.gradE.dat.data_ro[:] + self.intE_intR * self.R_u.dat.data_ro[:]

        if self.adaptivity:
            self.compute_integral_adaptivity(u=u_old.dat.data_ro, proj_term=self.proj_term.dat.data_ro)

            self.alpha0 = self.integrals_adaptivity[0]
            self.alpha1 = 2 * self.integrals_adaptivity[1]
            self.alpha2 = self.integrals_adaptivity[2]
            
            self.beta0 = self.beta * 0.5 * self.integrals_adaptivity[3]
            self.beta1 = self.beta * 2 * self.integrals_adaptivity[4]
            self.beta2 = self.beta * 3 * self.integrals_adaptivity[5]
            self.beta3 = self.beta * 2 * self.integrals_adaptivity[6]
            self.beta4 = self.beta * 0.5 * self.integrals_adaptivity[7]

            self.gamma0 = self.integrals_adaptivity[8]
            self.gamma1 = 2 * self.integrals_adaptivity[9]
            self.gamma2 = self.integrals_adaptivity[10]
        # t2 = time.time()

        # print(f"Time to compute energy and coefficients for the golden search: {t2-t1:.2f} seconds")

        # return E_old


    
class Gradient_a0_explicit(Gradient_a0):
    '''
    This class implements the a0 gradient by discretizing the gradient flow with Forward Euler:
    
    z^{n+1} =z^n - τ z^n - τ R_{a_0}(β |z|^2z) + τ (z^n + R_{a_0}(β |z^n|^2 z^n), z^n)_{L^2}/(R_{a_0}(z^n),z^n)_{L^2} R_{a_0}(z^n)
    '''
    def __init__(self, W, bcs, beta, v):
        super().__init__(W, bcs, beta, v, 'explicit')
      
    @profile      
    def step(self, u_old):
        '''
        Perform a step of the gradient flow.

        :param u_old (Function): the current iterate
        '''
        # t3 = time.time()
        super().step(u_old)
        # t4 = time.time()
        # print(f"Time to compute the step: {t4-t3:.2f} seconds")

        if self.adaptivity:
            def den(x):
                return ( self.gamma0 + x * self.gamma1 + x**2 * self.gamma2)**0.5

            def f(x):
                d2 = den(x)**2
                d4 = den(x)**4
                return (self.alpha0 / d2
                        + self.alpha1 / d2 *x
                        + self.alpha2 / d2 * x**2
                        + self.beta0 / d4 
                        + self.beta1 / d4 * x 
                        + self.beta2 / d4 * x**2 
                        + self.beta3 / d4 * x**3 
                        + self.beta4 / d4 * x**4) 
            # t6 = time.time()
            # Choose tau adaptively by 1D minimization of the model energy.
            self.tau = self.golden_search(f, a= 0.01, b = 1.0)
            # t7 = time.time()
            # print(f"Time to compute the golden search: {t7-t6} seconds")
            self.tau_history.append(self.tau) #.values()[0])

        # t8 = time.time()
        self.uh.dat.data[:] = u_old.dat.data_ro[:]  + self.tau * self.proj_term.dat.data_ro[:]
        # t9 = time.time()
        # print(f"Time to update the solution: {t9-t8} seconds")

        # return E_old