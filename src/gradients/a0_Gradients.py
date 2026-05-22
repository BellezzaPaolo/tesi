'''
This module implements the a0 gradient of the cost functional.
'''

import firedrake as fd
from .Gradients import Gradient

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
        #TODO: add support for adaptive time step, but the cost of a0 gradient is more than a single step of az or L2 so it's not implemented yet.
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
        Perform a step of the gradient flow.

        :param u_old (Function): the current iterate
        '''
        self.compute_Reiz_representative(u_old)
        self.compute_gradient(u_old)

        self.alpha0 = fd.assemble(0.5 * fd.inner(fd.grad(u_old), fd.grad(u_old)) * fd.dx \
                            + self.v * u_old * u_old * fd.dx)
        self.beta0 = fd.assemble(self.beta * 0.5 * (u_old)**4 * fd.dx)
        self.gamma0 = fd.assemble(u_old**2 * fd.dx)

        E_old = 0.5 * (self.alpha0/self.gamma0**2 + self.beta0/self.gamma0**4)
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
        
        E_old = super().step(u_old)

        intE = fd.assemble((u_old + self.R_u2u) * u_old * fd.dx)
        intR = fd.assemble(self.R_u * u_old * fd.dx)

        self.uh.assign((1 - self.tau) * u_old  - self.tau * self.R_u2u + self.tau * intE/intR * self.R_u)

        return E_old