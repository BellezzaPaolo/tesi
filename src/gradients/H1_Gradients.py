'''
This module implements the H1 gradient of the cost functional.
'''

import firedrake as fd
from .Gradients import Gradient

class Gradient_H1(Gradient):
    '''
    This class implements the H1 gradient of the cost functional.

    X = H0^1(Omega) and inner product (.,.)_H^1 = ∫ ∇. ∇. dx, so:

    The Riesz representative is given by the solution of the following problem:
    (R_{H0^1}(z),w)_{H^1} = ∫ ∇ R_{H0^1}(z) ∇ w dx = (z,w)_{L^2}

    The gradient is given by the solution of this linear system:
    (∇E_{H^1}(z),v)_{H^1} = ∫ ∇ ∇E_{H^1}(z) ∇ v dx = ∫ 1/2 ∇ v ∇ w + V v w + β |v|^2 v w dx 
    '''

    def __init__(self, W, bcs, beta, v, type_discr):
        super().__init__(W, bcs, beta, v, 'H1', type_discr)

    def compute_Reiz_representative(self, u_old):
        '''
        For the H1 gradient, the Riesz representative is computed at each step solving the linear variational problem.
        '''

        rhs_R = fd.assemble(u_old * self.w * fd.dx)

        self.solver_Stiffnes.solve(self.R_u, rhs_R)
        
        return self.R_u
    
    def compute_gradient(self, u_old):
        '''
        For the H1 gradient, the gradient is computed at each step solving the linear variational problem.
        '''
        
        rhs_E = fd.assemble(0.5 * fd.inner(fd.grad(u_old), fd.grad(self.w)) * fd.dx \
                + self.v * u_old * self.w * fd.dx \
                + self.beta * abs(u_old) **2 * u_old * self.w * fd.dx)

        self.solver_Stiffnes.solve(self.gradE, rhs_E)

        return self.gradE

    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        # TODO: add support for adaptive time step, but the cost of H1 gradient is more than a single step of az or L2 so it's not implemented yet.
        if tau is None:
            raise ValueError("The time step tau must be provided for the H1 gradient")
        
        self.tau = fd.Constant(tau)

        self.A = fd.assemble(fd.inner(fd.grad(self.u), fd.grad(self.w)) * fd.dx, bcs = self.bcs)

        # Pre-factorized stiffness solve for repeated Riesz projections.
        self.R_u = fd.Function(self.W)
        self.gradE = fd.Function(self.W)
        self.solver_Stiffnes = fd.LinearSolver(self.A)#, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        # used only to compute the LU factorization
        self.solver_Stiffnes.solve(self.R_u, fd.assemble( self.w * fd.dx))

    def step(self, u_old):
        '''
        Perform a single gradient flow step.

        :param u_old (fd function): the current state of the system
        :return: the new state of the system after one gradient flow step
        '''
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        self.compute_Reiz_representative(u_old)

        self.compute_gradient(u_old)

        self.alpha0 = fd.assemble(0.5 * fd.inner(fd.grad(u_old), fd.grad(u_old)) * fd.dx \
                            + self.v * u_old * u_old * fd.dx)
        self.beta0 = fd.assemble(self.beta * 0.5 * (u_old)**4 * fd.dx)
        self.gamma0 = fd.assemble(u_old**2 * fd.dx)

        return 0.5 * (self.alpha0/self.gamma0**2 + self.beta0/self.gamma0**4)

class Gradient_H1_explicit(Gradient_H1):
    '''
    This class implements the explicit gradient flow for the H1 gradient.
    The step is given by:
    z^{n+1} = z^n - τ ∇E(z^{n}) - τ (∇E(z^n),z^n)_{L^2}/(R_{H^1}(z^n),z^n)_{L^2} R_{H^1}(z^n)
    '''

    def __init__(self, W, bcs, beta, v):
        super().__init__(W, bcs, beta, v, 'explicit')

    def step(self, u_old):
        '''
        Perform a single explicit gradient flow step.

        :param u_old (fd function): the current state of the system
        :return: the new state of the system after one explicit gradient flow step
        '''
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        E_old = super().step(u_old)

        # Compute the projection coefficient.
        intR = fd.assemble(self.R_u * u_old * fd.dx)
        intE = fd.assemble(self.gradE * u_old * fd.dx)

        # Update the solution with an explicit step and projection.
        self.uh.assign(u_old - self.tau * self.gradE + self.tau * intE/intR * self.R_u)

        return E_old