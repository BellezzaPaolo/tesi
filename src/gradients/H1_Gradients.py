'''
This module implements the H1 gradient of the cost functional.
'''

import firedrake as fd
from .Gradients import Gradient

try:
    from line_profiler import profile
except NameError:
    def profile(function):
        return function


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

    @profile
    def compute_Reiz_representative(self, u_old):
        '''
        For the H1 gradient, the Riesz representative is computed at each step solving the linear variational problem.
        '''
        fd.assemble(u_old * self.w * fd.dx, tensor = self.rhs_R)

        self.solver_Stiffnes.solve(self.R_u, self.rhs_R)
        
        return self.R_u
    
    @profile
    def compute_gradient(self, u_old):
        '''
        For the H1 gradient, the gradient is computed at each step solving the linear variational problem.
        '''
        self.u_current.dat.data[:] = u_old.dat.data_ro[:]
        fd.assemble(self.rhs_e, tensor = self.rhs_E)

        self.solver_Stiffnes.solve(self.gradE, self.rhs_E)

        return self.gradE

    @profile
    def assemble_problem(self, tau):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        self.A = fd.assemble(fd.inner(fd.grad(self.u), fd.grad(self.w)) * fd.dx, bcs = self.bcs)

        # Pre-factorized stiffness solve for repeated Riesz projections.
        self.R_u = fd.Function(self.W)
        self.gradE = fd.Function(self.W)
        self.solver_Stiffnes = fd.LinearSolver(self.A)#, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        
        self.rhs_R = fd.assemble( self.w * fd.dx)

        self.u_current = fd.Function(self.W)
        self.rhs_e = 0.5 * fd.inner(fd.grad(self.u_current), fd.grad(self.w)) * fd.dx \
                + self.v * self.u_current * self.w * fd.dx \
                + self.beta * self.u_current**2 * self.u_current * self.w * fd.dx
        self.rhs_E = fd.assemble(self.rhs_e)

        # used only to compute the LU factorization
        self.solver_Stiffnes.solve(self.R_u, self.rhs_R)

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
            _ = self.compute_integral_adaptivity(self.R_u.dat.dat_ro, self.proj_term.dat.data_ro)

    @profile
    def step(self, u_old):
        '''
        Perform a single gradient flow step.

        :param u_old (fd function): the current state of the system
        :return: the new state of the system after one gradient flow step
        '''
        self.compute_Reiz_representative(u_old)

        self.compute_gradient(u_old)

        self.compute_integrals(u=u_old.dat.data_ro, R=self.R_u.dat.data_ro, gradE=self.gradE.dat.data_ro)

        self.intE_intR = self.integrals[1] / self.integrals[0]

        self.proj_term.dat.data[:] = - self.gradE.dat.data_ro[:] + self.intE_intR * self.R_u.dat.data_ro[:]

        if self.adaptivity:
            self.compute_integral_adaptivity(u=u_old.dat.data_ro, proj_term=self.proj_term.dat.data_ro)

            self.alpha0 = self.integrals[2]
            self.alpha1 = 2 * self.integrals[3]
            self.alpha2 = self.integrals[4]
            
            self.beta0 = self.beta * 0.5 * self.integrals[5]
            self.beta1 = self.beta * 2 * self.integrals[6]
            self.beta2 = self.beta * 3 * self.integrals[7]
            self.beta3 = self.beta * 2 * self.integrals[8]
            self.beta4 = self.beta * 0.5 * self.integrals[9]

            self.gamma0 = self.integrals[10]
            self.gamma1 = 2 * self.integrals[11]
            self.gamma2 = self.integrals[12]

class Gradient_H1_explicit(Gradient_H1):
    '''
    This class implements the explicit gradient flow for the H1 gradient.
    The step is given by:
    z^{n+1} = z^n - τ ∇E(z^{n}) - τ (∇E(z^n),z^n)_{L^2}/(R_{H^1}(z^n),z^n)_{L^2} R_{H^1}(z^n)
    '''

    def __init__(self, W, bcs, beta, v):
        super().__init__(W, bcs, beta, v, 'explicit')

    @profile
    def step(self, u_old):
        '''
        Perform a single explicit gradient flow step.

        :param u_old (fd function): the current state of the system
        :return: the new state of the system after one explicit gradient flow step
        '''        
        super().step(u_old)

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

        # self.M.M.handle.mult(self.uh.data.vec_ro, self.M_u)
        # return self.uh.dat.vec_ro.dot(self.M_u) ** 0.5