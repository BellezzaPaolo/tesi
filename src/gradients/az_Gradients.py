'''
This module implements the az gradient of the cost functional.
'''

import firedrake as fd
from .Gradients import Gradient
import time

try:
    from line_profiler import profile
except NameError:
    def profile(function):
        return function

class Gradient_az(Gradient):
    '''
    This class implements the az gradient of the cost functional.

    X = H0^1(Omega) and inner product a_z(.,.) = ∫ 1/2 ∇. ∇. + V . . + β |z|^2 . . dx, so:

    The Riesz representative is given by the solution of the following problem:
    a_z(R_{a_z}(z),w) = ∫ 1/2 ∇ R_{a_z}(z) ∇ w + V R_{a_z}(z) w + β |z|^2 * R_{a_z}(z) w  dx = (z,w)_{L^2}

    The gradient becomes simply:
    ∇_{a_z} E(z) = z
    '''
    
    def __init__(self, W, bcs, beta, v, type_discr):
        super().__init__(W, bcs, beta, v, 'az', type_discr)

    @profile
    def compute_Reiz_representative(self, u_old):
        '''
        For the az gradient, the Riesz representative is computed at each step solving the linear variational problem.
        '''
        fd.assemble(fd.inner(u_old , self.w) * fd.dx, tensor = self.rhs_R)

        self.nonlinear_coefficient.dat.data[:] = self.beta * u_old.dat.data_ro[:]**2
        
        fd.assemble(self.s, bcs=self.bcs, tensor=self.S)
        
        self.Solver_System.solve(self.R_u, self.rhs_R)

        return self.R_u
    
    @profile
    def compute_gradient(self, u_old):
        Warning("The gradient for the az gradient is the function itself, so this method is not needed.")

        return u_old
    
    @profile
    def assemble_problem(self, tau = None):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        # Built the Reisz representative and the linear part of the variational problem
        self.R_u = fd.Function(self.W)

        self.nonlinear_coefficient = fd.Function(self.W)

        self.s = 0.5 * fd.inner(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
                + self.v * self.u * self.w * fd.dx \
                + self.nonlinear_coefficient * self.u * self.w * fd.dx
        self.S = fd.assemble(self.s, bcs = self.bcs)

        self.Solver_System = fd.LinearSolver(self.S)

        self.rhs_R = fd.assemble(self.w * fd.dx)

        if tau is None:
            self.adaptivity = True
            self.tau = 1.0 # dummy value, the actual time step will be computed at each step.
            self.tau_history = []

        else:
            self.adaptivity = False
            self.tau = tau

        super().assemble_problem(compute_intE = True, compute_energy = False, adaptivity = self.adaptivity)

        # warm up with dummy inputs
        _ = self.compute_integrals(self.R_u.dat.data_ro, self.R_u.dat.data_ro, self.R_u.dat.data_ro)

        if self.adaptivity:
            _ = self.compute_integral_adaptivity(self.R_u.dat.data_ro, self.R_u.dat.data_ro)
        
    @profile
    def step(self, u_old):
        '''
        Perform a step of the gradient flow.

        :param u_old (Function): the current iterate
        '''        
        # t0 = time.time()
        # Compute the Riesz representative
        self.compute_Reiz_representative(u_old)
        # t1 = time.time()
        # print(f"Time to compute Riesz representative: {t1-t0:.2f} seconds")
        # t2 = time.time()

        self.compute_integrals(u=u_old.dat.data_ro, R=self.R_u.dat.data_ro, gradE=u_old.dat.data_ro)

        self.intE_intR = self.integrals[1] / self.integrals[0]

        if self.adaptivity:
            self.compute_integral_adaptivity(u=u_old.dat.data_ro, proj_term=self.R_u.dat.data_ro)
            self.intE_intR2 = self.intE_intR**2
            self.intE_intR3 = self.intE_intR**3
            self.intE_intR4 = self.intE_intR**4


            self.alpha0 = self.integrals_adaptivity[0]
            self.alpha1 = 2 * self.intE_intR * self.integrals_adaptivity[1]
            self.alpha2 = self.intE_intR2 * self.integrals_adaptivity[2]

            self.beta0 = self.beta * 0.5 * self.integrals_adaptivity[3]
            self.beta1 = self.beta * 2 * self.intE_intR * self.integrals_adaptivity[4]
            self.beta2 = self.beta * 3 * self.intE_intR2 * self.integrals_adaptivity[5]
            self.beta3 = self.beta * 2 * self.intE_intR3 * self.integrals_adaptivity[6]
            self.beta4 = self.beta * 0.5 * self.intE_intR4 * self.integrals_adaptivity[7]

            # self.M.M.handle.mult(u_old.dat.vec_ro, self.M_u)
            self.gamma0 = self.integrals_adaptivity[8]
            self.gamma1 = 2 * self.intE_intR * self.integrals_adaptivity[9]
            self.gamma2 = self.intE_intR2 * self.integrals_adaptivity[10]
        # t3 = time.time()
        # print(f"Time to compute energy and coefficients for the golden search: {t3-t2} seconds")
        # return E_old


    
class Gradient_az_explicit(Gradient_az):
    '''
    This class implements the az gradient by discretizing the gradient flow with Forward Euler:
    
    z^{n+1} = z^n - τ z^n + τ ||z^n||_{L^2}^2/(∫ R_{a_z}(z^n) z^n dx) z^n
    '''
    def __init__(self, W, bcs, beta, v):
        super().__init__(W, bcs, beta, v, 'explicit')
       
    @profile     
    def step(self, u_old):
        '''
        Perform a step of the gradient flow.

        :param u_old (Function): the current iterate
        '''
        
        super().step(u_old)

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
        #self.golden_search(f, a= 0.01, b = 3.0)  
            # Choose tau adaptively by 1D minimization of the model energy.
            # t5 = time.time()
            self.tau = self.golden_search(f, a = 0.01, b = 3.0)
            # t6 = time.time()
            # print(f"Time to compute the golden search: {t6-t5} seconds")
            
            self.tau_history.append(self.tau)#.values()[0])
        # t7 = time.time()
        self.uh.dat.data[:] = (1.0 - self.tau) * u_old.dat.data_ro[:] + self.tau * self.intE_intR * self.R_u.dat.data_ro[:]
        # t8 = time.time()
        # print(f"Time to update the solution: {t8-t7} seconds")
        # self.M.M.handle.mult(self.uh.data.vec_ro, self.M_u)
        # return self.uh.dat.vec_ro.dot(self.M_u) ** 0.5
    
class Gradient_az_semimplicit(Gradient_az):
    '''
    This class implements the az gradient by discretizing the gradient flow with Semi-Implicit Euler:
    
    (1 + τ) z^{n+1} = z^n + τ ||z^n||_{L^2}^2/(∫ R_{a_z}(z^n) z^n dx) z^n
    '''
    def __init__(self, W, bcs, beta, v):
        super().__init__(W, bcs, beta, v, 'semimplicit')
     
    @profile       
    def step(self, u_old):
        '''
        Perform a step of the gradient flow.

        :param u_old (Function): the current iterate
        '''
        super().step(u_old)

        if self.adaptivity:
            def den(x):
                return (1/(1+x)**2 * ( self.gamma0 + x * self.gamma1 + x**2 * self.gamma2))**0.5

            def f(x):
                d2 = (1+x)**2 * den(x)**2
                d4 = (1+x)**4 * den(x)**4
                return (self.alpha0 / d2
                        + self.alpha1 / d2 * x
                        + self.alpha2 / d2 * x**2
                        + self.beta0 / d4 
                        + self.beta1 / d4 * x 
                        + self.beta2 / d4 * x**2 
                        + self.beta3 / d4 * x**3 
                        + self.beta4 / d4 * x**4)
            # Choose tau adaptively by 1D minimization of the model energy.
            self.tau = self.golden_search(f, a= 0.01, b = 1000.0)
            self.tau_history.append(self.tau)#.values()[0])
            
        self.uh.dat.data[:] = 1.0 / (1.0 + self.tau) * ( u_old.dat.data_ro[:] + self.tau * self.intE_intR * self.R_u.dat.data_ro[:])

        # self.M.M.handle.mult(self.uh.data.vec_ro, self.M_u)
        # return self.uh.dat.vec_ro.dot(self.M_u) ** 0.5
