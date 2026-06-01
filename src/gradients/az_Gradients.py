'''
This module implements the az gradient of the cost functional.
'''

import firedrake as fd
from .Gradients import Gradient


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

    def compute_Reiz_representative(self, u_old):
        '''
        For the az gradient, the Riesz representative is computed at each step solving the linear variational problem.
        '''
        rhs_R = fd.inner(u_old , self.w) * fd.dx

        problem_R = fd.LinearVariationalProblem(self.a + self.beta * abs(u_old)**2 * self.u * self.w * fd.dx,
                                                rhs_R,
                                                self.R_u,
                                                self.bcs)
        solver_R = fd.LinearVariationalSolver(problem_R)#, solver_parameters={"ksp_view": None})
        solver_R.solve()

        return self.R_u
    
    def compute_gradient(self, u_old):
        Warning("The gradient for the az gradient is the function itself, so this method is not needed.")

        return u_old
    
    def assemble_problem(self, tau = None):
        '''
        Allocate and assembles forms and minimization quantities

        :param tau (float): time step
        '''
        if tau is None:
            self.adaptivity = True
            self.tau = fd.Constant(1.0) # dummy value, the actual time step will be computed at each step.
            self.tau_history = []
        else:
            self.tau = fd.Constant(tau)

        # Built the Reisz representative and the linear part of the variational problem
        self.R_u = fd.Function(self.W)

        # Reusable coefficients for adaptive-model assemblies (updated each step).
        self.inv_intR = fd.Constant(1.0)
        self.inv_intR2 = fd.Constant(1.0)
        self.inv_intR3 = fd.Constant(1.0)
        self.inv_intR4 = fd.Constant(1.0)

        self.a = 0.5 * fd.inner(fd.grad(self.u), fd.grad(self.w)) * fd.dx \
                + self.v * self.u * self.w * fd.dx
        
    def step(self, u_old):
        '''
        Perform a step of the gradient flow.

        :param u_old (Function): the current iterate
        '''
        if fd.norm(u_old, 'L2')-1 >1e-12:
            raise ValueError(f"The previous solution is {fd.norm(u_old, 'L2')}, cannot proceed with the minimization.")
        
        # Compute the Riesz representative
        self.compute_Reiz_representative(u_old)

        # Update as a convex combination of u_old and normalized Riesz direction.
        # intE = fd.assemble(u_old * u_old * fd.dx) # should be 1 in exact aritmetic because it's simply the norm of the previous uh
        self.intR = fd.assemble(self.R_u * u_old * fd.dx)


        self.alpha0 = fd.assemble(0.5 * fd.inner(fd.grad(u_old), fd.grad(u_old)) * fd.dx \
                            + self.v * u_old * u_old * fd.dx)
        self.beta0 = fd.assemble(self.beta * 0.5 * (u_old)**4 * fd.dx)
        self.gamma0 = fd.assemble(u_old**2 * fd.dx)

        E_old = 0.5 * (self.alpha0/self.gamma0**2 + self.beta0/self.gamma0**4)

        if self.adaptivity:
            self.inv_intR.assign(1.0 / self.intR)
            self.inv_intR2.assign(1.0 / self.intR**2)
            self.inv_intR3.assign(1.0 / self.intR**3)
            self.inv_intR4.assign(1.0 / self.intR**4)

            self.alpha1 = fd.assemble( 2 * self.inv_intR * 0.5 * fd.inner(fd.grad(self.R_u), fd.grad(u_old)) * fd.dx \
                                + 2 * self.inv_intR * self.v * self.R_u * u_old * fd.dx)
            self.alpha2 = fd.assemble( self.inv_intR2 * 0.5 * fd.inner(fd.grad(self.R_u), fd.grad(self.R_u)) * fd.dx \
                                + self.inv_intR2 * self.v * self.R_u * self.R_u * fd.dx)
            self.beta1 = fd.assemble(self.beta * 2 * (u_old)**3 * self.R_u * self.inv_intR * fd.dx)
            self.beta2 = fd.assemble(self.beta * 3 * (u_old)**2 * self.R_u**2 * self.inv_intR2 * fd.dx)
            self.beta3 = fd.assemble(self.beta * 2 * u_old * self.R_u**3 * self.inv_intR3 * fd.dx)
            self.beta4 = fd.assemble(self.beta * 0.5 * self.R_u**4 * self.inv_intR4 * fd.dx)
            self.gamma1 = fd.assemble(2 * u_old * self.R_u * self.inv_intR * fd.dx)
            self.gamma2 = fd.assemble(self.R_u**2 * self.inv_intR2 * fd.dx)
            
        return E_old


    
class Gradient_az_explicit(Gradient_az):
    '''
    This class implements the az gradient by discretizing the gradient flow with Forward Euler:
    
    z^{n+1} = z^n - τ z^n + τ ||z^n||_{L^2}^2/(∫ R_{a_z}(z^n) z^n dx) z^n
    '''
    def __init__(self, W, bcs, beta, v):
        super().__init__(W, bcs, beta, v, 'explicit')
            
    def step(self, u_old):
        '''
        Perform a step of the gradient flow.

        :param u_old (Function): the current iterate
        '''
        
        E_old = super().step(u_old)

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
            self.tau = fd.Constant(self.golden_search(f, a= 0.01, b = 3.0))
            
            self.tau_history.append(self.tau)

        self.uh.assign((1 - self.tau) * u_old + self.tau * 1/self.intR * self.R_u)
        return E_old
    
class Gradient_az_semimplicit(Gradient_az):
    '''
    This class implements the az gradient by discretizing the gradient flow with Semi-Implicit Euler:
    
    (1 + τ) z^{n+1} = z^n + τ ||z^n||_{L^2}^2/(∫ R_{a_z}(z^n) z^n dx) z^n
    '''
    def __init__(self, W, bcs, beta, v):
        super().__init__(W, bcs, beta, v, 'semimplicit')
            
    def step(self, u_old):
        '''
        Perform a step of the gradient flow.

        :param u_old (Function): the current iterate
        '''
        E_old = super().step(u_old)

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
            self.tau = fd.Constant(self.golden_search(f, a= 0.01, b = 1000.0))
            self.tau_history.append(self.tau)
            
        self.uh.assign(1 / (1+ self.tau) * ( u_old + self.tau * 1/self.intR * self.R_u))
        return E_old
