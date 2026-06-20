import abc
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
import gradients #import L2_Gradients, H1_Gradients, a0_Gradients, az_Gradients

class Optimizer(abc.ABC):
    '''
    Abstract class for the optimization algorithm. 
    This class defines the interface for the optimization algorithm and implements some common methods.
    '''

    @abc.abstractmethod
    def __init__(self, W, beta, v, name_optimizer):
        '''
        Constructor for the optimizer class. 

        :param W (fd space): The function space for the problem.
        :param beta (float): The beta parameter for the problem.
        :param v (fd function): The potential function for the problem.
        :param name_optimizer (str): A string identifier for the name of the optimizer used. This is used for logging and plotting purposes.

        NOTE: it is declared as an abstract method because every subclasses must implement the part to attach the solver(s).
        '''

        self.name_optimizer = name_optimizer

        # save problem parameters
        self.W = W
        self.beta = beta
        self.v = v

        # parameters of the solution
        self.lam = 0.
        self.E = 0.
        self.uh = fd.Function(W)
        self.u_old = fd.Function(W)
        self.history_E = []

        
    def get_solver(self, W, bcs, beta, v, grad_type_name, type_discr):
        '''
        Factory method to get the solver for the gradient flow.

        :param grad_type_name (str): A string identifier for the type of gradient method used.
        :param type_discr (str): A string identifier for the type of discretization used.
        '''
        if grad_type_name in ['L2', 'l2', 'L_2']:
            if type_discr in ['explicit', 'fully_explicit', 'E', 'e']:
                return gradients.L2_Gradients.Gradient_L2_explicit(W, bcs, beta, v)
            elif type_discr in ['semimplicit', 'semi-implicit', 'S', 's']:
                return gradients.L2_Gradients.Gradient_L2_semimplicit(W, bcs, beta, v)
            elif type_discr in ['no_projection', 'no-projection', 'N', 'n']:
                UserWarning("you are using the L2 gradient without the projection term. This will not work well for ParaFlowS.")
                return gradients.L2_Gradients.Gradient_L2_no_projection(W, bcs, beta, v)
            else:
                raise ValueError(f"Unknown discretization type {type_discr} for L2 gradient.")
            
        elif grad_type_name in ['H1', 'h1', 'H_1']:
            if type_discr in ['explicit', 'fully_explicit', 'E', 'e']:
                return gradients.H1_Gradients.Gradient_H1_explicit(W,  bcs, beta, v)
            else:
                raise ValueError(f"Unknown discretization type {type_discr} for H1 gradient.")
            
        elif grad_type_name in ['a0', 'A0', 'a_0']:
            if type_discr in ['explicit', 'fully_explicit', 'E', 'e']:
                return gradients.a0_Gradients.Gradient_a0_explicit(W,  bcs, beta, v)
            else:
                raise ValueError(f"Unknown discretization type {type_discr} for a0 gradient.")
            
        elif grad_type_name in ['az', 'AZ', 'a_z']:
            if type_discr in ['explicit', 'fully_explicit', 'E', 'e']:
                return gradients.az_Gradients.Gradient_az_explicit(W, bcs, beta, v)
            elif type_discr in ['semimplicit', 'semi-implicit', 'S', 's']:
                return gradients.az_Gradients.Gradient_az_semimplicit(W, bcs, beta, v)
            else:
                raise ValueError(f"Unknown discretization type {type_discr} for az gradient.")
            
        else:
            raise ValueError(f"Unknown gradient type {grad_type_name} or discretization type {type_discr}.")
    
    def energy(self, uh = None):
        '''
        Computes the energy associated to the given solution
        
        :param uh (fd.Function): solution where compute the energy, if not provided computes the solution in the point self.uh
        '''
        if uh is not None:
            return fd.assemble(0.5 *( 0.5 * fd.dot(fd.grad(uh), fd.grad(uh)) + self.v * uh**2 + self.beta/2 * abs(uh) **4) * fd.dx)

        else:
            return fd.assemble(( 0.25 * fd.dot(fd.grad(self.uh), fd.grad(self.uh)) + 0.5 * self.v * self.uh**2 + 0.25 * self.beta * abs(self.uh) **4 )* fd.dx)
        
        
    def compute_lambda(self):
        '''
        Compute the lambda associated to the energy of self.uh
        '''

        self.lam = 2 * self.E + float(self.beta) /2 * fd.norm(self.uh,'L4')**4
        
        return self.lam

        
    def compile(self, u0, E_ref):
        '''
        Compile the optimization algorithm by initializing the gradient solver and computing the initial energy.

        :param u0 (fd function): The initial guess for the solution.
        :param E_ref (float): The reference energy to compute the relative error.
        '''
        # Normalize the initial guess
        norm = fd.norm(u0, 'L2')
        self.u_old.interpolate(u0 / norm)

        # Save the reference energy for the stopping criterion
        self.E_ref = E_ref

        # Reset the history of the energy and the lambda values
        self.E = 0.
        self.lam = 0.
        self.history_E = []
    
    def golden_search(self, func, a = 0.01, b = 1.0, tol = 1e-5):
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

        # x = np.linspace(a - 0.3, b + 0.5, 100)

        while b - a > tol:
            c = b - (b - a) * invphi
            d = a + (b - a) * invphi
            if func(c) < func(d):
                b = d
            else:  # func(c) > func(d) to find the maximum
                a = c

        # fig, ax = plt.subplots()
        # ax.plot(x, func(x))
        # plt.plot((b + a) / 2, func((b + a) / 2), 'ro', label='Minimum')
        # plt.legend()
        # plt.show()

        # print()
        # print(f'new energy {func((a+b)/2)/2} and old energy {func(0)/2}')

        return (b + a) / 2

    @abc.abstractmethod
    def minimize(self, max_iter, toll_E, verbose):
        '''
        Abstract method to perform the optimization.

        :param max_iter (int): The maximum number of iterations.
        :param toll_E (float): The tolerance for the energy relative error to stop the optimization.
        :param verbose (bool): Whether to print the progress of the optimization.
        '''
        pass

    def plot_history(self):
        '''
        Plot the history of the energy during the optimization.
        '''
        pass

    def save_data(self, filename, res):
        '''
        Save minimization data to a csv file.
        :param filename (str): The name of the file to save the data.
        :param res (dict): The dictionary of results to save.
        '''
        pass