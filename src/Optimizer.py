import abc
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
from src.gradients import L2_Gradients, a0_Gradients, H1_Gradients, az_Gradients

class Optimizer(abc.ABC):
    '''
    Abstract class for the optimization algorithm. 
    This class defines the interface for the optimization algorithm and implements some common methods.
    '''

    @abc.abstractmethod
    def __init__(self, W, grad_type_name, type_discr, name_optimizer):
        '''
        Constructor for the optimizer class. 

        :param W (fd space): The function space for the problem.
        :param grad_type_name (str): A string identifier for the type of gradient method used.
        :param type_discr (str): A string identifier for the type of discretization used.

        NOTE: it is declared as an abstract method because every subclasses must implement the part to attach the solver(s).
        '''

        self.name_optimizer = name_optimizer
        self.grad_type_name = grad_type_name
        self.type_discr = type_discr

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
                return L2_Gradients.Gradient_L2_explicit(W, bcs, beta, v)
            elif type_discr in ['semimplicit', 'semi-implicit', 'S', 's']:
                return L2_Gradients.Gradient_L2_semimplicit(W, bcs, beta, v)
            elif type_discr in ['no_projection', 'no-projection', 'N', 'n']:
                UserWarning("you are using the L2 gradient without the projection term. This will not work well for ParaFlowS.")
                return L2_Gradients.Gradient_L2_no_projection(W, bcs, beta, v)
            else:
                raise ValueError(f"Unknown discretization type {type_discr} for L2 gradient.")
            
        elif grad_type_name in ['H1', 'h1', 'H_1']:
            if type_discr in ['explicit', 'fully_explicit', 'E', 'e']:
                return H1_Gradients.Gradient_H1_explicit(W,  bcs, beta, v)
            else:
                raise ValueError(f"Unknown discretization type {type_discr} for H1 gradient.")
            
        elif grad_type_name in ['a0', 'A0', 'a_0']:
            if type_discr in ['explicit', 'fully_explicit', 'E', 'e']:
                return a0_Gradients.Gradient_a0(W,  bcs, beta, v)
            else:
                raise ValueError(f"Unknown discretization type {type_discr} for a0 gradient.")
            
        elif grad_type_name in ['az', 'AZ', 'a_z']:
            if type_discr in ['explicit', 'fully_explicit', 'E', 'e']:
                return az_Gradients.Gradient_az_explicit(W, bcs, beta, v)
            elif type_discr in ['semimplicit', 'semi-implicit', 'S', 's']:
                return az_Gradients.Gradient_az_semimplicit(W, bcs, beta, v)
            else:
                raise ValueError(f"Unknown discretization type {type_discr} for az gradient.")
            
        else:
            raise ValueError(f"Unknown gradient type {grad_type_name} or discretization type {type_discr}.")
        
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