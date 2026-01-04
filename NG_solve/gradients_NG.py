import ngsolve as ng
import matplotlib.pyplot as plt
import time
import abc
import csv

class GradientsNG(abc.ABC):
    def __init__(self, beta, potential, hmax, mesh, order, dirichlet_bcs):

        self.h = hmax
        self.mesh = mesh
        self.fes = ng.H1(self.mesh, order=order, dirichlet=dirichlet_bcs)

        self.u = self.fes.TrialFunction()
        self.v = self.fes.TestFunction()

        self.beta = beta
        if potential == 'Harmonic':
            self.potential = 0.5 * (ng.x**2 + ng.y**2)
        elif potential == 'Harmonic and optical lattice':
            self.potential = 0.5 * (ng.x**2 + ng.y**2) + 20 + 20 * ng.sin(2 * ng.pi * ng.x) * ng.sin(2 * ng.pi * ng.y)
        else:
            raise ValueError("Potential not recognized")
        
        self.uh = ng.GridFunction(self.fes, name="solution")

        self.E = 0.0

        self.history_E = []

    def energy(self, u = None):
        '''
        Compute the Gross-Pitaevskii energy of the given function or, if not given, of the solution
        
        :param u (GridFunction): function to calculate the energy
        '''
        if u is not None:
            E_kin = 0.5 * ng.Integrate(ng.grad(u) * ng.grad(u), self.mesh)
            E_pot = ng.Integrate(self.potential * u**2, self.mesh)
            E_int = 0.5 * self.beta * ng.Integrate(u**4, self.mesh)

            return (E_kin + E_pot + E_int)/2, E_kin, E_pot, E_int

        E_kin = 0.5 * ng.Integrate(ng.grad(self.uh) * ng.grad(self.uh), self.mesh)
        E_pot = ng.Integrate(self.potential * self.uh**2, self.mesh)
        E_int = 0.5 * self.beta * ng.Integrate(self.uh**4, self.mesh)

        self.E = (E_kin + E_pot + E_int)/2
        return self.E, E_kin, E_pot, E_int
    
    def compute_lambda(self):
        '''
        Compute the lambda associated to the energy of self.uh
        '''
        self.energy()

        self.lam = 2 * self.E + self.beta / 2 * ng.Integrate(self.uh**4, self.mesh)

    def normalize(self):
        '''
        L2-normalization of the solution
        '''
        norm_sq = ng.Integrate(self.uh**2, self.mesh)
        self.uh.vec.data *= 1.0 / ng.sqrt(norm_sq)
        return ng.Integrate(self.uh**2, self.mesh)
    
    def assemble_problem(self, initial_guess, tau, E_ref):

        self.uh_old = ng.GridFunction(self.fes, name="previous solution")

        if initial_guess == 'normalized gaussian':
            self.uh_old.Set(1.0 / ng.sqrt(ng.pi) * ng.exp(-(ng.x**2 + ng.y**2) / 2.0))
        elif initial_guess == 'Thomas-Fermi density':
            mu_tf = ng.sqrt(self.beta/ng.pi)
            self.uh_old.Set(ng.IfPos(mu_tf - self.potential, ng.sqrt((mu_tf - self.potential)/self.beta), 0.0))
        else:
            ValueError('Initial guess not known')
        
        norm = ng.Integrate(self.uh_old**2, self.mesh)
        self.uh_old.vec.data *= 1.0 / ng.sqrt(norm)

        self.tau = tau

        self.E_ref = E_ref

        self.E = 0.0
        self.history_E = []

    @abc.abstractmethod
    def step(self):
        '''
        Step of the minimization, based on the choosen class wil be overridden
        '''
        pass

    def minimize(self, MaxIter, toll, verbose=True):
        '''
        Minimize the functional applying the step method iteratively
        
        :param MaxIter (int): maximum number of iteration allowed
        :param toll (float): tollerance of the stopping criteria
        :param verbose (bool): specify if print results or not (default True)  
        '''
        converged = False
        self.MaxIter = MaxIter

        t_start = time.time()

        for i in range(MaxIter):
            self.step()

            norm = self.normalize()
            
            E_total, E_kin, E_pot, E_int = self.energy()

            self.history_E.append(E_total)

            error = abs(E_total - self.E_ref) / self.E_ref

            self.uh_old.vec.data = self.uh.vec

            if verbose:
                print(f"\rIter {i:3d}: E = {E_total:12.8f}, error = {error:10.6e} norm = {norm:.6f}", end="", flush=True)
                
            if error < toll:
                converged = True
                break

        time_tot = time.time() - t_start

        self.compute_lambda()

        res = dict(converged = converged,
                   energy = self.E,
                   lam = self.lam,
                   iterate = i+1,
                   error = error,
                   norm = norm,
                   time_tot = time_tot,
                   mean_time = time_tot/(i+1))
        
        if verbose:
            print('\r', end="", flush=True)

        return res

    def plot_history(self, method_name, show = False, save = False):
        '''
        Plot the convergence history of the minimization

        :param method_name (string): name of the gradient method used'''
        fig, ax = plt.subplots(2,1, figsize=(5,10))
        fig.suptitle(f'{method_name}_gradient with h={self.h}, beta={self.beta }, tau={self.tau }')
        ax[0].semilogy(range(1,len(self.history_E)+1), [abs(E - self.E_ref)/self.E_ref for E in self.history_E], marker='o')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Relative Error on Energy')
        ax[0].set_title('Convergence History')
        ax[0].grid(True)

        ax[1].semilogy(range(1,len(self.history_E)+1), self.history_E, marker='o')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Energy')
        ax[1].set_title('Energy History')
        ax[1].grid(True)
        if show:
            plt.show()
        if save:
            fig.savefig("./images/plot_b"+str(self.beta )+"_N"+str(int(1/self.h))+"_tau"+str(self.tau )+"_it"+str(self.MaxIter)+".png")

    def save_data(self, filename, opt_name, res):
        '''
        Save minimization result in a csv file

        :param filename (string): path to the csv file 
        :param opt_name (string): specify which gradient has been used
        :param res (dict): dictionary taht contains all the important data
        '''
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([opt_name, self.h, self.beta , self.tau , res["energy"], res["lam"], res["iterate"], res["error"], res["time_tot"], res["mean_time"]])

class Gradient_L2(GradientsNG):
    def assemble_problem(self, initial_guess, tau, E_ref):
        return super().assemble_problem(initial_guess, tau, E_ref)
    
    def step(self):
        density = self.uh_old**2
    
        # Define bilinear form (left-hand side)
        # ∫[u · v + dt * ∇u·∇v + dt * V*u*v + dt * g|ψ|²*u*v] dx
        a = ng.BilinearForm(self.fes)
        a += self.u * self.v * ng.dx
        a += self.tau * 0.5 * ng.grad(self.u) * ng.grad(self.v) * ng.dx
        a += self.tau * self.potential * self.u * self.v * ng.dx
        a += self.tau * self.beta * density * self.u * self.v * ng.dx
        a.Assemble()
        
        # Define linear form (right-hand side)
        # ∫ ψ · v dx
        f = ng.LinearForm(self.fes)
        f += self.uh_old * self.v * ng.dx
        f.Assemble()
        
        # Solve linear system
        self.uh.vec.data = a.mat.Inverse(self.fes.FreeDofs()) * f.vec

class Gradient_az(GradientsNG):
    def assemble_problem(self, initial_guess, tau, E_ref):
        super().assemble_problem(initial_guess, tau, E_ref)

        self.Ru = ng.GridFunction(self.fes, name="Riesz projection")

    def step(self):
        density = self.uh_old**2
    
        # Define bilinear form (left-hand side)
        # ∫[u · v + dt * ∇u·∇v + dt * V*u*v + dt * g|ψ|²*u*v] dx
        a = ng.BilinearForm(self.fes)
        a += 0.5 * ng.grad(self.u) * ng.grad(self.v) * ng.dx
        a += self.potential * self.u * self.v * ng.dx
        a += self.beta * density * self.u * self.v * ng.dx
        a.Assemble()
        
        # Define linear form (right-hand side)
        # ∫ ψ · v dx
        f = ng.LinearForm(self.fes)
        f += self.uh_old * self.v * ng.dx
        f.Assemble()

        # compute Riesz projection
        self.Ru.vec.data = a.mat.Inverse(self.fes.FreeDofs()) * f.vec

        int_R = ng.Integrate(self.Ru * self.uh_old,self.mesh)
        int_E = ng.Integrate(self.uh_old ** 2, self.mesh)

        self.uh.vec.data = (1 - self.tau) * self.uh_old.vec.data + self.tau * int_E/int_R * self.Ru.vec.data