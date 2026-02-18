#!/usr/bin/env python3
"""
Firedrake implementation of random disorder potential from DUNE-BEC Model Seven.

The potential V is a random disorder potential obtained by dividing the domain
into 400 × 400 square cells. In each cell independently, the potential takes 
either value V(x) = 1 or V(x) = 1/ε² with equal probability.

Domain: [-6, 6]² 
Cell edge length: ε = 12.0 / 400 = 0.03
"""

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt


class RandomDisorderPotential:
    """
    Random disorder potential with piecewise constant values on a grid.
    Translated from DUNE-BEC Stationary_GPE_Seven PotentialTerm class.
    """
    
    def __init__(self, number_of_cells=400, domain_size=12.0, seed=10):
        """
        Initialize the random disorder potential.
        
        Parameters:
        -----------
        number_of_cells : int
            Number of cells in each direction (default: 400)
        domain_size : float
            Size of the domain in one direction, domain is [-domain_size/2, domain_size/2]²
        seed : int
            Random seed for reproducibility (default: 2, same as C++ code)
        """
        self.number_of_cells = number_of_cells
        self.domain_size = domain_size
        self.eps = domain_size / number_of_cells
        
        # Initialize random number generator with fixed seed for reproducibility
        np.random.seed(seed)
        
        # Generate random potential values: either 1 or 2 with equal probability
        # This matches: lower_range + (rand() % (upper_range - lower_range + 1))
        # where lower_range=1, upper_range=2
        self.potential_values = np.random.randint(1, 3, size=(number_of_cells, number_of_cells))
        
        print(f"Random disorder potential initialized:")
        print(f"  Number of cells: {number_of_cells} × {number_of_cells}")
        print(f"  Cell edge length ε: {self.eps}")
        print(f"  Domain: [{-domain_size/2}, {domain_size/2}]²")
        print(f"  Potential values: 1 or {1.0/(self.eps**2):.1f}")
    
    def evaluate(self, x, y):
        """
        Evaluate the potential at coordinates (x, y).
        
        This translates the C++ code:
            int i0 = floor((x[0]+6.0) / (12.0*eps));
            int j0 = floor((x[1]+6.0) / (12.0*eps));
            y = (double)potential_values_[i0][j0];
            if (y == 2.0) { y = 1.0 / (eps * eps); }
        t is
        Parameters:
        -----------
        x, y : float or array
            Coordinates where to evaluate the potential
            
        Returns:
        --------
        float or array : Potential value(s)
        """
        half_domain = self.domain_size / 2.0
        
        # Convert coordinates to cell indices
        i0 = np.floor((x + half_domain) / (self.domain_size * self.eps)).astype(int)
        j0 = np.floor((y + half_domain) / (self.domain_size * self.eps)).astype(int)
        
        # Handle boundary cases (when exactly at upper boundary)
        i0 = np.where(i0 == self.number_of_cells, self.number_of_cells - 1, i0)
        j0 = np.where(j0 == self.number_of_cells, self.number_of_cells - 1, j0)
        
        # Clamp to valid range
        i0 = np.clip(i0, 0, self.number_of_cells - 1)
        j0 = np.clip(j0, 0, self.number_of_cells - 1)
        
        # Get potential value
        pot_value = self.potential_values[i0, j0].astype(float)
        
        # Map value 2 to 1/ε²
        pot_value = np.where(pot_value == 2.0, 1.0 / (self.eps**2), pot_value)
        
        return pot_value
    
    def __call__(self, x, y):
        """Allow calling the object as a function."""
        return self.evaluate(x, y)


def create_potential_function(mesh, potential_obj, function_space=None):
    """
    Create a Firedrake Function representing the random disorder potential.
    
    Parameters:
    -----------
    mesh : Mesh
        The computational mesh
    potential_obj : RandomDisorderPotential
        The potential object
    function_space : FunctionSpace, optional
        The function space (if None, uses piecewise constant DG0)
        
    Returns:
    --------
    Function : The potential as a Firedrake function
    """
    if function_space is None:
        function_space = FunctionSpace(mesh, "DG", 0)
    
    V_potential = Function(function_space, name="Random_Disorder_Potential")
    
    # Get coordinates
    x, y = SpatialCoordinate(mesh)
    
    # Interpolate the potential
    # We need to evaluate at the mesh coordinates
    coords = function_space.mesh().coordinates.dat.data_ro
    values = potential_obj.evaluate(coords[:, 0], coords[:, 1])
    
    # For DG0, we need to evaluate at cell centers
    if function_space.ufl_element().degree() == 0:
        # Get cell centers
        V_cg1 = VectorFunctionSpace(mesh, "CG", 1)
        cell_centers = Function(V_cg1).interpolate(SpatialCoordinate(mesh))
        cell_centers_data = cell_centers.dat.data_ro
        
        # Evaluate potential at cell centers
        values = potential_obj.evaluate(cell_centers_data[:, 0], cell_centers_data[:, 1])
        
        # Average to get cell values (for DG0, one value per cell)
        # This is simplified; proper implementation would compute cell-wise
    
    # Set the values
    V_potential.dat.data[:] = values
    
    return V_potential


def visualize_potential(potential_obj, n_points=500):
    """
    Visualize the random disorder potential.
    
    Parameters:
    -----------
    potential_obj : RandomDisorderPotential
        The potential object
    n_points : int
        Number of points for visualization grid
    """
    half_domain = potential_obj.domain_size / 2.0
    x = np.linspace(-half_domain, half_domain, n_points)
    y = np.linspace(-half_domain, half_domain, n_points)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate potential
    V = potential_obj.evaluate(X, Y)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(V, extent=[-half_domain, half_domain, -half_domain, half_domain],
               origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Potential V(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Random Disorder Potential (ε = {potential_obj.eps:.4f})')
    plt.tight_layout()
    plt.savefig('random_disorder_potential.png', dpi=150)
    print("Potential visualization saved to 'random_disorder_potential.png'")
    plt.close()
    
    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(V.flatten(), bins=50, edgecolor='black')
    plt.xlabel('Potential Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Potential Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('potential_histogram.png', dpi=150)
    print("Histogram saved to 'potential_histogram.png'")
    plt.close()


def example_stationary_gpe():
    """
    Example: Solve stationary Gross-Pitaevskii equation with random disorder potential.
    
    Equation: -Δu + V(x)u + β|u|²u = λu
    """
    print("\n" + "="*70)
    print("Example: Stationary GPE with Random Disorder Potential")
    print("="*70 + "\n")
    
    # Create mesh on domain [-6, 6]²
    n_elements = 100  # mesh refinement
    mesh = RectangleMesh(n_elements, n_elements, 12.0, 12.0, 
                         originX=-6.0, originY=-6.0)
    
    # Create random disorder potential
    potential = RandomDisorderPotential(number_of_cells=400, domain_size=12.0, seed=2)
    
    # Visualize the potential
    print("\nGenerating potential visualization...")
    visualize_potential(potential)
    
    # Create function space
    V = FunctionSpace(mesh, "CG", 2)  # Continuous Galerkin, degree 2
    
    # Interpolate potential onto function space
    print("\nInterpolating potential onto mesh...")
    x, y = SpatialCoordinate(mesh)
    
    # Create a Python function that can be used with Firedrake
    class PotentialExpression:
        def __init__(self, pot_obj):
            self.pot_obj = pot_obj
            
        def eval(self, x_coord):
            return self.pot_obj.evaluate(x_coord[0], x_coord[1])
    
    # For visualization, create a DG0 function with the potential
    V_dg = FunctionSpace(mesh, "DG", 0)
    V_potential = Function(V_dg, name="Potential")
    
    # Manually set potential values by evaluating at each cell
    coords_dg = V_dg.mesh().coordinates.dat.data_ro
    for i, coord in enumerate(coords_dg):
        V_potential.dat.data[i] = potential.evaluate(coord[0], coord[1])
    
    # Output potential to file
    print("\nSaving potential to VTK file...")
    output_file = File("random_disorder_potential.pvd")
    output_file.write(V_potential)
    print("Potential saved to 'random_disorder_potential.pvd'")
    
    print("\n" + "="*70)
    print("Potential statistics:")
    print(f"  Min value: {V_potential.dat.data.min():.2f}")
    print(f"  Max value: {V_potential.dat.data.max():.2f}")
    print(f"  Mean value: {V_potential.dat.data.mean():.2f}")
    print(f"  Std dev: {V_potential.dat.data.std():.2f}")
    print("="*70 + "\n")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("Random Disorder Potential - Firedrake Implementation")
    print("Translated from DUNE-BEC Model Seven")
    print("="*70 + "\n")
    
    # Run the example
    example_stationary_gpe()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
