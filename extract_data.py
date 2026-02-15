import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm


def load_and_extract_data(csv_path):
    """
    Load a CSV file and extract tau_fine, tau_coarse, and iteration counts.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with selected columns
    """
    df = pd.read_csv(csv_path)
    
    # Extract relevant columns
    columns_to_extract = ['tau_fine', 'tau_coarse', 'iterate_coarse', 'iterate_fine','N_fine','N_coarse','fine_operator','coarse_operator']
    
    # Check which columns exist in the dataframe
    available_columns = [col for col in columns_to_extract if col in df.columns]
    
    if not available_columns:
        print(f"Warning: No relevant columns found in {csv_path}")
        return None
    
    extracted_data = df[available_columns].copy()
    
    # Add source file information
    extracted_data['source_file'] = os.path.basename(csv_path)
    extracted_data['source_directory'] = os.path.basename(os.path.dirname(csv_path))
    
    return extracted_data


def process_all_csv_files(base_directory='.'):
    """
    Process all CSV files in subdirectories and combine the data.
    
    Parameters:
    -----------
    base_directory : str
        Base directory to search for CSV files
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all extracted data
    """
    all_data = []
    
    # Find all CSV files in subdirectories
    folcers_to_search = ['graphs'] # Add more folder names if needed
    base_path = Path(base_directory)
    csv_files = []
    for folder in folcers_to_search:
        csv_files.extend(base_path.glob(f'./{folder}/PF.csv'))
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        data = load_and_extract_data(str(csv_file))
        if data is not None:
            all_data.append(data)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    else:
        return None


def plot_iterations_data(combined_data):
    """
    Create plots to visualize tau and iteration data.
    
    Parameters:
    -----------
    combined_data : pd.DataFrame
        Combined DataFrame with extracted data
    """
    # Convert numeric columns to proper types and drop any non-numeric rows
    numeric_cols = ['tau_fine', 'tau_coarse', 'iterate_coarse', 'iterate_fine']
    for col in numeric_cols:
        if col in combined_data.columns:
            combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')
    
    # Drop rows with NaN values in numeric columns
    combined_data = combined_data.dropna(subset=numeric_cols)
    
    if len(combined_data) == 0:
        print("No valid numeric data to plot")
        return
    
    # Extract unique values for fine and coarse operators
    fine_operators = combined_data['fine_operator'].unique()
    coarse_operators = combined_data['coarse_operator'].unique()

    n_fine = np.sort(combined_data['N_fine'].unique())
    n_coarse = np.sort(combined_data['N_coarse'].unique())
    
    # Calculate global min and max for colorbar consistency
    # combined_data['total_iterations'] = combined_data['iterate_fine'] + combined_data['iterate_coarse']
    # vmin = combined_data['total_iterations'].min()
    # vmax = combined_data['total_iterations'].max()
    # print(f"Colorbar range: {vmin} to {vmax} iterations")

    MAX = 100
    GD ={'L2_P': 24, 'az': 24}

    for fine_op in fine_operators:
        coarse_op = fine_op

        if fine_op == 'L2_P':
            for n_fin in n_fine:
                for n_coars in n_coarse:
                    subset = combined_data[(combined_data['fine_operator'] == fine_op) & 
                                        (combined_data['coarse_operator'] == coarse_op) & 
                                        (combined_data['N_fine'] == n_fin) & 
                                        (combined_data['N_coarse'] == n_coars)]
                    
                    if len(subset) > 0:
                        fig = plt.figure(figsize=(10, 6))
                        
                        # Prepare data for interpolation
                        tau_fine_data = subset['tau_fine']
                        tau_coarse_data = subset['tau_coarse']
                        iterations_data = (subset['iterate_fine'] + subset['iterate_coarse']).values

                        # Get unique tau values and create mapping
                        unique_tau_fine = sorted(tau_fine_data.unique())
                        unique_tau_coarse = sorted(tau_coarse_data.unique())
                        
                        # Initialize Z with MAX as default
                        Z = MAX * np.ones((len(unique_tau_coarse), len(unique_tau_fine)))
                        
                        # Create mapping from tau values to array indices
                        tau_fine_to_idx = {val: idx for idx, val in enumerate(unique_tau_fine)}
                        tau_coarse_to_idx = {val: idx for idx, val in enumerate(unique_tau_coarse)}
                        
                        # Fill in the actual values
                        capped_iterations = np.minimum(iterations_data, MAX)
                        for i in range(len(capped_iterations)):
                            tau_fine_val = tau_fine_data.iloc[i]
                            tau_coarse_val = tau_coarse_data.iloc[i]
                            
                            idx_x = tau_fine_to_idx[tau_fine_val]
                            idx_y = tau_coarse_to_idx[tau_coarse_val]
                            Z[idx_y, idx_x] = capped_iterations[i]
                        
                        # Create a finer grid for smooth interpolation (in log space)
                        tau_fine_min, tau_fine_max = tau_fine_data.min(), tau_fine_data.max()
                        tau_coarse_min, tau_coarse_max = tau_coarse_data.min(), tau_coarse_data.max()
                        
                        # Create fine grid in log space for better interpolation
                        tau_fine_interp = np.logspace(np.log10(tau_fine_min), np.log10(tau_fine_max), 200)
                        tau_coarse_interp = np.logspace(np.log10(tau_coarse_min), np.log10(tau_coarse_max), 200)
                        tau_fine_grid, tau_coarse_grid = np.meshgrid(tau_fine_interp, tau_coarse_interp)
                        
                        # Interpolate data using griddata for smooth contours
                        points = np.column_stack((tau_fine_data.values, tau_coarse_data.values))
                        iterate_grid = griddata(points, capped_iterations, 
                                              (tau_fine_grid, tau_coarse_grid),
                                              method='cubic', fill_value=MAX)
                        
                        # Ensure values stay within bounds
                        iterate_grid = np.clip(iterate_grid, capped_iterations.min(), MAX)

                        # Plot interpolated contour
                        contour = plt.contourf(tau_fine_grid, tau_coarse_grid, iterate_grid, levels=200, cmap='viridis',
                                vmin=capped_iterations.min(), vmax=MAX)
                        
                        # Add contour line at threshold value (25)
                        threshold = GD[fine_op]
                        contour_line = plt.contour(tau_fine_grid, tau_coarse_grid, iterate_grid, 
                                                  levels=[threshold], colors='red', linewidths=2, linestyles='--')
                        # plt.clabel(contour_line, inline=True, fontsize=10, fmt='Threshold=%d')
                        
                        plt.xlabel('Tau Fine', fontsize=12)
                        plt.ylabel('Tau Coarse', fontsize=12)
                        plt.title(f'Total Iterations (N_fine={n_fin}, N_coarse={n_coars})', fontsize=12)
                        plt.xscale('log')
                        plt.yscale('log')
                        plt.colorbar(contour, label='Total Iterations')
                        
                        # Scatter plot overlay showing actual data points
                        # Highlight points below threshold with red edge
                        below_threshold = capped_iterations < threshold

                        scatter = plt.scatter(subset['tau_fine'], subset['tau_coarse'], 
                                    c=capped_iterations, 
                                    cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                                    vmin=capped_iterations.min(), vmax=MAX)
                        
                        # Highlight points below threshold with red circles
                        if np.any(below_threshold):
                            plt.scatter(subset['tau_fine'][below_threshold], subset['tau_coarse'][below_threshold],
                                      s=120, facecolors='none', edgecolors='red', linewidth=1, 
                                      label=f'< {threshold} iterations')
                        
                        # Add text annotations only for points below 1.1 * threshold
                        annotation_threshold = 1.5 * threshold
                        for idx in subset.index:
                            iteration_val = int(capped_iterations[subset.index.get_loc(idx)])
                            
                            # Only annotate if below annotation threshold
                            if iteration_val < annotation_threshold:
                                tau_fine_val = subset.loc[idx, 'tau_fine']
                                tau_coarse_val = subset.loc[idx, 'tau_coarse']
                                
                                plt.annotate(f'{iteration_val}', 
                                           xy=(tau_fine_val, tau_coarse_val),
                                           xytext=(5, 5), textcoords='offset points',
                                           fontsize=9, color='red', fontweight='bold',
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='red'))
                        
                        # Highlight the minimum point
                        # min_idx = subset.index[capped_iterations.argmin()]
                        # min_tau_fine = subset.loc[min_idx, 'tau_fine']
                        # min_tau_coarse = subset.loc[min_idx, 'tau_coarse']
                        # min_iterations = int(capped_iterations.min())
                        # plt.scatter([min_tau_fine], [min_tau_coarse], 
                        #            s=200, marker='*', facecolors='gold', edgecolors='black', linewidth=2,
                        #            label=f'Minimum: {min_iterations} iterations', zorder=5)
                            
                        plt.xlabel('Tau Fine', fontsize=12)
                        plt.ylabel('Tau Coarse', fontsize=12)
                        plt.title(f'Total Iterations (N_fine={n_fin}, N_coarse={n_coars})', fontsize=12)
                        plt.xscale('log')
                        plt.yscale('log')
                        plt.colorbar(scatter, label='Total Iterations')
                        
                        plt.tight_layout()

                        plt.plot(tau_fine_data, tau_fine_data, 'k', label='dt equal')
                        plt.plot(tau_fine_data, tau_fine_data * n_fin, 'g', label='dt * Nf')
                        plt.plot(tau_fine_data, tau_fine_data / n_fin, 'b', label='dt / Nf')
                        plt.plot(tau_fine_data, tau_fine_data * n_fin / 2, 'r', label='dt * Nf / 2')

                        plt.legend(loc='upper left', fontsize=10)
                        
                        #plt.show()

                        os.makedirs('./graphs_alpha_min', exist_ok=True)
                        output_file = f'./graphs_alpha_min/iterations_{fine_op}_{n_fin}_{n_coars}.png'
                        plt.savefig(output_file)
                        print(f"Plot saved to {output_file}")
                        plt.close()
                    
    print("\nAll plots generated successfully!")


def main():
    """Main function to process CSV files and save results."""
    
    # Process all CSV files in subdirectories
    combined_data = process_all_csv_files()
    
    if combined_data is not None:
        # Save to a new CSV file
        output_file = 'extracted_iterations_data.csv'
        combined_data.to_csv(output_file, index=False)
        # print(f"\nData saved to {output_file}")
        # print(f"Total rows: {len(combined_data)}")
        # print(f"\nFirst few rows:")
        # print(combined_data.head())
        # print(f"\nSummary statistics:")
        # print(combined_data.describe())
        
        # Create plots
        plot_iterations_data(combined_data)
    else:
        print("No data extracted.")


if __name__ == "__main__":
    main()
