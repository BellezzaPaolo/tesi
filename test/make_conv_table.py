import pandas as pd
import sys

def make_latex_table(csv_file, output_file=None):
    """
    Generate LaTeX table from CSV file with convergence data.
    Creates 2 tables (one per optimizer) with rows for different beta values.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    output_file : str, optional
        Path to save the LaTeX output. If None, prints to stdout.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Map optimizer names to more readable labels
    optimizer_labels = {
        'az': '$a_u$-Sobolev Gradient flow',
        'L2_P': '$L^2$ Gradient flow'
    }
    
    latex_output = []
    
    # Group by optimizer_name
    for optimizer_name in df['optimizer_name'].unique():
        optimizer_df = df[df['optimizer_name'] == optimizer_name]
        
        # Get all tau values (assuming they're consistent across beta values)
        all_tau = sorted(optimizer_df['tau'].unique())
        all_beta = sorted(optimizer_df['beta'].unique())
        
        latex_output.append("\\begin{table}[ht]")
        latex_output.append("\\centering")
        
        # Caption
        optimizer_label = optimizer_labels.get(optimizer_name, optimizer_name)
        latex_output.append(f"\\caption{{{optimizer_label}}}")
        latex_output.append(f"\\label{{tab:{optimizer_name}}}")
        
        # Create tabular with dynamic number of columns
        n_cols = len(all_tau)
        col_spec = "|c||" + "c|" * n_cols
        latex_output.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_output.append("\\hline")
        
        # Header row with tau values
        tau_row = "$\\tau$ & " + " & ".join([f"${tau}$" for tau in all_tau]) + "\\\\"
        latex_output.append(tau_row)
        latex_output.append("\\hline")
        
        # One row for each beta value
        for beta in all_beta:
            beta_df = optimizer_df[optimizer_df['beta'] == beta].sort_values('tau')
            
            # Create a dictionary mapping tau to iterate count
            tau_to_iter = dict(zip(beta_df['tau'], beta_df['iterate']))
            
            # Build the row with N values for each tau
            n_values = []
            for tau in all_tau:
                if tau in tau_to_iter:
                    n_values.append(f"${int(tau_to_iter[tau])}$")
                else:
                    n_values.append("$-$")  # Missing data
            
            n_row = f"$\\beta = {int(beta)}$ & " + " & ".join(n_values) + "\\\\"
            latex_output.append(n_row)
            latex_output.append("\\hline")
        
        latex_output.append("\\end{tabular}")
        latex_output.append("\\end{table}")
        latex_output.append("")
        latex_output.append("\\vspace{0.5cm}")
        latex_output.append("")
    
    # Join all lines
    result = "\n".join(latex_output)
    
    # Output
    if output_file:
        with open(output_file, 'w') as f:
            f.write(result)
        print(f"LaTeX table saved to {output_file}")
    else:
        print(result)
    
    return result


if __name__ == "__main__":
    # Hardcoded file paths
    csv_file = "case_test3/GD.csv"
    output_file = "case_test3/GD_conv_table.tex"
    
    make_latex_table(csv_file, output_file)
