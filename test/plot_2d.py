import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
case_folder = Path(__file__).parent / 'case_test2'

# Read CSV files
print("Reading CSV files...")
df_gd = pd.read_csv(case_folder / 'GD.csv')
df_pf_dtxNf = pd.read_csv(case_folder / 'PF_dtxNf.csv')
df_pf_dt = pd.read_csv(case_folder / 'PF_dt.csv')
df_pf_dtxNf_2 = pd.read_csv(case_folder / 'PF_dtxNf_2.csv')
df_pf_dtxNf_2x3 = pd.read_csv(case_folder / 'PF_dtxNf_2x3.csv')
df_pf_dt_Nf = pd.read_csv(case_folder / 'PF_dt_Nf.csv')

N_fine = df_pf_dtxNf['N_fine'].unique()
N_coarse = df_pf_dtxNf['N_coarse'].unique()
opt = df_pf_dtxNf['fine_operator'].unique()

# Plot GD data
for name in opt:
    for nf in N_fine:
        for nc in N_coarse:
            print(f"Plotting for GD: {name}, PF: Nf={nf}, Nc={nc}")
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))

            subset = df_gd[df_gd['optimizer_name'] == name].sort_values('tau')
            it = subset['iterate'].values.copy()  # Convert to numpy array
            it[-1] = 560
            ax.plot(subset['tau'].values, it,
                marker='o', linewidth=4, markersize=8,
                label=f'GD - {name}')

            subset_dtxNf = df_pf_dtxNf[(df_pf_dtxNf['N_fine'] == nf) & (df_pf_dtxNf['N_coarse'] == nc)].sort_values('tau_fine')
            subset_dt = df_pf_dt[(df_pf_dt['N_fine'] == nf) & (df_pf_dt['N_coarse'] == nc)].sort_values('tau_fine')
            subset_dtxNf_2 = df_pf_dtxNf_2[(df_pf_dtxNf_2['N_fine'] == nf) & (df_pf_dtxNf_2['N_coarse'] == nc)].sort_values('tau_fine')
            subset_dtxNf_2x3 = df_pf_dtxNf_2x3[(df_pf_dtxNf_2x3['N_fine'] == nf) & (df_pf_dtxNf_2x3['N_coarse'] == nc)].sort_values('tau_fine')
            subset_dt_Nf = df_pf_dt_Nf[(df_pf_dt_Nf['N_fine'] == nf) & (df_pf_dt_Nf['N_coarse'] == nc)].sort_values('tau_fine')
            
            it = (subset_dtxNf['iterate_fine'] + subset_dtxNf['iterate_coarse']).values.copy()
            it[-1] = 560
            ax.plot(subset_dtxNf['tau_fine'].values, it,
                marker='s', linewidth=2, markersize=6, linestyle='--',
                label='PF - tau_f *Nf')

            it = (subset_dt['iterate_fine'] + subset_dt['iterate_coarse']).values.copy()
            it[-1] = 560
            ax.plot(subset_dt['tau_fine'].values, it,
                marker='^', linewidth=2, markersize=6,  linestyle='--',
                label='PF - tau_f')

            it = (subset_dtxNf_2['iterate_fine'] + subset_dtxNf_2['iterate_coarse']).values.copy()
            it[-1] = 560
            ax.plot(subset_dtxNf_2['tau_fine'].values, it,
                marker='d', linewidth=2, markersize=6, linestyle='--',
                label='PF - tau_f *Nf /2')


            it = (subset_dtxNf_2x3['iterate_fine'] + subset_dtxNf_2x3['iterate_coarse']).values.copy()
            it[-1] = 560
            ax.plot(subset_dtxNf_2x3['tau_fine'].values, it,
                marker='v', linewidth=2, markersize=6, linestyle='--',
                label='PF - tau_f *Nf * 1.5')

            it = (subset_dt_Nf['iterate_fine'] + subset_dt_Nf['iterate_coarse']).values.copy()
            it[-1] = 560
            ax.plot(subset_dt_Nf['tau_fine'].values, it,
                marker='<', linewidth=2, markersize=6, linestyle='--', 
                label='PF - tau_f /Nf')

            # Format plot
            ax.set_xlabel('Tau', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Iterations', fontsize=14, fontweight='bold')
            ax.set_title(f'Iterations vs Tau (Nf={nf}, Nc={nc})', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.5, linestyle='--')
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
            ax.set_yscale('log')

            plt.tight_layout()
            plt.show()

print("Done!")
