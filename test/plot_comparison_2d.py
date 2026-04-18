import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Configuration
case_folder = Path(__file__).parent / 'case_test3'

# Extract test case number from folder name
test_case_num = case_folder.name.replace('case_test', '')


# Read CSV files
print("Reading CSV files...")
df_gd = pd.read_csv(case_folder / 'GD.csv')
df_pf_dtxNf = pd.read_csv(case_folder / 'PFalpha_dtxNf.csv')
df_pf_dt = pd.read_csv(case_folder / 'PFalpha_dt.csv')
df_pf_dtxNf_2 = pd.read_csv(case_folder / 'PFalpha_dtxNf_2.csv')
df_pf_dtxNf_2x3 = pd.read_csv(case_folder / 'PFalpha_dtxNf_2x3.csv')
df_pf_dt_Nf = pd.read_csv(case_folder / 'PFalpha_dt_Nf.csv')

N_fine = df_pf_dtxNf_2['N_fine'].unique()
print(f"Unique N_fine values: {N_fine}")
N_coarse = df_pf_dtxNf['N_coarse'].unique()
opt = df_pf_dtxNf['fine_operator'].unique()

df_gd_ada = df_gd[df_gd['optimizer_name'] == 'az_ada']

color = {'L2_P': '#1f77b4', 'az': '#ff7f0e'}  # Blue for L2_P, Orange for az

# Plot GD data
    #if name == 'L2_P':
for nf in N_fine:
    for nc in N_coarse:
        fig, ax = plt.subplots(figsize=(14, 10))
        for name in opt:
            print(f"Plotting for GD: {name}, PF: Nf={nf}, Nc={nc}")
            # Create plot

            subset = df_gd[df_gd['optimizer_name'] == name].sort_values('tau')
            it = subset['iterate'].values.copy()  # Convert to numpy array
            if name == 'az':
                it[-1] = 10000
                label = fr'$a_u$-SGF'
            else:
                label = fr'$L^2$-SGF'
            ax.plot(subset['tau'].values, it,
                marker='o', linewidth=5, markersize=12,
                label=label, color=color[name])

            # subset_dtxNf = df_pf_dtxNf[(df_pf_dtxNf['N_fine'] == nf) & (df_pf_dtxNf['N_coarse'] == nc) & (df_pf_dtxNf['fine_operator'] == name)].sort_values('tau_fine')
            # subset_dt = df_pf_dt[(df_pf_dt['N_fine'] == nf) & (df_pf_dt['N_coarse'] == nc) & (df_pf_dt['fine_operator'] == name)].sort_values('tau_fine')
            subset_dtxNf_2 = df_pf_dtxNf_2[(df_pf_dtxNf_2['N_fine'] == nf) & (df_pf_dtxNf_2['N_coarse'] == nc) & (df_pf_dtxNf_2['fine_operator'] == name)].sort_values('tau_fine')
            subset_dtxNf_2x3 = df_pf_dtxNf_2x3[(df_pf_dtxNf_2x3['N_fine'] == nf) & (df_pf_dtxNf_2x3['N_coarse'] == nc) & (df_pf_dtxNf_2x3['fine_operator'] == name)].sort_values('tau_fine')
            # subset_dt_Nf = df_pf_dt_Nf[(df_pf_dt_Nf['N_fine'] == nf) & (df_pf_dt_Nf['N_coarse'] == nc) & (df_pf_dt_Nf['fine_operator'] == name)].sort_values('tau_fine')
            
            # it = (subset_dtxNf['iterate_fine'] + subset_dtxNf['iterate_coarse']).values.copy()
            # if name == 'az':
            #     it[-1] = 10000
            #     label = fr'$a_u$-PF: $\tau_F \times N_F$'
            # else:
            #     label = fr'$L^2$-PF: $\tau_F \times N_F$'
            # ax.plot(subset_dtxNf['tau_fine'].values, it,
            #     marker='s', linewidth=3, markersize=10, linestyle='--',
            #     label=label)

            # it = (subset_dt['iterate_fine'] + subset_dt['iterate_coarse']).values.copy()
            # if name == 'az':
            #     it[-1] = 10000
            #     label = fr'$a_u$-PF: $\tau_F$'
            # else:
            #     label = fr'$L^2$-PF: $\tau_F$'
            # ax.plot(subset_dt['tau_fine'].values, it,
            #     marker='^', linewidth=3, markersize=10,  linestyle='--',
            #     label=label)
            if name == 'az':
                it = (subset_dtxNf_2['iterate_fine'] + subset_dtxNf_2['iterate_coarse']).values.copy()
                if name == 'az':
                    it[-1] = 10000
                    label = fr'$a_u$-PF: $\tau_F \times N_F / 2$'
                else:
                    label = fr'$L^2$-PF: $\tau_F \times N_F / 2$'
                ax.plot(subset_dtxNf_2['tau_fine'].values, it,
                    marker='d', linewidth=3, markersize=10, linestyle='--', color = '#d62728',
                    label=label)
                df_pf_ada = pd.read_csv(case_folder / 'PFalpha_ada.csv')
                ax.plot(subset_dtxNf_2['tau_fine'].values,
                        (df_pf_ada['iterate_fine'] + df_pf_ada['iterate_coarse']).values.copy() * np.ones(subset_dtxNf_2['tau_fine'].values.shape),
                        label = fr'$a_u$-PF: adaptive $\tau_F$')
                ax.plot(subset_dtxNf_2['tau_fine'].values,
                        df_gd_ada['iterate'].values.copy() * np.ones(subset_dtxNf_2['tau_fine'].values.shape),
                        label = fr'$a_u$: adaptive $\tau_F$')

            else:
                it = (subset_dtxNf_2x3['iterate_fine'] + subset_dtxNf_2x3['iterate_coarse']).values.copy()
                if name == 'az':
                    it[-1] = 10000
                    label = fr'$a_u$-PF: $\tau_F \times N_F \times 1.5$'
                else:
                    label = fr'$L^2$-PF: $\tau_F \times N_F \times 1.5$'
                ax.plot(subset_dtxNf_2x3['tau_fine'].values, it,
                    marker='v', linewidth=3, markersize=10, linestyle='--', color = '#2ca02c',
                    label=label)

            # it = (subset_dt_Nf['iterate_fine'] + subset_dt_Nf['iterate_coarse']).values.copy()
            # if name == 'az':
            #     it[-1] = 10000
            #     label = fr'$a_u$-PF: $\tau_F / N_F$'
            # else:
            #     label = fr'$L^2$-PF: $\tau_F / N_F$'
            # ax.plot(subset_dt_Nf['tau_fine'].values, it,
            #     marker='<', linewidth=3, markersize=10, linestyle='--', 
            #     label=label)

            # Calculate best values
            gd_iterations = subset['iterate'].values.copy()
            if name == 'az':
                gd_iterations = gd_iterations[:-1]  # Exclude the last modified value
            best_gd = int(gd_iterations.min())
            
            # Collect all PF iterations (excluding modified values)
            pf_iterations_all = []
            for pf_subset in [ subset_dtxNf_2, subset_dtxNf_2x3]:#[subset_dtxNf, subset_dt, subset_dtxNf_2, subset_dtxNf_2x3, subset_dt_Nf]:
                it_vals = (pf_subset['iterate_fine'] + pf_subset['iterate_coarse']).values.copy()
                if name == 'az':
                    it_vals = it_vals[:-1]  # Exclude the last modified value
                pf_iterations_all.extend(it_vals)
            
            best_pf = int(min(pf_iterations_all))
            
            # Format plot
            ax.set_xlabel(r'Time step $\tau$', fontsize=30, fontweight='bold')
            ax.set_ylabel('Computational cost', fontsize=30, fontweight='bold')
            ax.set_title(fr'Computational cost vs time step $\tau$ ($N_F$={nf}, $N_G$={nc})', fontsize=30, fontweight='bold')
            ax.legend(loc='upper center', fontsize=30, framealpha=0.9, ncol=2)
            
            # Add best values as text annotation
            if name == 'az':
                textstr = f'Best $a_u$-SGF: {best_gd}\nBest $a_u$-PF: {best_pf}'
            else:
                textstr = f'Best $L^2$-SGF: {best_gd}\nBest $L^2$-PF: {best_pf}'
            # ax.text(0.012, 0.12, textstr, transform=ax.transAxes, fontsize=28,
            #     verticalalignment='top', fontweight='bold',
            #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.set_yscale('log')
            ax.set_xscale('log')

            # Fix y-axis for az method in test case 1
            if test_case_num == '1':
                ax.set_ylim([20, 300]) # ax.set_ylim([20, 500])
                ax.set_xlim([0.1, 10])
            # Fix y-axis for az method in test case 1
            if test_case_num == '2':
                ax.set_ylim([15, 2100]) # ax.set_ylim([20, 300])
                ax.set_xlim([0.1, 10])
            # Fix y-axis for az method in test case 1
            if test_case_num == '3':
                ax.set_ylim([20, 1000]) # ax.set_ylim([30, 1300])
                ax.set_xlim([0.1, 10])
            
            ax.grid(True, alpha=0.5, linestyle='-', which='both')
            
            # Set tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=30)
            ax.tick_params(axis='both', which='minor', labelsize=30)

            plt.tight_layout()
        plt.show()
        # exit()
        # plt.savefig(case_folder / f'plot_{test_case_num}_comp_Nf{nf}_Nc{nc}.png', dpi=300)
        # plt.close()

print("Done!")
