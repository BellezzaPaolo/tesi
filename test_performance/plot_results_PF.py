import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

if len(sys.argv) > 1:
    test_case_num = sys.argv[1]
    gamma_param = float(sys.argv[2])
else:
    raise ValueError('Test case number and parameter gamma must be provided as argument. \n Possible values for tests are: 1.1, 1.2, 1.3, 2, 3. While gamma is a float value, e.g. 0.5, 1.0, 2.0, etc. or "adaptive" for adaptive gamma.')

# xmin = 0.013
# xmax = 100

case_folder = Path(__file__).parent / f"test_case{test_case_num[0]}"


# Load GD results
df_gd = pd.read_csv(case_folder / 'GD.csv',
                    dtype={"method": str, "h": float, "test_name": str, "tau": float, "energy": float, "lambda": float, "iterate": int, "error": float, "total_time": float, "mean_time": float})

# Load PF results
files_PF = ['PF_dt_Nf.csv', 'PF_dt.csv', 'PF_dt*Nf_2.csv', 'PF_dt*Nf.csv', 'PF_dt*Nf*1.5.csv',  'PF_dt*Nf*2.csv', 'PF_ada.csv']

df_pf = []

for f in files_PF:
    df_pf.append({'table': pd.read_csv(case_folder / f,
                        dtype={"fine_operator": str, "coarse_operator": str, "gamma": float, "h": float, "test_name": str, "N_fine": int, "N_coarse": int, "tau_fine": float, "tau_coarse": float, 
                            "energy": float, "lambda": float, "iterate_coarse": int, "iterate_fine": int, "iterate": int, "error": float, "total_time": float, "mean_time": float}),
                            'name': f.split('.')[0]})

# Filter data for the specified test case and gamma parameter.
filtered_gd = df_gd[df_gd['test_name'] == test_case_num]

# filtered_pf = []
# for elem in df_pf:
#     filtered_pf.append([elem['table'][elem['table']['test_name'] == test_case_num], elem['name']])


#################################################################################################################################################
# Make the table with the best results for each method and save it as markdown file.
##################################################################################################################################################

# For each method, find the row with the best performance (fewest total iterations, then least time).
# def best_row_for_method(group: pd.DataFrame) -> pd.Series:
#     # Pick the run with the fewest total iterations across fine and coarse levels.
#     return group.assign(
#         total_iterate=group['iterate_fine'] + group['iterate_coarse']
#     ).sort_values(['total_iterate', 'total_time'], ascending=[True, True]).iloc[0]


# summary_rows = []
# for df in filtered_pf:
#     for method_name, group in df.groupby('fine_operator', sort=True):
#         best = best_row_for_method(group)
#         summary_rows.append({
#             'method': method_name,
#             'min_iterations': int(best['iterate_fine']+ best['iterate_coarse']),
#             'iterate_fine': int(best['iterate_fine']),
#             'iterate_coarse': int(best['iterate_coarse']),
#             'time_used_s': float(best['total_time']),
#             'tau_fine': float(best['tau_fine']),
#             'tau_coarse': float(best['tau_coarse']),
#             'gamma': float(best['gamma']),
#         })

# summary_df = pd.DataFrame(summary_rows).sort_values('method').reset_index(drop=True)
# summary_df['time_used_s'] = summary_df['time_used_s'].map(lambda x: round(x, 6))

# # Build markdown table manually (same style as plot_budget.py).
# columns = list(summary_df.columns)
# lines = [
#     "| " + " | ".join(columns) + " |",
#     "| " + " | ".join(["---"] * len(columns)) + " |",
# ]
# for row in summary_df.itertuples(index=False, name=None):
#     formatted_row = []
#     for value in row:
#         if isinstance(value, float):
#             formatted_row.append(f'{value:.6f}')
#         else:
#             formatted_row.append(str(value))
#     lines.append("| " + " | ".join(formatted_row) + " |")

# table_path = case_folder / f"PF_table_{test_case_num}.md"
# table_path.write_text("\n".join(lines) + "\n")
# print(f"Saved markdown summary table to: {table_path}")

####################################################################################################################################
# Start the plotting of the results
#####################################################################################################################################

# color = {"L2_explicit":'#e377c2', "L2_semimplicit": '#1f77b4',"H1_explicit": '#9467bd', 
#        "a0_explicit": '#d62728', "az_explicit": '#ff7f0e', "az_semimplicit": "#2ca02c"}

color = ['#1f77b4', '#9467bd', '#ff7f0e', '#d62728', "#2ca02c", '#e377c2', '#bcbd22', '#17becf']

opt = []
for m in sorted(df_gd['method'].unique()):
    if not(m[-3:] == 'ada'):
        opt.append(m)

print(f"Unique methods: {opt}")

summary_rows = []
for i,name in enumerate(opt):
    if not(name == 'H1_explicit'):
        fig,ax = plt.subplots(1,2,figsize=(20, 10))

        row =   {
            'optimizer': None,
            'method': None,
            'min_iterations': float('inf'),
            'iterate_fine': None,
            'iterate_coarse': None,
            'time_used_s': None,
            'time_per_step': None,
            'tau_fine': None,
            'tau_coarse': None,
            'ratio': None,
            'gamma': None,
        }

        for j, df in enumerate(df_pf[:-1]): # Exclude the last one (adaptive gamma) for now
            subset = df['table'][df['table'][df['table']['test_name'] == test_case_num]['fine_operator'] == name].sort_values('tau_fine')

            it = (subset['iterate_fine'] + subset['iterate_coarse']).values.copy()
            ax[0].loglog(subset['tau_fine'].values, it, marker='o', c = color[j], label = df['name'])

            ax[1].loglog(subset['tau_fine'].values, subset['total_time'].values, marker='o', c = color[j], label = df['name'])

            k = j

            # table

            if it.min() < row['min_iterations']:
                row['optimizer'] = 'PF '
                row['method'] = name
                row['min_iterations'] = int(it.min())
                best_row = subset.iloc[int(it.argmin())]
                row['iterate_fine'] = int(best_row['iterate_fine'])
                row['iterate_coarse'] = int(best_row['iterate_coarse'])
                row['time_used_s'] = float(best_row['total_time'])
                row['tau_fine'] = float(best_row['tau_fine'])
                row['tau_coarse'] = float(best_row['tau_coarse'])
                row['ratio'] = df['name']
                row['gamma'] = float(best_row['gamma'])
                row['time_per_step'] = float(best_row['total_time']) / float(it.min())

            
        summary_rows.append(row)

        xmin = subset['tau_fine'].values.min() + 0.05
        xmax = subset['tau_fine'].values.max() + 0.5

        if name == 'a0_explicit' or name == 'az_explicit' or name == 'az_semimplicit':
            filtered_ada = df_pf[-1]['table'][df_pf[-1]['table'][df_pf[-1]['table']['test_name'] == test_case_num]['fine_operator'] == name + '_ada'].sort_values('gamma')

            for ind, gamma in enumerate(filtered_ada['gamma'].values):

                label = name + '_ada' + "_gamma" if not(gamma == 1.0) else name + '_ada'

                it = (filtered_ada[(filtered_ada['gamma'] == gamma)]['iterate_fine'] + filtered_ada[(filtered_ada['gamma'] == gamma)]['iterate_coarse']).values.copy()
                tau = filtered_ada[(filtered_ada['gamma'] == gamma)]['tau_fine'].values.copy()
                ax[0].loglog([xmin,xmax], [it,it], '--', label= label, color = color[k + ind])
                ax[0].loglog(tau, it, marker = 's', color = color[k + ind])

                time_tot = (filtered_ada[(filtered_ada['gamma'] == gamma)]['total_time'].values.copy())
                ax[1].loglog([xmin,xmax], [time_tot,time_tot], '--',label= label, color = color[k + ind])
                ax[1].loglog(tau, time_tot, marker = 's', color = color[k + ind])

                summary_rows.append({
                    'optimizer': 'PF ',
                    'method': label,
                    'min_iterations': int(it.min()),
                    'iterate_fine': int(filtered_ada[(filtered_ada['gamma'] == gamma)]['iterate_fine'].values.copy()[it.argmin()]),
                    'iterate_coarse': int(filtered_ada[(filtered_ada['gamma'] == gamma)]['iterate_coarse'].values.copy()[it.argmin()]),
                    'time_used_s': float(filtered_ada[(filtered_ada['gamma'] == gamma)]['total_time'].values.copy()[it.argmin()]),
                    'time_per_step': float(filtered_ada[(filtered_ada['gamma'] == gamma)]['total_time'].values.copy()[it.argmin()]) / float(it.min()),
                    'tau_fine': float(filtered_ada[(filtered_ada['gamma'] == gamma)]['tau_fine'].values.copy()[it.argmin()]),
                    'tau_coarse': float(filtered_ada[(filtered_ada['gamma'] == gamma)]['tau_coarse'].values.copy()[it.argmin()]),
                    'ratio': None, 
                    'gamma': float(gamma),
                })

        # gradient descent results for comparison
        subset = filtered_gd[filtered_gd['method'] == name].sort_values('tau')
        ax[0].loglog(subset['tau'].values, subset['iterate'].values, marker='o', label=name, color='black')

        ax[1].loglog(subset['tau'].values, subset['total_time'].values, marker='o', label=name, color='black')

        summary_rows.append({
            'optimizer': 'GD ',
            'method': name,
            'min_iterations': int(subset['iterate'].min()),
            'iterate_fine': None,
            'iterate_coarse': None,
            'time_used_s': float(subset['total_time'].min()),
            'time_per_step': float(subset['total_time'].min()) / float(subset['iterate'].min()),
            'tau_fine': float(subset['tau'].values[0]),
            'tau_coarse': None,
            'ratio': None,
            'gamma': None,
        })


        if name == 'a0_explicit' or name == 'az_explicit' or name == 'az_semimplicit':
            it = filtered_gd[filtered_gd['method'] == name + '_ada']['iterate'].values
            tau = filtered_gd[filtered_gd['method'] == name + '_ada']['tau'].values
            ax[0].loglog([xmin,xmax], [it,it], label=name, linestyle='--', color='black')
            ax[0].loglog(tau, it, marker = 's', color='black')

            time_tot = filtered_gd[filtered_gd['method'] == name + '_ada']['total_time'].values
            ax[1].loglog([xmin,xmax], [time_tot,time_tot], '--',label=name, color='black')
            ax[1].loglog(tau, time_tot, marker = 's', color='black')

            summary_rows.append({
                'optimizer': 'GD ',
                'method': name + '_ada',
                'min_iterations': int(subset['iterate'].min()),
                'iterate_fine': None,
                'iterate_coarse': None,
                'time_used_s': float(subset['total_time'].min()),
                'time_per_step': float(subset['total_time'].min()) / float(subset['iterate'].min()),
                'tau_fine': float(subset['tau'].values[0]),
                'tau_coarse': None,
                'ratio': None,
                'gamma': None,
            })

        ax[0].set_xlim([xmin,xmax])
        ax[0].set_ylim([None, 450])
        ax[0].set_xlabel('Tau')
        ax[0].set_ylabel('Iterate')
        ax[0].set_title('Number of iterations')
        ax[0].legend()
        ax[0].grid(True, which="both", ls="--")

        ax[1].set_xlim([xmin,xmax])
        ax[1].set_ylim([None, 1000])
        ax[1].set_xlabel('Tau')
        ax[1].set_ylabel('Total Time (s)')
        ax[1].set_title('Total Time')
        ax[1].legend()
        ax[1].grid(True, which="both", ls="--")

        fig.suptitle(f"Results for method: {name} and test case {test_case_num}", fontsize=16)


        fig.savefig(case_folder / "images" / f"PF_results{test_case_num}_{name}.png")
        # plt.show()


summary_df = pd.DataFrame(summary_rows).sort_values('method').reset_index(drop=True)
summary_df['time_used_s'] = summary_df['time_used_s'].map(lambda x: round(x, 6))

# Build markdown table manually (same style as plot_budget.py).
columns = list(summary_df.columns)
lines = [
    "| " + " | ".join(columns) + " |",
    "| " + " | ".join(["---"] * len(columns)) + " |",
]
for row in summary_df.itertuples(index=False, name=None):
    formatted_row = []
    for value in row:
        if isinstance(value, float):
            formatted_row.append(f'{value:.6f}')
        elif value is None:
            formatted_row.append(' - ')
        else:
            formatted_row.append(str(value))
    lines.append("| " + " | ".join(formatted_row) + " |")

table_path = case_folder / f"PF_table_{test_case_num}.md"
table_path.write_text("\n".join(lines) + "\n")
print(f"Saved markdown summary table to: {table_path}")