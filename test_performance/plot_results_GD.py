import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

if len(sys.argv) > 1:
    test_case_num = sys.argv[1]
else:
    raise ValueError('Test case number must be provided as argument. possible values are: 1.1, 1.2, 1.3, 2, 3')

xmin = 0.013
xmax = 100

case_folder = Path(__file__).parent / f"test_case{test_case_num[0]}"

df_gd = pd.read_csv(case_folder / 'GD.csv',
                    dtype={"method": str, "h": float, "test_name": str, "tau": float, "energy": float, "lambda": float, "iterate": int, "error": float, "total_time": float, "mean_time": float})

filtered_gd = df_gd[df_gd['test_name'] == test_case_num]

opt = sorted(filtered_gd['method'].unique())
print(f"Unique methods: {opt}")


def best_row_for_method(group: pd.DataFrame) -> pd.Series:
    # Pick the run with minimum iterations; break ties with minimum total time.
    return group.sort_values(['iterate', 'total_time'], ascending=[True, True]).iloc[0]


summary_rows = []
for method_name, group in filtered_gd.groupby('method', sort=True):
    best = best_row_for_method(group)
    summary_rows.append({
        'method': method_name,
        'min_iterations': int(best['iterate']),
        'time_used_s': float(best['total_time'])
    })

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
        else:
            formatted_row.append(str(value))
    lines.append("| " + " | ".join(formatted_row) + " |")

table_path = case_folder / f"GD_table_{test_case_num}.md"
table_path.write_text("\n".join(lines) + "\n")
print(f"Saved markdown summary table to: {table_path}")


color = {"L2_explicit":'#e377c2', "L2_semimplicit": '#1f77b4',"H1_explicit": '#9467bd', 
       "a0_explicit": '#d62728', "az_explicit": '#ff7f0e', "az_semimplicit": "#2ca02c"}


fig,ax = plt.subplots(1,2,figsize=(20, 10))

for i,name in enumerate(opt):
    subset = filtered_gd[filtered_gd['method'] == name].sort_values('tau')
    if name[-3:] == 'ada':
        it = subset['iterate'].values
        ax[0].loglog([xmin,xmax], [it,it], '--', linewidth = 3, label=name, color=color[name[:-4]])
        ax[0].loglog(subset['tau'].values, it, marker = 's', markersize=8, color=color[name[:-4]])

        time_tot = subset['total_time'].values
        ax[1].loglog([xmin,xmax], [time_tot,time_tot],'--', linewidth = 3, label=name, color=color[name[:-4]])
        ax[1].loglog(subset['tau'].values, time_tot, marker = 's', markersize=8, color=color[name[:-4]])
    else:
        ax[0].loglog(subset['tau'].values, subset['iterate'].values, '-', linewidth = 3, marker='o', label=name, color=color[name])

        ax[1].loglog(subset['tau'].values, subset['total_time'].values, '-', linewidth = 3, marker='o', label=name, color=color[name])

ax[0].set_xlim([xmin,xmax])
ax[0].set_ylim([None, 450])
ax[0].set_xlabel('Tau')
ax[0].set_ylabel('Iterate')
ax[0].set_title(f'Results for Test Case {test_case_num}')
ax[0].legend()
ax[0].grid(True, which="both", ls="--")
ax[0].tick_params(axis='both', which='major')
ax[0].tick_params(axis='both', which='minor')

ax[1].set_xlim([xmin,xmax])
ax[1].set_ylim([None, 1000])
ax[1].set_xlabel('Tau')
ax[1].set_ylabel('Total Time (s)')
ax[1].set_title(f'Results for Test Case {test_case_num}')
ax[1].legend()
ax[1].grid(True, which="both", ls="--")
ax[0].tick_params(axis='both', which='major')
ax[0].tick_params(axis='both', which='minor')


fig.savefig(case_folder / "images" / f"GD_results{test_case_num}.png")
plt.show()

