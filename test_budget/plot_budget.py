# Import plotting dependencies only when needed.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv('~/Desktop/tesi/tesi/test_budget/result.csv',#'./results/Budget_definition_pointwise.csv',
                    dtype={"method": str, "adaptivity": int, "h": float, "N": int, "time_assemble": float, "time_step": float})

# Discard smallest meshes to improve readability in log-log scaling plots.
discard_first_N = 2

#--------------------------------------------------------------------------------------------------------------------------------
# Start of markdown table generation.
#--------------------------------------------------------------------------------------------------------------------------------
# Average repeated measurements with the same method, adaptivity, and h.
table = (
    df.groupby(['method', 'adaptivity', 'h'], as_index=False)
      .agg(
          N=('N', 'first'),
          time_assemble=('time_assemble', 'mean'),
          time_step=('time_step', 'mean'),
      )
)
# Output path for the markdown table.
output_dir = Path('~/Desktop/tesi/tesi/test_budget').expanduser()
output_file = output_dir / 'budget_table.md'
output_file.parent.mkdir(parents=True, exist_ok=True)

# Filter out the smallest meshes according to `discard_first_N` before writing the table
unique_n = np.sort(df['N'].unique())
keep_n = unique_n[discard_first_N:]
table_filtered = table[table['N'].isin(keep_n)].copy()

# Construct the markdown table as a list of lines, starting with the header and separator, then adding each row formatted appropriately.
columns = list(table_filtered.columns)
lines = [
    "| " + " | ".join(columns) + " |",
    "| " + " | ".join(["---"] * len(columns)) + " |",
]
for row in table_filtered.itertuples(index=False, name=None):
    formatted_row = []
    for value in row:
        if isinstance(value, float):
            formatted_row.append(f'{value:.3}')
        else:
            formatted_row.append(str(value))
    lines.append("| " + " | ".join(formatted_row) + " |")
output_file.write_text("\n".join(lines) + "\n")

#-----------------------------------------------------------------------------------------------------------------------------------------
# End of markdown table generation.
#-----------------------------------------------------------------------------------------------------------------------------------------

# Extract unique values for h, method, adaptivity, and N, discarding the smallest meshes according to `discard_first_N`.
h = (df["h"].unique())[discard_first_N:]
type = df['method'].unique()
adaptivity = df['adaptivity'].unique()
N = (df['N'].unique())[discard_first_N:]
#N = np.sqrt(N) # since N is the number of dofs, we take the square root to have the number of refinements

T_assemble = []
T_step = []

for t in type:
    for ada in adaptivity:
        df_sub = df[(df['method'] == t) & (df['adaptivity'] == ada)]
        if df_sub.empty:
            continue
        Ta_m = []
        Ts_m = []
        for hi in h:
            df_filtered = df_sub[df_sub['h'] == hi]
            if df_filtered.empty:
                Ta_m.append(np.nan)
                Ts_m.append(np.nan)
                continue
            Ta_m.append(df_filtered["time_assemble"].mean())
            Ts_m.append(df_filtered["time_step"].mean())
            #print(f'{t} {"adaptive" if ada == 1 else "fixed"}: h: {hi}, time assemble: {Ta_m[-1]}, time step: {Ts_m[-1]}')

        T_assemble.append((t,ada,Ta_m))
        T_step.append((t,ada,Ts_m))

#-----------------------------------------------------------------------------------------------------------------------------------------
# Plotting section.
#-----------------------------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1,2, figsize=(20,10))

#First subplot: time_assemble vs N, with log-log scaling.
ax[0].set_title('Assemble Time')
ax[0].set_yscale('log', base = 2)
ax[0].set_xscale('log', base = 2)
# ax[0].plot(N, N, 'k--', label='O(N)')
# ax[0].plot(N, (N)**2, 'k:', label='O(N^2)')
for method, ada, Ta_m in T_assemble:
    ax[0].plot(N, Ta_m, 'o-',linewidth=3, markersize=7, label=f'{method} {"adaptive" if ada == 1 else "fixed"}')
# ax[0].plot(N,(N)**1.5, 'k*',label = 'O(N^1.5)')
# ax[0].plot(N,(N)**0.5, 'k.',label = 'O(N^0.5)')
ax[0].set_xlabel('N')
ax[0].set_ylabel('Time assemble [s]')
ax[0].legend()
ax[0].grid()

# Second subplot: time_assemble normalized by the first value (coarsest mesh) vs N, with log-log scaling and reference lines for O(N) and O(N^0.5).
ax[1].set_title('Assemble Time Normalized')
ax[1].set_yscale('log', base = 2)
ax[1].set_xscale('log', base = 2)
ax[1].plot(N, N/N[0], 'k--', label='O(N)')
ax[1].plot(N,(N/N[0])**0.5, 'k:',label = 'O(N^0.5)')
# ax[1].plot(N, (N/N[0])**2, 'k:', label='O(N^2)')
for method, ada, Ta_m in T_assemble:
    ax[1].plot(N, Ta_m/Ta_m[0], 'o-',linewidth=3, markersize=7, label=f'{method} {"adaptive" if ada == 1 else "fixed"}')
# ax[1].plot(N,(N/N[0])**1.5, 'k*',label = 'O(N^1.5)')
ax[1].set_xlabel('N')
ax[1].set_ylabel('Time assemble [s]')
ax[1].legend()
ax[1].grid()

# Save the figure for the assemble time plots.
fig.savefig(output_dir / 'images/time_assemble.png', dpi=300, bbox_inches='tight')



fig, ax = plt.subplots(1,2, figsize=(20,10))
# First subplot: time_step vs N, with log-log scaling.
ax[0].set_yscale('log', base = 2)
ax[0].set_xscale('log', base = 2)
# ax[0].plot(N, (N), 'k--', label='O(N)')
# ax[0].plot(N, (N)**2, 'k:', label='O(N^2)')
for method, ada, Ts1_m in T_step:
    ax[0].plot(N, Ts1_m, 'o-', linewidth=3, markersize=7, label=f'{method} {"adaptive" if ada == 1 else "fixed"}')
# ax[0].plot(N,(N)**0.75, 'k*',label = 'O(N^0.75)')
ax[0].set_xlabel('Number of refinements')#, fontsize=28, fontweight='bold')
ax[0].set_ylabel('Computational time [s]')#, fontsize=28, fontweight='bold')
ax[0].set_title('Computational time per iteration')#, fontsize=30, fontweight='bold')
ax[0].legend()#fontsize=28)#, fontweight='bold')
ax[0].tick_params()#axis='both', which='major', labelsize=25)
ax[0].grid()

# Second subplot: time_step normalized by the first value (coarsest mesh) vs N, with log-log scaling and reference lines for O(N) and O(N^0.75).
ax[1].set_yscale('log', base = 2)
ax[1].set_xscale('log', base = 2)
ax[1].plot(N, (N/N[0]), 'k--', label='O(N)') # , linewidth=3, markersize=10
ax[1].plot(N,(N/N[0])**0.75, 'k:',label = 'O(N^0.75)')
# ax[1].plot(N, (N/N[0])**2, 'k:', label='O(N^2)')
for method, ada, Ts1_m in T_step:
    ax[1].plot(N, Ts1_m/Ts1_m[0], 'o-', linewidth=3, markersize=7, label=f'{method} {"adaptive" if ada == 1 else "fixed"}')
ax[1].set_xlabel('Number of refinements')#, fontsize=28, fontweight='bold')
ax[1].set_ylabel('Computational time [s]')#, fontsize=28, fontweight='bold')
ax[1].set_title('Computational time per iteration')#, fontsize=30, fontweight='bold')
ax[1].legend()#fontsize=28)#, fontweight='bold')
ax[1].tick_params()#axis='both', which='major', labelsize=25)
ax[1].grid()

# Save the figure for the time step plots.
fig.savefig(output_dir / 'images/time_step.png', dpi=300, bbox_inches='tight')

plt.show()