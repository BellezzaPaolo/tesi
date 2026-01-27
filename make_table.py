import pandas as pd

# -----------------------
# Load CSV files
# -----------------------
df_PF = pd.read_csv("./incontro2/PF.csv")
df_GD = pd.read_csv("./incontro2/GD.csv")

# Extract unique values
N_fine_vals   = df_PF["N_fine"].unique()
N_coarse_vals = df_PF["N_coarse"].unique()
tau_vals      = df_PF["tau_fine"].unique()
fine_solver   = df_PF["fine_operator"].unique()
coarse_solver = df_PF["coarse_operator"].unique()

# Output LaTeX file
filename_table = "./paraflow_table.tex"

with open(filename_table, "w") as f:

    for fs in fine_solver:
        for cs in coarse_solver:
            f.write(r'\begin{table}')
            f.write(r'  \centering')
            f.write("\n")
            f.write(r'      \resizebox{\textwidth}{!}{')
            # -----------------------
            # Header: column format
            # -----------------------
            ncols = sum(len(N_coarse_vals) for _ in N_fine_vals)
            f.write(r"      \begin{tabular}{|c||c|" + "c|" * ncols + "}\n")
            f.write(r"      \hline" + "\n")

            # -----------------------
            # First header row: N_fine
            # -----------------------
            f.write(r"      $\tau_f$  & GD ")
            for Nf in N_fine_vals:
                f.write(rf"& \multicolumn{{{len(N_coarse_vals)}}}{{c|}}{{$N_{{fine}}={Nf}$}} ")
            f.write(r"\\ \hline" + "\n")

            # -----------------------
            # Second header row: N_coarse
            # -----------------------
            f.write("     &     ")
            for _ in N_fine_vals:
                for Nc in N_coarse_vals:
                    f.write(rf"& ${Nc}$ ")
            f.write(r"\\ \hline\hline" + "\n")

            # -----------------------
            # Table body
            # -----------------------
            for tau in tau_vals:
                f.write(rf"       ${tau}$ ")
                f.write(rf"& {int(df_GD[(df_GD['tau'] == tau) & (df_GD['optimizer_name'] == fs)]['iterate'].values[0])} ")

                for Nf in N_fine_vals:
                    for Nc in N_coarse_vals:

                        df_filt = df_PF[
                            (df_PF["tau_fine"] == tau) &
                            (df_PF["N_fine"] == Nf) &
                            (df_PF["N_coarse"] == Nc) &
                            (df_PF["fine_operator"] == fs) &
                            (df_PF["coarse_operator"] == cs)]

                        if df_filt.empty:
                            f.write(r"& -- ")
                        else:
                            row = df_filt.iloc[0]
                            cell = (
                                f"{int(row['iterate_fine'])} | "
                                f"{int(row['iterate_coarse'])}"
                            )
                            if int(row['iterate_fine']) + int(row['iterate_coarse']) <= int(df_GD[(df_GD['tau'] == tau) & (df_GD['optimizer_name'] == fs)]['iterate'].values[0]):
                                f.write(rf"& \cellcolor{{gray}}${cell}$")
                            else:
                                f.write(rf"& ${cell}$")

                f.write(r"\\ \hline" + "\n")

            # -----------------------
            # End table
            # -----------------------
            f.write(r"      \end{tabular}" + "\n" + r"      }" + "\n")
            f.write(r"      \caption{ParaflowS results with fine operator $" + fs + r"$, coarse operator $" + cs + r"$ and calls fine | calls coarse}")
            f.write(r"    \label{tab:Paraflow" + fs + r"}" + "\n")
            f.write(r"\end{table}")

print(f"Table saved in {filename_table}")