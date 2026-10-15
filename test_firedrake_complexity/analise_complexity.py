import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import sys
from pathlib import Path

# Parse command-line arguments before importing Firedrake so PETSc does not see `--log2h`.
parser = argparse.ArgumentParser(description='Benchmark gradient assembly and step timings', add_help=True)
parser.add_argument('--solver', type=str, required=False, default= None,
                    help = 'solver ot be used')
args, remaining_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining_args]
solver = args.solver

folder = '~/Desktop/tesi/tesi/test_firedrake_complexity/' + solver +'_solver/' if solver is not None else '~/Desktop/tesi/tesi/test_firedrake_complexity/default_solver/'
results_file = (Path(folder) / "times.csv").expanduser()
output_dir = (Path(folder) / "images").expanduser()
output_dir.mkdir(exist_ok=True)

# Number of coarsest mesh sizes to omit from the plots.
discard_first_N = 2

df = pd.read_csv(results_file)
required_columns = {"h", "N"}
if not required_columns.issubset(df.columns):
    missing = ", ".join(sorted(required_columns - set(df.columns)))
    raise ValueError(f"Missing required column(s): {missing}")

# Average the repeated runs at each mesh size and keep the mesh ordering explicit.
df_mean = df.groupby(["h", "N"], as_index=False).mean(numeric_only=True).sort_values("N")
if not 0 <= discard_first_N < len(df_mean):
    raise ValueError(
        f"discard_first_N must be between 0 and {len(df_mean) - 1}, "
        f"got {discard_first_N}"
    )
df_mean = df_mean.iloc[discard_first_N:].reset_index(drop=True)
N = df_mean["N"]

timing_groups = {
    "assembly": ["assemble_mass", "assemble_laplacian", "assemble_system"],
    "solver_definition": [
        "definition_solver_mass",
        "definition_solver_laplacian",
        "definition_solver_system",
    ],
    "solve": ["solve_mass", "solve_laplacian", "solve_system"],
    "solve_2": ["solve_2_laplacian", "solve_2_mass", "solve_2_system"],
    "solve_3": ["solve_3_laplacian", "solve_3_mass", "solve_3_system"]
}
labels = {
    "mass": "Mass",
    "laplacian": "Laplacian",
    "system": "System",
}
colors = {"mass": "#0072B2", "laplacian": "#D55E00", "system": "#009E73"}


def calculate_ratio(x_values, y_values, index):
    x_previous, x_current = x_values.iloc[index - 1], x_values.iloc[index]
    y_previous, y_current = y_values.iloc[index - 1], y_values.iloc[index]
    return (y_current - y_previous) / (x_current - x_previous) * (x_current / y_current)


def annotate_ratio_boxes(axis, x_values, y_values, color, vertical_offset):
    """Show each point's plotted ratio in a small square below the x-axis."""
    for index, x_value in enumerate(x_values):
        ratio_text = "-" if index == 0 else f"{calculate_ratio(x_values, y_values, index):.2e}"
        axis.annotate(
            ratio_text,
            (x_value, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7,
            bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "0.5"},
            clip_on=False,
            color=color
        )


for group_name, columns in timing_groups.items():
    missing = [column for column in columns if column not in df_mean.columns]
    if missing:
        raise ValueError(f"Missing timing column(s): {', '.join(missing)}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xlabel("Number of degrees of freedom (N)")
        axis.set_ylabel("Time [s]")
        axis.grid(True, which="both", alpha=0.3)

    for column in columns:
        operator = column.rsplit("_", 1)[-1]
        values = df_mean[column]
        label = labels[operator]
        axes[0].plot(N, values, "o-", linewidth=2, markersize=5,
                     label=label, color=colors[operator])

        normalized_values = values / values.iloc[0]
        axes[1].plot(N, normalized_values, "o-", linewidth=2,
                     markersize=5, label=label, color=colors[operator])
        vertical_offset = -22 - 18 * list(labels).index(operator)
        annotate_ratio_boxes(axes[0], N, values, colors[operator], vertical_offset)
        annotate_ratio_boxes(axes[1], N, normalized_values, colors[operator], vertical_offset)

        
    axes[1].plot(N, N/N[0], "k--", linewidth=1, alpha=0.5, label="O(N)")
    axes[1].plot(N, N**0.5/N[0]**0.5, "k:", linewidth=1, alpha=0.5, label="O(N^0.5)")
    axes[0].plot(N, N, "k--", linewidth=1, alpha=0.5, label="O(N)")
    axes[0].plot(N, N**0.5, "k:", linewidth=1, alpha=0.5, label="O(N^0.5)")
    axes[0].set_title(f"{group_name.replace('_', ' ').title()} time")
    axes[1].set_title(f"{group_name.replace('_', ' ').title()} time (normalized)")
    axes[1].set_ylabel("Time / time on coarsest mesh")
    axes[0].legend()
    axes[1].legend()
    fig.savefig(output_dir / f"time_{group_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

system_timing_columns = {
    "Assembly": "assemble_system",
    "Solver definition": "definition_solver_system",
    "First solve": "solve_system",
    "Repeated solve (LU)": "solve_2_system",
}
missing = [column for column in system_timing_columns.values() if column not in df_mean.columns]
if missing:
    raise ValueError(f"Missing system timing column(s): {', '.join(missing)}")

fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
axis.set_xscale("log", base=2)
axis.set_yscale("log", base=2)
axis.set_xlabel("Number of degrees of freedom (N)")
axis.set_ylabel("Time [s]")
axis.set_title("System solution times")
axis.grid(True, which="both", alpha=0.3)

for label, column in system_timing_columns.items():
    axis.plot(N, df_mean[column], "o-", linewidth=2, markersize=5, label=label)

axis.legend()
fig.savefig(output_dir / "time_system_all.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {len(timing_groups) + 1} figures to {output_dir}")