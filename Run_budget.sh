# Batch runner for computational-budget experiments.
#
# It executes define_budget.py over several mesh sizes (log2(h)) and repeats
# each configuration multiple times to average timing noise.

source ~/venv-firedrake/bin/activate

# Mesh sizes to benchmark.
log2h=(-4 -5 -6 -7 -8 -9 -10) # -11)

# Repeat each benchmark point to collect more stable timing statistics.
for hi in "${log2h[@]}"
do
    echo "Running budget definition for h = $hi"
    for i in {1..10}
    do
        python3 define_budget.py --log2h $hi
        echo "Iteration $i completed for h = $hi"
    done
done