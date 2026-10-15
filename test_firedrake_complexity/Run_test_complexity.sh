# Batch runner for computational-budget experiments.
#
# It executes define_budget.py over several mesh sizes (log2(h)) and repeats
# each configuration multiple times to average timing noise.

source ~/venv-firedrake/bin/activate

# Mesh sizes to benchmark.
log2h=(-4 -5 -6 -7 -8 -9 -10) # -11)
solver='cg'
# log2h=(-7)

cd ~/Desktop/tesi/tesi

# Repeat each benchmark point to collect more stable timing statistics.
for hi in "${log2h[@]}"; do
    echo "Running budget definition for h = $hi"
    for i in {1..10}; do
        if [[ "$i" -eq 1 ]]; then
            python3 test_firedrake_complexity/test_complexity.py \
                --solver "$solver" \
                --log2h "$hi" \
                --save_txt True \
                > "test_firedrake_complexity/${solver}_solver/log/solver_output_${hi}.txt" 2>&1
        else
            python3 test_firedrake_complexity/test_complexity.py \
                --solver "$solver" \
                --log2h "$hi" \
                --save_txt False
        fi
        echo "  Done iteration $i"
    done
done