source ~/venv-firedrake/bin/activate

log2h=(-4 -5 -6 -7 -8 -9 -10) # -11)

for hi in "${log2h[@]}"
do
    echo "Running budget definition for h = $hi"
    for i in {1..10}
    do
        python3 define_budget.py --log2h $hi
        echo "Iteration $i completed for h = $hi"
    done
done