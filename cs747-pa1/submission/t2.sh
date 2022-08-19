i=1
for inst in "../instances/instances-task2/i-1.txt" "../instances/instances-task2/i-2.txt" "../instances/instances-task2/i-3.txt" "../instances/instances-task2/i-4.txt" "../instances/instances-task2/i-5.txt"; do
for scle in 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3; do
for rs in {0..49}; do
echo $i/3750 >&2
i=$((i + 1))
python bandit.py --instance $inst --algorithm ucb-t2 --horizon 10000 --randomSeed $rs --epsilon 0.02 --scale $scle --threshold 0.75
done; done; done
