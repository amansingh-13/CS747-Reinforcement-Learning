i=1
for inst in "../instances/instances-task1/i-1.txt" "../instances/instances-task1/i-2.txt" "../instances/instances-task1/i-3.txt"; do
for alg in "epsilon-greedy-t1" "ucb-t1" "thompson-sampling-t1"; do
for rs in {0..49}; do
echo $i/450 >&2
i=$((i + 1))
python bandit.py --instance $inst --algorithm $alg --horizon 100 400 1600 6400 25600 102400 --randomSeed $rs --epsilon 0.02 --scale 2 --threshold 0.75
done; done; done
