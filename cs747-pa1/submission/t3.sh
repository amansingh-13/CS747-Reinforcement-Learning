i=1
for inst in "../instances/instances-task3/i-1.txt" "../instances/instances-task3/i-2.txt"; do
for rs in {0..49}; do
echo $i/100 >&2
i=$((i + 1))
python bandit.py --instance $inst --algorithm alg-t3 --horizon 100 400 1600 6400 25600 102400 --scale 2 --randomSeed $rs --epsilon 0.02 --threshold 0.75
done; done;
