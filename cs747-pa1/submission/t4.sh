i=1
for inst in "../instances/instances-task4/i-1.txt" "../instances/instances-task4/i-2.txt"; do
for rs in {0..49}; do
for thresh in 0.2 0.6 ; do
echo $i/200 >&2
i=$((i + 1))
python bandit.py --instance $inst --algorithm alg-t4 --horizon 100 400 1600 6400 25600 102400 --scale 2 --randomSeed $rs --epsilon 0.02 --threshold $thresh
done; done; done;
