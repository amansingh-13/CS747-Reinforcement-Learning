import matplotlib.pyplot as plt
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1], header=None)
df1 = df.iloc[:3600]
df2 = df.iloc[3600:3600+3750]
m1 = df1.groupby([0,1,6])[7].mean()
df3 = df.iloc[3600+3750:3600+3750+600]
df4 = df.iloc[3600+3750+600:]

def q1f(i):
    y1 = list(m1['../instances/instances-task1/i-'+str(i)+'.txt'][' epsilon-greedy-t1'])
    y2 = list(m1['../instances/instances-task1/i-'+str(i)+'.txt'][' ucb-t1'])
    y3 = list(m1['../instances/instances-task1/i-'+str(i)+'.txt'][' kl-ucb-t1'])
    y4 = list(m1['../instances/instances-task1/i-'+str(i)+'.txt'][' thompson-sampling-t1'])
    x = list(m1['../instances/instances-task1/i-'+str(i)+'.txt'][' thompson-sampling-t1'].keys())
    plt.xscale('log')
    plt.plot(x, y1, label="Epsilon Greedy")
    plt.plot(x, y2, label="UCB")
    plt.plot(x, y3, label="KL-UCB")
    plt.plot(x, y4, label="Thompson Sampling")
    plt.legend()
    plt.xlabel('horizon (log scale)')
    plt.ylabel('avg regret')
    plt.show()

def q2f():
    y1 = list(df2.groupby([0,4])[7].mean()['../instances/instances-task2/i-1.txt'])
    y2 = list(df2.groupby([0,4])[7].mean()['../instances/instances-task2/i-2.txt'])
    y3 = list(df2.groupby([0,4])[7].mean()['../instances/instances-task2/i-3.txt'])
    y4 = list(df2.groupby([0,4])[7].mean()['../instances/instances-task2/i-4.txt'])
    y5 = list(df2.groupby([0,4])[7].mean()['../instances/instances-task2/i-5.txt'])
    x  = list(df2.groupby([0,4])[7].mean()['../instances/instances-task2/i-5.txt'].keys())
    plt.xscale('log')
    plt.plot(x, y1, label="Instance 1")
    plt.plot(x, y2, label="Instance 2")
    plt.plot(x, y3, label="Instance 3")
    plt.plot(x, y4, label="Instance 4")
    plt.plot(x, y5, label="Instance 5")
    plt.legend()
    plt.xlabel('scale')
    plt.ylabel('avg regret')
    plt.show()

def q3f(i):
    y = list(df3.groupby([0,6])[7].mean()['../instances/instances-task3/i-'+str(i)+'.txt'])
    x=list(df3.groupby([0,6])[7].mean()['../instances/instances-task3/i-'+str(i)+'.txt'].keys())
    plt.xscale('log')
    plt.plot(x, y, label="Instance "+str(i))
    plt.legend()
    plt.xlabel('horizon (log scale)')
    plt.ylabel('avg regret')
    plt.show()

def q4f(i,j):
    y = list(df4.groupby([0,5,6])[7].mean()['../instances/instances-task4/i-'+str(i)+'.txt'][j])
    x = list(df4.groupby([0,5,6])[7].mean()['../instances/instances-task4/i-'+str(i)+'.txt'][j].keys())
    print(x,y)
    plt.xscale('log')
    plt.plot(x, y, label="Instance "+str(i)+" threshold "+str(j))
    plt.legend()
    plt.xlabel('horizon (log scale)')
    plt.ylabel('avg regret')
    plt.show()


q1f(1)
q1f(2)
q1f(3)
q2f()
q3f(1)
q3f(2)
q4f(1,0.2)
q4f(2,0.2)
q4f(1,0.6)
q4f(2,0.6)
