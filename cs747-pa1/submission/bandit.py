import numpy as np
import matplotlib.pyplot as plt
import argparse
from math import log

class Bandit:
    def __init__(self, probabs, reward):
        self.probabs = probabs
        self.reward = reward
        self.total_reward = 0
        self.total_pulls  = 0
    def pull(self, i):
        x = np.random.choice(self.reward, p=self.probabs[i])
        self.total_reward += x
        self.total_pulls  += 1
        return x
    def regret(self, args):
        max_mean = -1
        for pdf in self.probabs:
           mean = np.sum(pdf*self.reward)
           if(mean > max_mean): max_mean = mean
        regret = max_mean*self.total_pulls - self.total_reward
        if(self.total_pulls in args.horizon):
          print(args.instance,args.algorithm,args.randomSeed,args.epsilon,args.scale,
                args.threshold,self.total_pulls,regret,self.total_reward,sep=", ")

def make_bandit(txt, task, threshold = 0.75):
    if(task == 1 or task == 2):
        probabs = []
        for l in txt:
            probabs.append(np.array([1-float(l), float(l)]))
        return Bandit(np.array(probabs), np.array([0.0, 1.0]))
    elif(task == 3):
        reward = np.array(txt[0].split(), dtype=float)
        probabs = list(map(lambda x : np.array(x.split(),dtype=float), txt[1:]))
        return Bandit(np.array(probabs), reward)
    elif(task == 4):
        reward = np.array(txt[0].split(), dtype=float)
        probabs = []
        for l in txt[1:]:
            high_p = np.sum([float(x) for i,x in enumerate(l.split()) if reward[i]>threshold])
            probabs.append(np.array([1-high_p, high_p]))
        return Bandit(np.array(probabs), np.array([0.0, 1.0]))

def e_greedy_a(bandit, epsilon, horizon, args):
    no_arms  = len(bandit.probabs)
    emp_mean = np.zeros(no_arms)
    no_pulls = np.zeros(no_arms)
    
    # At beginning, pull all arms once. Break ties in
    # empirical means by taking smallest index arm
    for i in range(no_arms):
        emp_mean[i] = bandit.pull(i)
        no_pulls[i] += 1
        bandit.regret(args)
    for _ in range(horizon[-1]-no_arms):
        #print(no_pulls)
        a2pull = -1
        if(np.random.random() < epsilon): 
            a2pull = np.random.randint(0, no_arms)
        else:
            a2pull = np.argmax(emp_mean)
        emp_mean[a2pull] = \
            (emp_mean[a2pull]*no_pulls[a2pull]+bandit.pull(a2pull))/(no_pulls[a2pull]+1)
        no_pulls[a2pull] += 1
        bandit.regret(args)

def ucb_a(bandit, scale, horizon, args):
    no_arms  = len(bandit.probabs)
    arm_reward = np.zeros(no_arms)
    no_pulls = np.zeros(no_arms)
    no_total_pulls = 0
    ucb = np.zeros(no_arms)

    # At beginning, pull all arms once. Break ties
    # in ucb by taking smallest index arm
    for i in range(no_arms):
        arm_reward[i] += bandit.pull(i)
        no_pulls[i] += 1
        no_total_pulls += 1
        bandit.regret(args)
    ucb = arm_reward/no_pulls + np.sqrt(scale*log(no_total_pulls)/no_pulls)

    for _ in range(horizon[-1]-no_arms):
        #print(arm_reward/no_pulls)
        #print(no_pulls)
        a2pull = np.argmax(ucb)
        arm_reward[a2pull] += bandit.pull(a2pull)
        no_pulls[a2pull] += 1
        no_total_pulls += 1
        
        ucb = arm_reward/no_pulls + np.sqrt(scale*log(no_total_pulls)/no_pulls)
        bandit.regret(args)

def get_klucb(u, p, t, c = 3):
    kl = lambda q : (p*log(p/q) if p else 0) + ((1-p)*log((1-p)/(1-q)) if 1-p else 0)
    f = lambda x : u*kl(x) - log(t) - c*log(log(t))
    x, y = p, 1
    while(y-x > 0.00001):
        m = (x+y)/2
        if(f(m) > 0): y = m
        else: x = m
    return (x+y)/2

def kl_ucb_a(bandit, horizon, args): 
    no_arms  = len(bandit.probabs)
    arm_reward = [0.0]*no_arms
    no_pulls = [0.0]*no_arms
    no_total_pulls = 0.0
    klucb = [0.0]*no_arms

    # At beginning, pull all arms once. Break ties in klucb
    # by taking smallest index arm, precision=0.00001, c=3
    for i in range(no_arms):
        arm_reward[i] += float(bandit.pull(i))
        no_pulls[i] += 1
        no_total_pulls += 1
        bandit.regret(args)
    for i in range(no_arms):
        klucb[i] = get_klucb(no_pulls[i], arm_reward[i]/no_pulls[i], no_total_pulls)
    
    for _ in range(horizon[-1]-no_arms):
        #print(arm_reward/no_pulls)
        #print(klucb)
        a2pull = np.argmax(klucb)
        arm_reward[a2pull] += float(bandit.pull(a2pull))
        no_pulls[a2pull] += 1
        no_total_pulls += 1
        
        for i in range(no_arms):
            klucb[i] = get_klucb(no_pulls[i], arm_reward[i]/no_pulls[i], no_total_pulls)
        bandit.regret(args)

def thompson_a(bandit, horizon, args):
    no_arms  = len(bandit.probabs)
    arm_reward = np.zeros(no_arms)
    arm_fail = np.zeros(no_arms)

    #  Break ties by taking smallest index arm
    for i in range(horizon[-1]):
        a2pull = np.argmax(np.random.beta(arm_reward+1, arm_fail+1))
        res = bandit.pull(a2pull)
        arm_reward[a2pull] += res
        arm_fail[a2pull]   += 1-res
        bandit.regret(args)

def thompson_t3_a(bandit, horizon, args):
    no_arms  = len(bandit.probabs)
    arm_reward = np.zeros(no_arms)
    arm_fail = np.zeros(no_arms)

    #  Break ties by taking smallest index arm
    for i in range(horizon[-1]):
        #print(arm_fail+arm_reward)
        a2pull = np.argmax(np.random.beta(arm_reward+1, arm_fail+1))
        res = bandit.pull(a2pull)
        arm_reward[a2pull] += res
        arm_fail[a2pull]   += 1-res
        bandit.regret(args)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str)
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--randomSeed', type=int)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--scale', type=float)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--horizon', nargs='+', type=int)
    args = parser.parse_args()

    f = open(args.instance)
    txt = f.readlines()
    f.close()
    np.random.seed(args.randomSeed)
    
    ################# send numpy arrays only after processing file ################

    if(args.algorithm == "epsilon-greedy-t1"):
        b = make_bandit(txt, 1)
        e_greedy_a(b, args.epsilon, args.horizon, args)
    elif(args.algorithm == "ucb-t1"):
        b = make_bandit(txt, 1)
        ucb_a(b, args.scale, args.horizon, args)
    elif(args.algorithm == "kl-ucb-t1"):
        b = make_bandit(txt, 1)
        kl_ucb_a(b, args.horizon, args)
    elif(args.algorithm == "thompson-sampling-t1"):
        b = make_bandit(txt, 1)
        thompson_a(b, args.horizon, args)
    elif(args.algorithm == "ucb-t2"):
        b = make_bandit(txt, 2)
        ucb_a(b, args.scale, args.horizon, args)
    elif(args.algorithm == "alg-t3"):
        b = make_bandit(txt, 3)
        thompson_t3_a(b, args.horizon, args)
    elif(args.algorithm == "alg-t4"):
        b = make_bandit(txt, 4, args.threshold)
        thompson_a(b, args.horizon, args)
    else:
        print("incorrect --algorithm passed")

