import numpy as np
import argparse
import pulp
from math import log, sqrt

debug = lambda x : print(x)

class MDP:
    def __init__(self, txt):
        self.nstates  = int(txt[0].split()[1])
        self.nactions = int(txt[1].split()[1])
        self.end = list(map(int, txt[2].split()[1:]))
        self.t = [[{} for _ in range(self.nactions)] for _ in range(self.nstates)]
        self.r = [[{} for _ in range(self.nactions)] for _ in range(self.nstates)]

        idx = 3
        while(txt[idx][:10] == "transition"):
            s1, ac, s2 = list(map(int, txt[idx].split()[1:4]))
            r, p = list(map(float, txt[idx].split()[4:]))
            self.t[s1][ac][s2] = p
            self.r[s1][ac][s2] = r
            idx += 1

        self.type = txt[-2].split()[1]
        self.gamma = float(txt[-1].split()[1])

    def T(self, s1, ac, s2):
        if s2 in self.t[s1][ac]:
            return self.t[s1][ac][s2]
        else: return 0
    def R(self, s1, ac, s2):
        if s2 in self.r[s1][ac]:
            return self.r[s1][ac][s2]
        else: return 0
    
    def V(self, pi):
        n = self.nstates
        M = np.zeros((n,n))
        b = np.zeros(n)
        for i in range(self.nstates):
            for j in self.t[i][pi[i]].keys():
                M[i][j] = -self.gamma*self.T(i, pi[i], j)
        for i in range(self.nstates):
            M[i][i] += 1
        for s in range(self.nstates):
            for s2 in self.r[s][pi[s]].keys():
                b[s] += self.t[s][pi[s]][s2] * self.r[s][pi[s]][s2]
        return np.linalg.solve(M, b)

    def Q(self, V, s, a):
        ret = 0
        for s2 in self.r[s][a].keys():
            ret += self.t[s][a][s2] * (self.r[s][a][s2] + self.gamma * V[s2])
        return ret

    def pi(self, V, s):
        argmax, maxval = 0, float('-inf')
        for a in range(self.nactions):
            val = self.Q(V, s, a)
            if(val > maxval):
                maxval, argmax = val, a
        return maxval, argmax

    def print_v(self, v):
        for s in range(self.nstates):
            print(v[s], self.pi(v, s)[1])

def vi(mdp, epsilon):
    v = np.zeros(mdp.nstates)
    oldv = np.zeros(mdp.nstates)
    while(True):
        oldv = np.copy(v)
        for s in range(mdp.nstates):
            v[s], _ = mdp.pi(oldv, s)
        if(np.max(np.abs(v-oldv)) < epsilon):
            break
    mdp.print_v(v)

def lp(mdp):
    problem = pulp.LpProblem('mdp-planning', pulp.LpMinimize)
    v = [0]*mdp.nstates
    dv = []
    objective = ""

    for s in range(mdp.nstates):
        dv.append(pulp.LpVariable('V'+str(s)))
        objective += dv[-1]
    problem += objective
    for s in range(mdp.nstates):
        if s in mdp.end:
            problem += (dv[s] == 0)
            continue
        for a in range(mdp.nactions):
            rhs = ""
            for s2 in mdp.t[s][a].keys():
                rhs += mdp.T(s,a,s2)*(mdp.R(s,a,s2) + mdp.gamma*dv[s2])
            problem += (dv[s] >= rhs)
    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    
    for dvi in problem.variables():
        v[int(dvi.name[1:])] = dvi.varValue
    mdp.print_v(v)

def hpi(mdp, epsilon):
    pi = [0]*mdp.nstates
    v  = mdp.V(pi)
    while(True):
        is_exist = False
        for s in range(mdp.nstates):
            for a in range(mdp.nactions):
                if(mdp.Q(v, s, a)-v[s] > epsilon):
                    pi[s] = a
                    is_exist = True
                    break
        if(not is_exist): break
        v = mdp.V(pi)
    mdp.print_v(v)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', type=str)
    parser.add_argument('--algorithm', type=str, default='hpi')
    args = parser.parse_args()

    f = open(args.mdp)
    txt = f.readlines()
    f.close()
    
    mdp = MDP(txt)

    if(args.algorithm == 'vi'):
        vi(mdp, 1e-10)
    elif(args.algorithm == 'lp'):
        lp(mdp)
    elif(args.algorithm == 'hpi'):
        hpi(mdp, 1e-10)
    else:
        print("pass correct --algorithm")

