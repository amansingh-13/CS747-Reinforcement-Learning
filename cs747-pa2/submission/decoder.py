import numpy as np
import argparse
from math import log, sqrt

debug = lambda x : print(x)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--value-policy', type=str)
    parser.add_argument('--states', type=str)
    parser.add_argument('--player-id', type=str)
    args = parser.parse_args()

    f = open(args.states)
    states = list(map(lambda x : x.strip(), f.readlines()))
    f.close()
    
    print(args.player_id)

    f = open(args.value_policy)
    for idx,l in enumerate(f.readlines()):
        nxt = int(l.split()[1])
        arr = ['0']*9
        arr[nxt] = '1'
        if(idx < len(states)):
            print(states[idx], " ".join(arr))
    
