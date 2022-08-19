import numpy as np
import argparse
from math import log, sqrt

debug = lambda x : print(x)

# checks if "state" is win for us
def isWin(state: str, opnt: str):
    ooo = opnt*3
    return state[:3] == ooo or state[3:6] == ooo or state[6:] == ooo or \
           state[::3] == ooo or state[1::3] == ooo or state[2::3] == ooo or \
           (state[0] == opnt and state[4] == opnt and state[8] == opnt) or \
           (state[2] == opnt and state[4] == opnt and state[6] == opnt)

def makeMove(state: str, pos: int, player: str):
    if(state[pos] == '0'):
        return state[:pos]+player+state[pos+1:]
    else: 
        return 0

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str)
    parser.add_argument('--states', type=str)
    args = parser.parse_args()

    f = open(args.states)
    stateMap = {}
    idx = 0
    for l in f.readlines():
        stateMap[l.strip()] = idx
        idx += 1
    f.close()
    
    policy = {}
    f = open(args.policy)
    lines = f.readlines()
    for l in lines[1:]:
        ls = l.split()
        policy[ls[0]] = list(map(float, ls[1:]))
    f.close()

    opponent = lines[0].strip()
    player   = '1' if opponent=='2' else '2'
    
    lose = len(stateMap)
    win  = len(stateMap)+1

    print("numStates", len(stateMap)+2)
    print("numActions", 9)
    print("end", lose, win)    # lose, win

    for state in stateMap:
        for action in range(9):
            nxtstate = makeMove(state, action, player)
            
            if(not nxtstate):
                # illegal move
                print("transition", stateMap[state], action, lose, -1, 1.0)
                continue

            if(nxtstate in policy):
                mvs = policy[nxtstate]
                for pos, pr_pos in enumerate(mvs): 
                    if(pr_pos == 0): continue
                    nnstate = makeMove(nxtstate, pos, opponent)
                    assert nnstate, f"Policy file contains illegal move : [{nxtstate}]  put {opponent} at {pos}"

                    if(nnstate in stateMap):
                        print("transition", stateMap[state], action, stateMap[nnstate], 0.0, pr_pos)
                    else:
                        # we must have won or drawn!
                        if(isWin(nnstate, opponent)):
                            print("transition", stateMap[state], action, win, 1.0, pr_pos)
                        else:
                            print("transition", stateMap[state], action, lose, 0.0, pr_pos)

            else:
                # we must have lost or drawn!
                print("transition", stateMap[state], action, lose, 0.0, 1.0)

    print("mdptype", 'episodic')
    print("discount", 1)

