import subprocess, os, time
import numpy as np

P1_STATE_FILE = "./data/attt/states/states_file_p1.txt"
P2_STATE_FILE = "./data/attt/states/states_file_p2.txt"
P1_INIT_STRAT = ["./data/attt/policies/p1_policy1.txt","./data/attt/policies/p1_policy2.txt"]
P2_INIT_STRAT = ["./data/attt/policies/p2_policy1.txt","./data/attt/policies/p2_policy2.txt"]

def closeness(pol1, pol2):
    count = 0
    zipped = list(zip(pol1.split("\n"), pol2.split("\n")))
    for l1, l2 in zipped[1:]:
        if(l1 == l2): count += 1
    return str(count)+"/"+str(len(zipped)-1)

def improve(policy:str):
    player = '1' if policy[0]=='2' else '2'
    states = P1_STATE_FILE if player=='1' else P2_STATE_FILE

    f = open('.policy_opnt','w')
    f.write(policy)
    f.close()

    cmd_encoder = "python","encoder.py","--policy",".policy_opnt","--states",states
    f = open('.mdp_file','w')
    subprocess.call(cmd_encoder, stdout=f)
    f.close()

    cmd_planner = "python","planner.py","--mdp",".mdp_file"
    f = open('.policy_encoded','w')
    subprocess.call(cmd_planner, stdout=f)
    f.close()

    cmd_decoder = "python","decoder.py","--value-policy",".policy_encoded","--states",states,"--player-id",player
    cmd_output = subprocess.check_output(cmd_decoder, universal_newlines=True)

    os.remove('.policy_opnt')
    os.remove('.mdp_file')
    os.remove('.policy_encoded')
    return cmd_output

if __name__ == '__main__':
    np.random.seed(int(time.time()))
    
    f = open(np.random.choice(P1_INIT_STRAT))
    p1 = f.read()
    f.close()

    f = open(np.random.choice(P2_INIT_STRAT))
    p2 = f.read()
    f.close()

    for _ in range(10):
        newp1 = improve(p2)
        newp2 = improve(newp1)
        print(f"No. of same actions in updated, previous P1 policy = {closeness(newp1,p1)}", \
              f"No. of same actions in updated, previous P2 policy = {closeness(newp2,p2)}", sep='\t| ')
        p1 = newp1
        p2 = newp2
