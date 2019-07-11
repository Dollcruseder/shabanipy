import pandas as pd
from shabanipy.quantum_hall.wal.universal.trajectories import identify_trajectory
import numpy as np

f = open('paper_data.txt','r')

dt = pd.read_csv(f, delim_whitespace = True)

number = dt['n']
seed = dt['seed']
n_scat_paper = dt['n_scat']
L = dt['L']
S = dt['S']
cosj = dt["cosj'"]

n_scat_max = 1000000
d = 2.5e-5

n_scat_cal = np.empty(len(number), dtype=np.int)
check = []

a = 0
b = 0
for i in range(0, len(number)):
#for i in [4653, 6764, 11026]:
    n_scat_cal[i] = identify_trajectory(seed[i], n_scat_max, d)
    if n_scat_cal[i] == n_scat_paper[i]:
        check.append("True")
    else:
        check.append("False")
        a += 1
        if n_scat_cal[i] != 1000001:
            print(f"Term {i+1} is wrong.")
            print("cal",n_scat_cal[i],"paper",n_scat_paper[i])
            b += 1

print(a)
print(b)



#dft = {'n_scat_cal': n_scat_cal,
      #'n_scat_paper': n_scat_paper,
      #'check_result': check}

#df = pd.DataFrame(dft)

#print(df)

#if np.sum(n_scat_cal - n_scat_paper) == 0:
    #print("All the number are right.")
