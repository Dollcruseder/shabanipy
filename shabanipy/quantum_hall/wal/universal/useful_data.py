import pandas as pd
import h5py
from trajectories import generate_trajectory, identify_trajectory
from trajectory_parameter import *

f2 = h5py.File("useful_data.hdf5", "w")
f1 = open('paper_data.txt','r')

dt = pd.read_csv(f1, delim_whitespace = True)

number = dt['n']
seed = dt['seed']
n_scat = dt['n_scat']
L = dt['L']
S = dt['S']
cosj = dt["cosj'"]

S_new = []
L_new = []
cosj_new = []

n_scat_max = 5000
d = 2.5e-5

n_scat_cal = np.empty(len(number), dtype=np.int)

k = 0
for i in range(0, len(number)):

    n_scat_cal[i] = identify_trajectory(seed[i], n_scat_max, d)
    if n_scat_cal[i] == n_scat[i]:
        S_new.append(S[i])
        L_new.append(L[i])
        cosj_new.append(cosj[i])
        k += 1

print(k)

dset1 = f2.create_dataset("Surface", (len(S_new),))
dset1[...] = S_new

dset2 = f2.create_dataset("Length", (len(L_new),))
dset2[...] = L_new

dset3 = f2.create_dataset("cosj'", (len(cosj_new),))
dset3[...] = cosj_new
