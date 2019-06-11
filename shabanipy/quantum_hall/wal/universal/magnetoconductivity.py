import h5py
from math import *

f = h5py.File("useful_data.hdf5","r")

B = 0
L_phi = 10
Sum = 0

S = f["Surface"][:]
L = f["Length"][:]
cosj = f["cosj'"][:]
T = f["Trace"][:]


for i in range(0, len(S)):

    xj = exp(- L[i] / L_phi) * 0.5 * T[i].real * (1 + cosj[i])
    a = xj * cos(B * S[i])
    Sum = Sum + a


print(Sum / len(S))
