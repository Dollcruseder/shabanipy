import h5py
from trajectory_parameter import find_trace
from math import pi

N_orbit = 147818
f1 = h5py.File("data_for_trace_cal.hdf5", "r")
f2 = h5py.File(f"trace_data_{N_orbit}.hdf5", "w")


step = 0.25 * pi

for i in [0, 1, 2, 4, 8]:
    theta = step * i
    T = []
    print("i = ", i)

    for n in range (0, N_orbit):
        l = f1[f"n={n}"]["l"][:]
        angle = f1[f"n={n}"]["angle"][:]
        T_a = find_trace(l, angle, theta)
        T.append(T_a)

    dset = f2.create_dataset(f"theta={i * 0.25}pi", (len(T),))
    dset[...] = T
