import h5py
from trajectories import generate_trajectory
from trajectory_parameter import find_each_length, find_each_angle

"""



"""

f1 = h5py.File(f"data_for_trace_cal.hdf5", "w")
f2 = h5py.File("paper_data.hdf5", "r")

number = f2["n"][:]
seed = f2["seed"][:]
n_scat = f2["n_scat"][:]


for i in range(0, len(number)):

    g1 = f1.create_group(f"n={i + 1}")

    tj = generate_trajectory(seed[i], n_scat[i])
    x = tj[0]
    y = tj[1]
    x[-1] = 0
    y[-1] = 0

    l = find_each_length(x, y)
    angle = find_each_angle(x, y)

    dset = g1.create_dataset("l", (len(l),))
    dset[...] = l

    dset = g1.create_dataset("angle", (len(angle),))
    dset[...] = angle
