import h5py
import numpy as np
from numpy import *
import pandas as pd


f = h5py.File("cal_data.hdf5", "r")

n = input("n = ")

x = f[f"n={n}"]["x"][:]
y = f[f"n={n}"]["y"][:]
l = f[f"n={n}"]["l"][:]
lx = f[f"n={n}"]["l_x"][:]
ly = f[f"n={n}"]["l_y"][:]
angle = f[f"n={n}"]["angle"][:]

l = insert(l, 0, 0)
lx = insert(lx, 0, 0)
ly = insert(ly, 0, 0)
angle = insert(angle, 0, 0)

dft = {"x": x,
       "y": y,
       "l": l,
       "l_x": lx,
       "l_y": ly,
       "angle": angle}
df = pd.DataFrame(dft)

print("Seed is", f[f"n={n}"].attrs["seed"])
print("Scatting number is", f[f"n={n}"].attrs["n_scat"])
print("Total length is", f[f"n={n}"].attrs["Length"])
print("Surface is", f[f"n={n}"].attrs["Surface"])
print("Trace is", f[f"n={n}"].attrs["trace"])

print(df)
