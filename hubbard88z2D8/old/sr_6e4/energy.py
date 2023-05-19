import numpy as np
import h5py
data = []
apps = ['','1','2','3','4']
for app in apps:
    out = open(f'out{app}.out', 'r').readlines()
    for l in out:
        if l[:5] == "step=":
            ls = l.split(',')
            data.append(float(ls[1].split('=')[-1]))
data = np.array(data)
print(len(data))
f = h5py.File('energy.hdf5','w')
f.create_dataset('data',data=data)
f.close()

data = []
for app in apps:
    out = open(f'out{app}.out', 'r').readlines()
    for l in out:
        if l[:5] == "step=":
            ls = l.split(',')
            data.append(float(ls[3].split('=')[-1]))
data = np.array(data)
f = h5py.File('err.hdf5','w')
f.create_dataset('data',data=data)
f.close()

data = []
for app in apps:
    out = open(f'out{app}.out', 'r').readlines()
    for ix,l in enumerate(out):
        if l[:5] == "step=":
            ls = l.split(',')
            data.append(float(ls[4].split('=')[-1]))
data = np.array(data)
f = h5py.File('gradient.hdf5','w')
f.create_dataset('data',data=data)
f.close()
