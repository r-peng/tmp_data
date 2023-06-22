import numpy as np
import h5py
data = []
out = open('out.out', 'r').readlines()
for l in out:
    if l[:5] == "step=":
        ls = l.split(',')
        data.append(float(ls[1].split('=')[-1]))
out = open('out1.out', 'r').readlines()
for l in out:
    if l[:5] == "step=":
        ls = l.split(',')
        data.append(float(ls[1].split('=')[-1]))
data = np.array(data)
f = h5py.File('energy.hdf5','w')
f.create_dataset('data',data=data)
f.close()

data = []
out = open('out.out', 'r').readlines()
for l in out:
    if l[:5] == "step=":
        ls = l.split(',')
        data.append(float(ls[3].split('=')[-1]))
out = open('out1.out', 'r').readlines()
for l in out:
    if l[:5] == "step=":
        ls = l.split(',')
        data.append(float(ls[3].split('=')[-1]))
data = np.array(data)
f = h5py.File('err.hdf5','w')
f.create_dataset('data',data=data)
f.close()

data = []
out = open('out.out', 'r').readlines()
for ix,l in enumerate(out):
    ls = l.split(',')
    if len(ls)==4:
        ls = ls[0].split('=')
        print(ls)
        try:
            data.append(float(ls[1]))
        except ValueError:
            pass
out = open('out1.out', 'r').readlines()
for ix,l in enumerate(out):
    ls = l.split(',')
    if len(ls)==4:
        ls = ls[0].split('=')
        try:
            data.append(float(ls[1]))
        except ValueError:
            pass
data = np.array(data)[1::2]
print(data)
f = h5py.File('gradient.hdf5','w')
f.create_dataset('data',data=data)
f.close()
