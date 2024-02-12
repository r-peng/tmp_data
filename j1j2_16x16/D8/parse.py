import numpy as np
from quimb.tensor.tensor_2d import PEPS
from quimb.tensor.tensor_2d_vmc import (
    write_tn_to_disc,load_tn_from_disc,
)

peps = dict()
out = open('D8', 'r').readlines()
for ix,l in enumerate(out):
    ls = l.split(' ')
    data = []
    for c in ls:
        if len(c)<3:
             continue
        data.append(float(c))
    #print(data)

    spin = ix % 2
    ix_ = ix // 2
    i,j = ix_ // 16, ix_ % 16
    print(i,j,spin,len(data))

    nleg = 4
    if i==0 or i==15:
        nleg -= 1
    if j==0 or j==15:
        nleg -= 1

    if (i,j) not in peps:
        peps[i,j] = np.zeros((8,)*nleg+(2,)) 
    peps[i,j][...,spin] = np.array(data).reshape((8,)*nleg)

arrays = []
for i in range(16):
    row = []
    for j in range(16):
        row.append(peps[i,j])
        print(i,j,row[-1])
    arrays.append(row)
peps = PEPS(arrays,shape='ldrup') 
print(peps)
write_tn_to_disc(peps,f'tmpdir/init',provided_filename=True)
