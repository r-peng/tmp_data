import numpy as np
from autoray import do, to_numpy, dag
from quimb.tensor.tensor_2d_tebd import LocalHam2D
from quimb.tensor.tensor_2d_tebd_pbc import SimpleUpdate 
from quimb.tensor.tensor_2d_vmc import (
    write_tn_to_disc,load_tn_from_disc,
    get_product_state,
    peps2pbc,
    get_gate1,get_gate2,
)

import itertools
Lx,Ly = 4,4
D = 2
J = 1.
h = 0.

h2 = get_gate2(J,to_bk=True)
print(np.linalg.norm(h2-h2.transpose(1,0,3,2)))
print(np.linalg.norm(h2-h2.transpose(2,3,0,1)))
terms = dict()
for i in range(Lx):
    for j in range(Ly):
        if j<Ly-1:
            where = (i,j),(i,j+1)
        else:
            where = (i,0),(i,j)
        terms[where] = h2

        if i<Lx-1:
            where = (i,j),(i+1,j)
        else:
            where = (0,j),(i,j)
        terms[where] = h2
ham = LocalHam2D(Lx,Ly,terms)

config = 1,0,1,0,\
         0,1,0,1,\
         1,0,1,0,\
         0,1,0,1
peps = get_product_state(Lx,Ly,config,bdim=D,eps=.1)
peps = peps2pbc(peps)
su = SimpleUpdate(peps,ham,D=D,compute_energy_final=False)
su.evolve(steps=100,tau=0.1,progbar=True)
write_tn_to_disc(su.state,f'tmpdir/su_{Lx},{Ly}_rand',provided_filename=True)


