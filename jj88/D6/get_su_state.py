import numpy as np
from autoray import do, to_numpy, dag
from quimb.tensor.tensor_2d_tebd import LocalHam2D
from quimb.tensor.tensor_2d_tebd import SimpleUpdate 
from quimb.tensor.tensor_2d_vmc import (
    write_tn_to_disc,load_tn_from_disc,
    get_product_state,
    get_gate1,get_gate2,
    flatten,flat2site,
)

import itertools
Lx,Ly = 8,8
D = 6
J1 = 1.
J2 = 0.5

h2 = get_gate2(1.,to_bk=True)
print(np.linalg.norm(h2-h2.transpose(1,0,3,2)))
print(np.linalg.norm(h2-h2.transpose(2,3,0,1)))
terms = dict()
for i in range(Lx):
    for j in range(Ly):
        if j<Ly-1:
            where = (i,j),(i,j+1)
            terms[where] = h2

        if i<Lx-1:
            where = (i,j),(i+1,j)
            terms[where] = h2 * J1
terms_ = dict()
for i in range(Lx):
    for j in range(Ly):
        if i+1<Lx and j+1<Ly:
            where = (i,j),(i+1,j+1)
            terms_[where] = h2 * J2
            where = (i,j+1),(i+1,j)
            terms_[where] = h2 * J2
terms.update(terms_)
ham = LocalHam2D(Lx,Ly,terms)

config = 1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,
#peps = get_product_state(Lx,Ly,config,bdim=D,eps=.1)
#peps = peps2pbc(peps)
peps = load_tn_from_disc(f'tmpdir/su_{Lx},{Ly}_rand')
su = SimpleUpdate(peps,ham,D=D,compute_energy_final=False)
su.print_conv = True
su.evolve(steps=50,tau=0.001,progbar=True)
write_tn_to_disc(su.state,f'tmpdir/su_{Lx},{Ly}_rand',provided_filename=True)


