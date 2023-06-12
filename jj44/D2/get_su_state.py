import numpy as np
from autoray import do, to_numpy, dag
from quimb.tensor.tensor_2d_tebd import LocalHam2D,SimpleUpdate 
from quimb.tensor.tensor_2d import PEPS 
from quimb.tensor.tensor_2d_vmc import (
    write_tn_to_disc,load_tn_from_disc,
    get_product_state,
    get_gate1,get_gate2,
)

import itertools
Lx,Ly = 4,4
D = 2
J1 = 1.
J2 = 0.5

h2 = get_gate2(1.,to_bk=True)
terms = dict()
for i in range(Lx):
    for j in range(Ly):
        if i+1<Lx:
            where = (i,j),(i+1,j)
            terms[where] = h2 * J1 
        if j+1<Ly:
            where = (i,j),(i,j+1)
            terms[where] = h2 * J1 
for i in range(Lx-1):
    for j in range(Ly-1):
        where = (i,j),(i+1,j+1)
        terms[where] = h2 * J2
        where = (i,j+1),(i+1,j)
        terms[where] = h2 * J2
ham = LocalHam2D(Lx,Ly,terms)

config = 1,0,1,0,\
         0,1,0,1,\
         1,0,1,0,\
         0,1,0,1
peps = get_product_state(Lx,Ly,config,bdim=D,eps=0.1)

#peps = load_tn_from_disc(f'tmpdir/su_{Lx},{Ly}_rand')

su = SimpleUpdate(peps,ham,D=D,chi=200,compute_energy_every=20,compute_energy_final=True)
su.print_conv = False
su.evolve(steps=200,tau=0.01,progbar=True)
print('energy per site=',su.energies[-1] / (Lx * Ly))
write_tn_to_disc(su.state,f'tmpdir/su_{Lx},{Ly}_rand',provided_filename=True)


