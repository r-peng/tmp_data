import numpy as np
from autoray import do, to_numpy, dag
from quimb.tensor.tensor_2d_tebd import LocalHam2D,SimpleUpdate
from quimb.tensor.tensor_2d_vmc import get_product_state
from quimb.tensor.tensor_vmc import (
    write_tn_to_disc,load_tn_from_disc,
    get_gate1,get_gate2,
)
import itertools
Lx,Ly = 8,6 
D = 8
J1 = 1.
J2 = .5

h2 = get_gate2(1.,to_bk=False)
terms = dict()
for i in range(Lx):
    for j in range(Ly):
        if j<Ly-1:
            where = (i,j),(i,j+1)
            terms[where] = h2 * J1
        if i<Lx-1:
            where = (i,j),(i+1,j)
            terms[where] = h2 * J1
for i in range(Lx-1):
    for j in range(Ly-1):
        where = (i,j),(i+1,j+1)
        terms[where] = h2 * J2
        where = (i,j+1),(i+1,j)
        terms[where] = h2 * J2
ham = LocalHam2D(Lx,Ly,terms)

config = tuple([(i+j)%2 for i,j in itertools.product(range(Lx),range(Ly))])
#peps = get_product_state(Lx,Ly,config=config,bdim=D,eps=.1)
peps = load_tn_from_disc(f'suD{D}')
su = SimpleUpdate(peps,ham,D=D,compute_energy_final=False)
su.print_conv = False
su.evolve(steps=200,tau=0.01,progbar=True)
write_tn_to_disc(su.state,f'suD{D}',provided_filename=True)


