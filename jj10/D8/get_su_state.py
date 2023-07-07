import numpy as np
from quimb.tensor.tensor_2d_tebd import LocalHam2D,SimpleUpdate 
from quimb.tensor.tensor_2d import PEPS 
from quimb.tensor.tensor_2d_vmc_ import (
    write_tn_to_disc,load_tn_from_disc,
    get_product_state,
    get_gate1,get_gate2,
)

import itertools
Lx,Ly = 10,10
D = 8
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

#peps = load_tn_from_disc(f'tmpdir/su_{Lx},{Ly}_rand')
#peps = load_tn_from_disc(f'sr/psi1')
#E = peps.compute_local_expectation(ham.terms,normalized=True)
#print('energy=',E)
#print('energy per site=',E / (Lx * Ly))
#exit()
#
#h1 = get_gate1()
#terms = {(i,j):h1.copy() for i in range(Lx) for j in range(Ly)}
#terms = peps.compute_local_expectation(terms,normalized=True,return_all=True) 
#Sz = 0.
#for key,(s,n) in terms.items():
#    print(key,s/n)
#    Sz += s/n
#print('Sz=',Sz)
#exit()

config = []
for i in range(Lx):
    for j in range(Ly):
        ci = 0 if (i+j)%2==0 else 1
        config.append(ci)
#peps = get_product_state(Lx,Ly,config=tuple(config),bdim=D,eps=.01)
peps = load_tn_from_disc('su')
su = SimpleUpdate(peps,ham,D=D,compute_energy_final=False)
su.print_conv = True
su.evolve(steps=50,tau=0.001,progbar=True)
#print('energy per site=',su.energies[-1] / (Lx * Ly))
write_tn_to_disc(su.state,'su',provided_filename=True)


