import numpy as np
from autoray import do, to_numpy, dag
from quimb.tensor.tensor_2d_tebd import LocalHam2D,SimpleUpdate 
from quimb.tensor.tensor_2d import PEPS 
from quimb.tensor.tensor_vmc import (
    write_tn_to_disc,load_tn_from_disc,
    get_gate1,get_gate2,
)
from quimb.tensor.tensor_2d_vmc import (
    get_product_state,
)
import itertools
Lx,Ly = 6,4
D = 4 
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

#config = [(i+j)%2 for i,j in itertools.product(range(Lx),range(Ly))]
#peps = get_product_state(Lx,Ly,config=config,bdim=D)
peps = load_tn_from_disc(f'suD{D}')
#peps = load_tn_from_disc(f'D3/peps/psi400')
su = SimpleUpdate(peps,ham,D=D,chi=8,compute_energy_every=50,compute_energy_final=True)
su.print_conv = False
su.evolve(steps=200,tau=0.01,progbar=True)
print('energy per site=',su.energies[-1] / (Lx * Ly))
write_tn_to_disc(su.state,f'suD{D}',provided_filename=True)


