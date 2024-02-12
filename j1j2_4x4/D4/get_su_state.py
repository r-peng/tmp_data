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
class Ham(LocalHam2D):
    def _expm_cached(self,x,y):
        cache = self._op_cache["expm"]
        key = (id(x), y)
        if key not in cache:
            x = x.reshape((4,4))
    
            el, ev = do("linalg.eigh", x)
            x  = ev @ do("diag", do("exp", el * y)) @ dag(ev)
    
            x = x.reshape((2,)*4)
            cache[key] = x
        return cache[key]
ham = Ham(Lx,Ly,terms)

peps = load_tn_from_disc(f'tmpdir/su_{Lx},{Ly}_rand')
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

config = 1,0,1,0,\
         0,1,0,1,\
         1,0,1,0,\
         0,1,0,1
#peps = get_product_state(Lx,Ly,config)
#peps = PEPS.rand(Lx,Ly,D) 
su = SimpleUpdate(peps,ham,D=D,chi=200,compute_energy_every=20,compute_energy_final=True)
su.evolve(steps=200,tau=0.01,progbar=True)
print('energy per site=',su.energies[-1] / (Lx * Ly))
write_tn_to_disc(su.state,f'tmpdir/su_{Lx},{Ly}_rand',provided_filename=True)


