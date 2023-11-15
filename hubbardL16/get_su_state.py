import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    get_product_state,
    flat2site,flatten,
    Hubbard,
)

from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,
    load_ftn_from_disc,
    get_data_map,
)
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D,SimpleUpdate 
from quimb.tensor.tensor_vmc import scale_wfn

import itertools
Lx,Ly = 1,16
nelec = 7,7 
t = 1.
u = 8.
D = 10 
symmetry = 'z2'
spinless = False 
mu = -1.5
ham = Hubbard2D(t,u,Lx,Ly,mu=mu,symmetry=symmetry,spinless=spinless)

#load_fname = None
load_fname = 'su' 
if load_fname is None:
    config = 1,2,1,2,\
             0,1,2,1,\
             1,2,0,2,\
             2,1,2,1,
    fpeps = get_product_state(Lx,Ly,config,symmetry=symmetry,flat=True,spinless=spinless)
else:
    fpeps = load_ftn_from_disc(load_fname)

su = SimpleUpdate(fpeps,ham,D=D,compute_energy_final=True)
#su.print_conv = True
su.print_conv = False 
su.evolve(steps=200,tau=0.01,progbar=True)
fpeps = su.state
write_ftn_to_disc(fpeps,f'su',provided_filename=True)
#exit()

#fpeps = scale_wfn(fpeps,1.5)
data_map = get_data_map(symmetry=symmetry,spinless=spinless)
if spinless:
    cre = data_map['cre']
    ann = data_map['ann']
    pn = np.tensordot(cre,ann,axes=([1],[0]))
else:
    pn = 0
    for spin in ('a','b'):
        cre = data_map[f'cre_{spin}']
        ann = data_map[f'ann_{spin}']
        pn = pn + np.tensordot(cre,ann,axes=([1],[0]))
    pn.shape = 4,4
terms = {(i,j):pn.copy() for i,j in itertools.product(range(Lx),range(Ly))}
expects = fpeps.compute_local_expectation(terms,max_bond=16,normalized=True,return_all=True)
print('norm=',[n for _,n in expects.values()])
print('N=',sum([e/n for e,n in expects.values()]))
write_ftn_to_disc(fpeps,f'su',provided_filename=True)
print(fpeps)
exit()

nsite = Lx * Ly
h1 = np.zeros((nsite,)*2)
for i,j in itertools.product(range(Lx),range(Ly)):
    ix1 = flatten(i,j,Ly)
    if i+1<Lx:
        ix2 = flatten(i+1,j,Ly)
        h1[ix1,ix2] = -t
        h1[ix2,ix1] = -t
    if j+1<Ly:
        ix2 = flatten(i,j+1,Ly)
        h1[ix1,ix2] = -t
        h1[ix2,ix1] = -t
print('check symmetric=',np.linalg.norm(h1-h1.T))
w,v = np.linalg.eigh(h1)
print('eigvals=',w[:nelec],w[nelec:])
print('diagonalization energy=',sum(w[:nelec]))
#print('diagonalization energy=',2 * sum(w[:nelec[0]])/(Lx * Ly))
exit()

energy = fpeps.compute_local_expectation(ham.terms,normalized=True) 
print('fpeps energy=',energy)
