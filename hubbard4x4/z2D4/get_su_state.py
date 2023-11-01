import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    get_random_state,get_data_map,
    flat2site,flatten,
    FermionAmplitudeFactory2D,
)

from quimb.tensor.fermion.fermion_vmc import write_ftn_to_disc,load_ftn_from_disc
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D,SimpleUpdate 
from quimb.tensor.tensor_vmc import scale_wfn

import itertools
Lx,Ly = 4,4
nsite = Lx * Ly
nelec = 7,7

t = 1.
u = 8.
mu = -1.5

D = 4 
chi = 8
symmetry = 'z2'
spinless = False

ham = Hubbard2D(t,u,Lx,Ly,mu=mu,symmetry=symmetry,spinless=spinless)

#load_fname = None
load_fname = 'su' 
if load_fname is None:
    fpeps = get_random_state(Lx,Ly,symmetry=symmetry,spinless=spinless,normalize=False)
else:
    fpeps = load_ftn_from_disc(load_fname)
    fpeps = scale_wfn(fpeps,scale=2.)

su = SimpleUpdate(fpeps,ham,D=D,compute_energy_final=False)
#su.print_conv = True
su.print_conv = False 
su.evolve(steps=100,tau=0.01,progbar=True)
fpeps = su.state
write_ftn_to_disc(fpeps,f'su',provided_filename=True)

expects = fpeps.compute_local_expectation(ham.terms,normalized=True,return_all=True,max_bond=chi) 
norm = [n for _,n in expects.values()]
print(norm)
e = sum([e/n for e,n in expects.values()])
print('fpeps energy=',e/(Lx*Ly))

data_map = get_data_map(symmetry=symmetry,spinless=spinless)
for spin in ('a','b'):
    cre = data_map[f'cre_{spin}']
    ann = data_map[f'ann_{spin}']
    pn = np.tensordot(cre,ann,axes=([1],[0]))
    pn.shape = 4,4
    terms = {(i,j):pn.copy() for i,j in itertools.product(range(Lx),range(Ly))}
    pn = fpeps.compute_local_expectation(terms,normalized=True,max_bond=chi) 
    print(f'N{spin}=',pn)
