import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    get_product_state,
    flat2site,
    set_options,
)
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D,SimpleUpdate
from quimb.tensor.fermion.fermion_vmc import write_ftn_to_disc,load_ftn_from_disc

import itertools
Lx,Ly = 4,4
nelec = 7,7
t = 1.
u = 4.
D = 4
set_options(symmetry='u1',flat=True)
ham = Hubbard2D(t,u,Lx,Ly,symmetry='u1')
#exit()

load_fname = None
load_fname = 'su'
if load_fname is None:
    config = 2, 0, 2, 1, \
             1, 2, 1, 2, \
             2, 1, 2, 1, \
             1, 2, 1, 0 
    fpeps = get_product_state(Lx,Ly,config,symmetry='u1')
else:
    fpeps = load_ftn_from_disc(load_fname)


su = SimpleUpdate(fpeps,ham,D=D,chi=16,compute_energy_every=20,compute_energy_final=True)
su.print_conv = False
su.evolve(steps=100,tau=0.002,progbar=True)
write_ftn_to_disc(su.state,f'su',provided_filename=True)
print('energy per site=',su.energies[-1]/16)
#print(fpeps[0,0].data)
exit()

#norm = fpeps.make_norm()
#norm = norm.contract()
#print('norm=',norm)

#energy = fpeps.compute_local_expectation(ham.terms,normalized=True) 
#print('fpeps energy=',energy)

#ann_a = cre_a.dagger
#ann_b = cre_b.dagger
#na = np.tensordot(cre_a,ann_a,axes=([1],[0]))
#nb = np.tensordot(cre_b,ann_b,axes=([1],[0]))
#terms = {(i,j):na.copy() for i,j in itertools.product(range(Lx),range(Ly))}
#Na = fpeps.compute_local_expectation(terms,normalized=True) 
#terms = {(i,j):nb.copy() for i,j in itertools.product(range(Lx),range(Ly))}
#Nb = fpeps.compute_local_expectation(terms,normalized=True) 
#print('Na,Nb=',Na,Nb)
