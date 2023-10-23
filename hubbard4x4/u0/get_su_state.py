import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc_ import (
    get_product_state,
    flat2site,
    ExchangeSampler,
    write_ftn_to_disc,load_ftn_from_disc,
    set_options,
)
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D 
from quimb.tensor.fermion.fermion_2d_tebd import SimpleUpdate 

import itertools
Lx,Ly = 4,4
nelec = 7,7
t = 1.
u = 0.
D = 4
data_map = set_options(symmetry='u1',flat=True,deterministic=False)
ham = Hubbard2D(t,u,Lx,Ly,symmetry='u1')
#exit()

load_fname = None
load_fname = 'su' 
if load_fname is None:
    config = 2, 1, 2, 1, \
             1, 0, 1, 2, \
             2, 1, 2, 0, \
             1, 2, 1, 2
    fpeps = get_product_state(Lx,Ly,config,symmetry='u1')
else:
    fpeps = load_ftn_from_disc(load_fname)


su = SimpleUpdate(fpeps,ham,D=D,compute_energy_final=False)
su.print_conv = False
su.evolve(steps=50,tau=0.01,progbar=True)
fpeps = su.state
write_ftn_to_disc(fpeps,'su',provided_filename=True)
#print(fpeps[0,0].data)
#exit()

#norm = fpeps.make_norm()
#norm = norm.contract()
#print('norm=',norm)

energy = fpeps.compute_local_expectation(ham.terms,max_bond=200,normalized=True) 
print('fpeps energy=',energy,energy/(Lx*Ly))

#ann_a = cre_a.dagger
#ann_b = cre_b.dagger
#na = np.tensordot(cre_a,ann_a,axes=([1],[0]))
#nb = np.tensordot(cre_b,ann_b,axes=([1],[0]))
#terms = {(i,j):na.copy() for i,j in itertools.product(range(Lx),range(Ly))}
#Na = fpeps.compute_local_expectation(terms,normalized=True) 
#terms = {(i,j):nb.copy() for i,j in itertools.product(range(Lx),range(Ly))}
#Nb = fpeps.compute_local_expectation(terms,normalized=True) 
#print('Na,Nb=',Na,Nb)
