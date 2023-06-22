import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc_ import (
    set_options,
    get_product_state,
    flat2site,flatten,
    write_ftn_to_disc,load_ftn_from_disc,
)
from quimb.tensor.fermion.fermion_2d_vmc_subspace import parse_config
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D,SimpleUpdate 

import itertools
Lx,Ly = 4,4
nelec = 7,7
t = 1.
u = 0.
D = 4 
subspace = 'a'
set_options(symmetry='u1',flat=True)
ham = Hubbard2D(t,u,Lx,Ly,symmetry='u1',subspace=subspace)

config = 2, 0, 1, 2, \
         0, 1, 2, 1, \
         1, 2, 1, 2, \
         2, 1, 2, 1
config_a,config_b,config_full = parse_config(config)
#load_fname = None
load_fname = f'./tmpdir/su_{Lx},{Ly}_D{D}_{subspace}'
if load_fname is None:
    if subspace=='a':
        fpeps = get_product_state(Lx,Ly,config_a,symmetry='u1',flat=True,subspace=subspace)
    elif subspace=='b':
        fpeps = get_product_state(Lx,Ly,config_b,symmetry='u1',flat=True,subspace=subspace)
    else:
        fpeps = get_product_state(Lx,Ly,config_full,symmetry='u1',flat=True,subspace=subspace)
else:
    fpeps = load_ftn_from_disc(load_fname)

#su = SimpleUpdate(fpeps,ham,D=D,compute_energy_final=False)
#su.print_conv = True
#su.print_conv = False 
#su.evolve(steps=50,tau=0.001,progbar=True)
#fpeps = su.state
#write_ftn_to_disc(su.state,f'./tmpdir/su_{Lx},{Ly}_D{D}_{subspace}',provided_filename=True)
#exit()

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
print('eigvals=',w[:nelec[0]],w[nelec[0]:])
print('diagonalization energy=',sum(w[:nelec[0]]))
print('per site energy=',sum(w[:nelec[0]])/(Lx*Ly)*2)
exit()

energy = fpeps.compute_local_expectation(ham.terms,normalized=True,max_bond=8) 
print('fpeps energy=',energy)
