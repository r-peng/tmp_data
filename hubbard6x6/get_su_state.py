import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    set_options,
    get_product_state,
    flat2site,flatten,
)

from quimb.tensor.fermion.fermion_vmc import write_ftn_to_disc,load_ftn_from_disc
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D,SimpleUpdate 

import itertools
Lx,Ly = 6,6
nelec = 32
nelec = (nelec//2,) * 2
t = 1.
u = 8.
D = 8 
symmetry = 'u1'
set_options(symmetry=symmetry,flat=True)
ham = Hubbard2D(t,u,Lx,Ly,symmetry=symmetry)

hole = [(1,1),(4,1),(1,4),(4,4)]
config = []
for i,j in itertools.product(range(Lx),range(Ly)):
    if (i,j) in hole:
        ci = 0
    elif (i+j)%2==0:
        ci = 1
    else:
        ci = 2
    config.append(ci)
fname = f'suD{D}'
#load_fname = None
#load_fname = 'suD14' 
load_fname = fname
if load_fname is None:
    fpeps = get_product_state(Lx,Ly,config,symmetry=symmetry,flat=True)
else:
    fpeps = load_ftn_from_disc(load_fname)

su = SimpleUpdate(fpeps,ham,D=D,compute_energy_final=False)
su.print_conv = True
#su.print_conv = False 
su.evolve(steps=50,tau=0.001,progbar=True)
fpeps = su.state
write_ftn_to_disc(fpeps,fname,provided_filename=True)
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
print('eigvals=',w[:nelec[0]],w[nelec[0]:])
print('diagonalization energy=',sum(w[:nelec[0]]))
#print('diagonalization energy=',2 * sum(w[:nelec[0]])/(Lx * Ly))
#exit()

energy = fpeps.compute_local_expectation(ham.terms,normalized=True) 
print('fpeps energy=',energy)
