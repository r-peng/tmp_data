import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    Hubbard2D,
    FermionExchangeSampler2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
    FermionDenseSampler,
)
from quimb.tensor.tensor_vmc import (
    SR,
    scale_wfn,
    write_tn_to_disc,load_tn_from_disc,
)

import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 3,3
nelec = 3,3
t = 1.
u = 8.
step = 0 
symmetry = 'z2'
model = Hubbard2D(t,u,Lx,Ly,symmetry=symmetry)
#print(model.batched_pairs)
#exit()

if step==0:
    fpeps = load_ftn_from_disc(f'../su_{symmetry}D4')
    if RANK==0:
        print(fpeps)
    for tid,tsr in fpeps.tensor_map.items():
        #print(tid,tsr.phase)
        tsr.phase = {}
else:
    fpeps = load_ftn_from_disc(f'psi{step}')
af = FermionAmplitudeFactory2D(fpeps,model,symmetry=symmetry)

sampler = FermionDenseSampler(Lx*Ly,nelec,exact=True) 
#sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = 2, 1, 0, \
         1, 2, 0, \
         0, 1, 2,

start = step 
stop = step + 50

tnvmc = SR(sampler,normalize=True,solve_full=True,solve_dense=False)
tnvmc.tmpdir = './' 
#tmpdir = None
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('nparam=',af.nparam)
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.check = 'energy'
tnvmc.batchsize = 5000
tnvmc.progbar = True
tnvmc.run(start,stop,save_wfn=True)
