import numpy as np
#from quimb.tensor.fermion.product_2d_vmc import (
#    ProductAmplitudeFactory2D,
#    PEPSJastrow,
#)
from quimb.tensor.fermion.fermion_2d_vmc import (
    set_options,
    Hubbard,
    FermionExchangeSampler2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
    FermionDenseSampler,
)
from quimb.tensor.tensor_vmc import (
    TNVMC,
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
set_options(symmetry='u1',flat=True,deterministic=False)
model = Hubbard(t,u,Lx,Ly)
#print(model.batched_pairs)
#exit()

if step==0:
    fpeps = load_ftn_from_disc(f'../su')
    if RANK==0:
        print(fpeps)
    for tid,tsr in fpeps.tensor_map.items():
        #print(tid,tsr.phase)
        tsr.phase = {}
else:
    fpeps = load_ftn_from_disc(f'psi{step}')
af = FermionAmplitudeFactory2D(fpeps)
af.model = model 
af.is_tn = True

sampler = FermionDenseSampler(Lx*Ly,nelec,exact=True) 
#sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = 2, 1, 0, \
         1, 2, 0, \
         0, 1, 2,

start = step 
stop = step + 10

tnvmc = TNVMC(sampler,normalize=True,optimizer='sr',pure_newton=True,solve_full=True,solve_dense=False)
tnvmc.tmpdir = './' 
#tmpdir = None
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('nparam=',af.nparam)
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.check = 'energy'
tnvmc.batchsize = 1000
tnvmc.run(start,stop,save_wfn=False)
