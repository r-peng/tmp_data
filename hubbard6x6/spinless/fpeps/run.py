import numpy as np
#from quimb.tensor.fermion.product_2d_vmc import (
#    ProductAmplitudeFactory2D,
#    PEPSJastrow,
#)
from quimb.tensor.fermion.fermion_2d_vmc import (
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
Lx,Ly = 6,6
nelec = 15
t = 1.
u = 8.
spinless = True
symmetry = 'u1'
step = 0 
model = Hubbard(t,u,Lx,Ly,spinless=spinless,symmetry=symmetry)
#print(model.batched_pairs)
#exit()

if step==0:
    fpeps = load_ftn_from_disc(f'../su')
    if RANK==0:
        print(fpeps)
    for tid,tsr in fpeps.tensor_map.items():
        #print(tid,tsr.phase)
        tsr.phase = {}
    config = 1,0,1,0,1,0,\
             0,1,0,1,0,1,\
             1,0,0,0,1,0,\
             0,1,0,1,0,1,\
             1,0,1,0,0,0,\
             0,0,0,1,0,1,
    config = np.array([config])
else:
    fpeps = load_ftn_from_disc(f'psi{step}')
    config = np.load(f'config{step-1}.npy')
af = FermionAmplitudeFactory2D(fpeps,symmetry=symmetry,spinless=spinless)
af.model = model 
#print(fpeps[0,0].data)
#print(af.data_map)
#exit()
#af.is_tn = True

#sampler = FermionDenseSampler(Lx*Ly,nelec,exact=True,spinless=spinless) 
sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:])

start = step 
stop = step + 100

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
tnvmc.batchsize = 5000
tnvmc.run(start,stop,save_wfn=True)
