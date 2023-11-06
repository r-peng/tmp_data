import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    set_options,
    Hubbard,
    FermionExchangeSampler2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
)
from quimb.tensor.tensor_vmc import (
    TNVMC,
    scale_wfn,
)

import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 16,4
nelec = 56
nelec = (nelec//2,) * 2
t = 1.
u = 8.
D = 10 
step = 123
chi = 40 
set_options(symmetry='u1',flat=True,deterministic=False)
model = Hubbard(t,u,Lx,Ly)

if step==0:
    fpeps = load_ftn_from_disc(f'{Lx}x{Ly}D{D}_rc')
    if RANK==0:
        print(fpeps)
    for tid,tsr in fpeps.tensor_map.items():
        #print(tid,tsr.phase)
        tsr.phase = {}
    config = np.load(f'{Lx}x{Ly}D{D}_config.npy')
else:
    fpeps = load_ftn_from_disc(f'psi{step}')
    config = np.load(f'config{step-1}.npy')
scale = 1.0 
#fpeps = scale_wfn(fpeps,scale)
af = FermionAmplitudeFactory2D(fpeps,max_bond=chi)
af.model = model 
af.is_tn = True

sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:])

#tmpdir = None
start = step 
stop = step + 100 

tnvmc = TNVMC(sampler,normalize=True,optimizer='sr',pure_newton=False,solve_full=True,solve_dense=False)
tnvmc.tmpdir = './' 
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('nparam=',af.nparam)
    print('D=',D)
    print('chi=',chi)
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.check = 'energy'
tnvmc.batchsize = int(5e4+1e-6) 
tnvmc.batchsize_small = int(1e4+1e-6)
tnvmc.run(start,stop)
