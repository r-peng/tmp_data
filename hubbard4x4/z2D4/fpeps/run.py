import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
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
    write_tn_to_disc,load_tn_from_disc,
)

import itertools,h5py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 4,4
nelec = 7,7
t = 1.
u = 8.
step = 0 
chi = 16 
symmetry = 'z2'
model = Hubbard(t,u,Lx,Ly,symmetry=symmetry)

af = []
if step==0:
    fpeps = load_ftn_from_disc(f'../init/psi28')
    config = np.load(f'../init/config27.npy') 
else:
    fpeps = load_ftn_from_disc(f'psi{step}')
    config = np.load(f'config{step-1}.npy') 
for tid,tsr in fpeps.tensor_map.items():
    #print(tid,tsr.phase)
    tsr.phase = {}
scale = 1.1
#fpeps = scale_wfn(fpeps,scale)
af = FermionAmplitudeFactory2D(fpeps,symmetry=symmetry,max_bond=chi)
af.model = model 
af.spin = None 

sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:]) 

start = step 
stop = step + 200 

tnvmc = TNVMC(sampler,normalize=True,optimizer='sr',pure_newton=True,solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('nparam=',af.nparam)
tnvmc.tmpdir = './' 
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.check = 'energy'
tnvmc.batchsize = int(5e4+.1) 
save_wfn = True
tnvmc.run(start,stop,save_wfn=save_wfn)
