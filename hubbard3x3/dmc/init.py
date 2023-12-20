import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    Hubbard2D,
    FermionExchangeSampler2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
)
from quimb.tensor.tensor_dmc import scale_wfn,Progbar

import itertools,h5py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 3,3
nelec = 3,3
t = 1.
u = 8.
step = 0 
deterministic = False 
model = Hubbard2D(t,u,Lx,Ly)

fpeps = load_ftn_from_disc(f'../fpeps/psi39')
config = 0,2,1,2,0,2,1,0,1
config = np.array([config])
for tid,tsr in fpeps.tensor_map.items():
    #print(tid,tsr.phase)
    tsr.phase = {}
af = FermionAmplitudeFactory2D(fpeps,model,deterministic=deterministic,dmc=True)
sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:]) 
sampler._burn_in(exclude=[],progbar=1)

ntotal = 1000
batch_size = ntotal // SIZE + 1
ls = []
if RANK==0:
    pg = Progbar(total=batch_size)
for _ in range(batch_size):
    ls.append(sampler.sample()[0])
    if RANK==0:
        pg.update() 
np.save(f'init_RANK{RANK}_configs.npy',np.array(ls))
