import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    Hubbard2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
)
from quimb.tensor.tensor_dmc import Walker,Sampler,scale_wfn
np.set_printoptions(suppress=True,precision=6,threshold=int(1e5+.1))

import itertools,h5py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 3,3
nelec = 3,3
t = 1.
u = 8.
deterministic = False 
model = Hubbard2D(t,u,Lx,Ly)

tau = 0.6

fpeps = load_ftn_from_disc(f'../../psi39')
af = FermionAmplitudeFactory2D(fpeps,model,deterministic=deterministic,dmc=True)
wk = Walker(af,tau)
wk.gamma = 0
wk.shift =  -0.739 * Lx * Ly
if RANK==0:
    print('shift',wk.shift)

configs = []
for _ in range(5):
    for rank in range(30): 
        configs.append(np.load(f'../configs/init_RANK{rank}_configs.npy'))

nmin = 500
nmax = 10000
#sampler = SamplerBranch(wk,nmin,nmax) 
sampler = Sampler(wk) 
sampler.init(np.concatenate(configs))

start = 0 
stop = 200 
sampler.progbar = True

tmpdir = f'./'
sampler.run(start,stop,tmpdir=tmpdir)
