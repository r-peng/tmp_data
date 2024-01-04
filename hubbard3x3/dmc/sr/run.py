import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    Hubbard2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
)
from quimb.tensor.tensor_dmc import Walker,Sampler, scale_wfn
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

gamma = 0
tau = 10 
lamb = nelec[0] * u

fpeps = load_ftn_from_disc(f'../fpeps/psi39')
af = FermionAmplitudeFactory2D(fpeps,model,deterministic=deterministic,dmc=True)
wk = Walker(af,tau,lamb=lamb,gamma=gamma)

configs = []
for rank in range(30): 
    configs.append(np.load(f'../cross/init_RANK{RANK}_configs.npy'))

L = 10
N = 10 
sampler = Sampler(wk,L,N) 
sampler.init(config=np.concatenate(configs))
sampler.progbar = True

start = 0
stop = 50
tmpdir = './'
sampler.run(start,stop,tmpdir=tmpdir)
