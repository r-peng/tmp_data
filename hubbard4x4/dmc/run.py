import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    Hubbard2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
)
from quimb.tensor.tensor_dmc import Walker,SamplerSR,SamplerBranch, scale_wfn
np.set_printoptions(suppress=True,precision=6,threshold=int(1e5+.1))

import itertools,h5py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 4,4
nelec = 7,7 
t = 1.
u = 8.
deterministic = False 
chi = 16
model = Hubbard2D(t,u,Lx,Ly)

tau = 0.01
kp = 20 

fpeps = load_ftn_from_disc(f'psi400')
af = FermionAmplitudeFactory2D(fpeps,model,deterministic=deterministic,dmc=True,max_bond=chi)
wk = Walker(af,tau,kp)
wk.gamma = 0
wk.shift = -0.6 * Lx * Ly
wk.method = 1 

configs = []
for rank in range(30): 
    configs.append(np.load(f'RANK{rank}.npy'))

nmin = 10000
nmax = 20000
sampler = SamplerBranch(wk,nmin,nmax) 
sampler.init(np.concatenate(configs))
#sampler.progbar = True

start = 0
stop = 50 
tmpdir = './'
sampler.run(start,stop,tmpdir=tmpdir)
