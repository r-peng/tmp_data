import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    Hubbard2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
)
from quimb.tensor.tensor_dmc import Sampler, scale_wfn
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
step = 0 
deterministic = False 
model = Hubbard2D(t,u,Lx,Ly)

fpeps = load_ftn_from_disc(f'../fpeps/psi39')
af = FermionAmplitudeFactory2D(fpeps,model,deterministic=deterministic,dmc=True)
shift = -0.7392952143033105 * af.nsite
#shift = 0

configs = np.load(f'init_RANK{RANK}_configs.npy')
weights = np.ones(configs.shape[0])
sampler = Sampler(af,configs,weights) 
sampler.tau = 0.001
sampler.accum = 10
sampler.shift = shift
sampler.progbar = True

start = 0
stop = 10
for i in range(start,stop):
    sampler.sample(fname=f'step{step}')
