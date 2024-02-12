import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory2D,
    ExchangeSampler2D,
)
from quimb.tensor.tensor_vmc import (
    write_tn_to_disc,load_tn_from_disc,
    )
from quimb.tensor.tensor_dmc import Walker,Sampler, scale_wfn
np.set_printoptions(suppress=True,precision=6,threshold=int(1e5+.1))

import itertools,h5py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 6,4
nspin = 12,12
J1 = 1.
J2 = 0.5
chi = 8
deterministic = False 
model = J1J2(J1,J2,Lx,Ly)
step = 460
peps = load_tn_from_disc(f'../peps/psi{step}')
config = np.load(f'../peps/config{step-1}.npy')
configs = [config.copy() for _ in range(10)]

tau = 0.1

af = AmplitudeFactory2D(peps,model,max_bond=chi,deterministic=deterministic,dmc=True)
wk = Walker(af,tau)
wk.gamma = 0
wk.shift =  -0.4685 * Lx * Ly
if RANK==0:
    print('shift',wk.shift)

sampler = Sampler(wk) 
sampler.init(np.concatenate(configs))

start = 0 
stop = 200 
sampler.progbar = True

tmpdir = f'./'
sampler.run(start,stop,tmpdir=tmpdir)
