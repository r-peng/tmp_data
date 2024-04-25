import numpy as np
from quimb.tensor.tensor_1d_vmc import (
    J1J2,
    AmplitudeFactory1D,
    ExchangeSampler1D,
)
from quimb.tensor.tensor_vmc import (
    RGN,
    write_tn_to_disc,load_tn_from_disc,
    scale_wfn,
)
import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
L = 120 
nspin = 60,60
J1 = 1.
J2 = .5
step = 10
if step==0:
    psi = load_tn_from_disc('../ref/su')
    config = tuple([i%2 for i in range(L)])
else:
    psi = load_tn_from_disc(f'psi{step}')
    config = np.load(f'config{step-1}.npy')
    config = tuple(config[RANK%config.shape[0]])

model = J1J2(J1,J2,L)
af = AmplitudeFactory1D(psi,model)

burn_in = 40
#sampler = DenseSampler(L,nspin,exact=True,thresh=1e-28,fix_sector=True) 
sampler = ExchangeSampler1D(L,burn_in=burn_in,nsweep=2)
sampler.af = af
sampler.config = config

tmpdir = './' 
#tmpdir = None
start = step 
stop = 40

tnvmc = RGN(sampler,guess=1,pure_newton=False,solve_full=True,solve_dense=True)
if RANK==0:
    print('SIZE=',SIZE)
    print('L=',L)
    print('nspin=',nspin)
    print('nparam=',af.nparam)
tnvmc.rate1 = .1
tnvmc.rate2 = 1.
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-1
tnvmc.batchsize = 2000
tnvmc.tmpdir = tmpdir

tnvmc.run(start,stop)
