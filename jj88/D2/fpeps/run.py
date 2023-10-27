import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory2D,
    ExchangeSampler2D,
    set_options,
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
Lx,Ly = 8,8
nspin = (Lx * Ly // 2,) * 2
D = 2 
J1 = 1.
J2 = 0.5
chi = 8
set_options(deterministic=False)
step = 0 
if step==0:
    fpeps = load_tn_from_disc(f'../psi50')
    config = [(i+j)%2 for i,j in itertools.product(range(Lx),range(Ly))]
    config = np.array([config])
else:
    fpeps = load_tn_from_disc(f'psi{step}')
    config = np.load(f'config{step-1}.npy')
scale = 1.2
#fpeps = scale_wfn(fpeps,scale)
#fpeps = load_tn_from_disc(f'./psi100')

af = AmplitudeFactory2D(fpeps,max_bond=chi)
af.model = J1J2(J1,J2,Lx,Ly)

burn_in = 40
sampler = ExchangeSampler2D(Lx,Ly,burn_in=burn_in) 
#sampler = DenseSampler(Lx,Ly,nspin,exact=True) 
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:])

tnvmc = TNVMC(sampler,normalize=True,optimizer='sr',pure_newton=False,solve_full=True,solve_dense=False)
tnvmc.tmpdir = './' 
start = step 
stop = step + 50
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.check = 'energy'

tnvmc.batchsize = int(5e4 + .1)
tnvmc.batchsize_small = int(5e3 + .1)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
    print('rate=',tnvmc.rate1,tnvmc.rate2)
    print('cond=',tnvmc.cond1,tnvmc.cond2)
    print('nparams=',af.nparam)
tnvmc.run(start,stop)
