import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory2D,
    ExchangeSampler2D,
)
from quimb.tensor.tensor_vmc import (
    SR,
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
D = 3 
J1 = 1.
J2 = 0.5
chi = 8
step = 200 
if step==0:
    psi = load_tn_from_disc(f'../suD{D}')
    config = [(i+j)%2 for i,j in itertools.product(range(Lx),range(Ly))]
    config = np.array([config])
    scale = 1.2
    fpeps = scale_wfn(psi,scale)
else:
    psi = load_tn_from_disc(f'psi{step}')
    config = np.load(f'config{step-1}.npy')
#fpeps = load_tn_from_disc(f'./psi100')

model = J1J2(J1,J2,Lx,Ly)
af = AmplitudeFactory2D(psi,model,max_bond=chi)

burn_in = 40
sampler = ExchangeSampler2D(Lx,Ly,burn_in=burn_in) 
#sampler = DenseSampler(Lx,Ly,nspin,exact=True) 
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:])

tnvmc = SR(sampler,normalize=True,solve_full=True,solve_dense=False)
tnvmc.tmpdir = './' 
start = step 
stop = step + 200
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.check = 'energy'

tnvmc.batchsize = int(5e3 + .1)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
tnvmc.run(start,stop,save_wfn=True)
