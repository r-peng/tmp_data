import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory2D,
    ExchangeSampler2D,
)
from quimb.tensor.tensor_vmc import (
    SR,
    write_tn_to_disc,load_tn_from_disc,
    scale_wfn,
)
import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 8,6
nspin = 24,24
D = 8
chi = 8
J1 = 1.
J2 = 0.5
step = 0
model = J1J2(J1,J2,Lx,Ly)
if step==0:
    fpeps = load_tn_from_disc(f'../suD{D}')
    config = tuple([(i+j)%2 for i,j in itertools.product(range(Lx),range(Ly))])
else:
    fpeps = load_tn_from_disc(f'psi{step}')
fpeps = scale_wfn(fpeps,1.1)

burn_in = 40
sampler = ExchangeSampler2D(Lx,Ly,burn_in=burn_in) 
sampler.af = AmplitudeFactory2D(fpeps,model,max_bond=chi)
sampler.config = tuple(config)

tnvmc = SR(sampler,normalize=True,solve_dense=False,solve_full=True)
tnvmc.tmpdir = './' 
start = step 
stop = 100
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3

tnvmc.batchsize = int(1e4 + .1)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
    print('rate=',tnvmc.rate1)
    print('cond=',tnvmc.cond1)
tnvmc.run(start,stop,save_wfn=True)
