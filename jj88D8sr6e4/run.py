import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory,
    ExchangeSampler,
    DenseSampler,
    set_options,
    write_tn_to_disc,load_tn_from_disc,
)
from quimb.tensor.vmc import TNVMC,DMRG

import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 8,8
nspin = 32,32
D = 8
chi = 8
J1 = 1.
J2 = 0.5
thresh = 1e-20
set_options(deterministic=False,max_bond=chi)
#fpeps = load_tn_from_disc(f'../tmpdir/su_{Lx},{Ly}_rand')
fpeps = load_tn_from_disc(f'./psi35')
scale = 1.1
for tid in fpeps.tensor_map:
    tsr = fpeps.tensor_map[tid]
    tsr.modify(data=tsr.data*scale)

amp_fac = AmplitudeFactory(fpeps)
ham = J1J2(J1,J2,Lx,Ly,grad_by_ad=False)

burn_in = 40
sampler = ExchangeSampler(Lx,Ly,burn_in=burn_in) 
#sampler = DenseSampler(Lx,Ly,nspin,exact=True) 

tnvmc = TNVMC(ham,sampler,amp_fac,optimizer='sr',solve='iterative')
tmpdir = './' 
#tnvmc.tmpdir = tmpdir 
start = 35
stop = 50
tnvmc.rate2 = .5
tnvmc.cond2 = 1e-3
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.check = 'energy'

tnvmc.batchsize = int(6e4 + .1)
tnvmc.config = 1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
    print('rate=',tnvmc.rate1,tnvmc.rate2)
    print('cond=',tnvmc.cond1,tnvmc.cond2)
    print('nparams=',len(tnvmc.x))
tnvmc.run(start,stop,tmpdir=tmpdir)
