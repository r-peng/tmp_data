import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    Hubbard,
    AmplitudeFactory,
    ExchangeSampler,
    DenseSampler,
    set_options,
    write_ftn_to_disc,load_ftn_from_disc,
)
from quimb.tensor.vmc import TNVMC,DMRG

import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 8,8
nelec = 28,28
t = 1.
u = 8.
D = 8
chi = 8
set_options(symmetry='z2',flat=True,deterministic=False,max_bond=chi)
#fpeps = load_ftn_from_disc(f'../sr_6e3/psi19')
fpeps = load_ftn_from_disc(f'./psi100')
#scale = 1.25
scale = 1.05
for tid in fpeps.tensor_map:
    tsr = fpeps.tensor_map[tid]
    tsr.modify(data=tsr.data * scale)
amp_fac = AmplitudeFactory(fpeps)

ham = Hubbard(t,u,Lx,Ly,grad_by_ad=False)

burn_in = 40
sampler = ExchangeSampler(Lx,Ly,nelec,burn_in=burn_in) 

tmpdir = './' 
start = 100 
stop = 150

tnvmc = TNVMC(ham,sampler,amp_fac,normalize=True,optimizer='sr',solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('D=',D)
    print('chi=',chi)
    print('burn in=',burn_in)
    print('nparam=',len(tnvmc.x))
tnvmc.batchsize = int(3e5+.1) 
tnvmc.config = 2, 0, 2, 1, 2, 1, 2, 1, \
             1, 2, 1, 0, 1, 2, 1, 2, \
             2, 1, 2, 1, 2, 0, 2, 1, \
             1, 2, 1, 2, 1, 2, 1, 0, \
             2, 1, 2, 0, 2, 1, 2, 1, \
             1, 2, 1, 2, 1, 2, 1, 2, \
             0, 1, 2, 1, 2, 1, 2, 1, \
             1, 0, 1, 2, 1, 2, 0, 2, 
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.run(start,stop,tmpdir=tmpdir)
