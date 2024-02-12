import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory,
    ExchangeSampler,
    DenseSampler,
    set_options,
    write_tn_to_disc,load_tn_from_disc,
)
from quimb.tensor.vmc import TNVMC

import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 6,6
nspin = 18,18
D = 4
chi = 4
J1 = 1.
J2 = 0.5
set_options(deterministic=True,pbc=True,max_bond=chi)
fpeps = load_tn_from_disc(f'../tmpdir/su_{Lx},{Ly}_rand')
scale = 1.
for tid in fpeps.tensor_map:
    tsr = fpeps.tensor_map[tid]
    tsr.modify(data=tsr.data * scale)
#fpeps = load_tn_from_disc(f'./psi19')
tmpdir = './' 

amp_fac = AmplitudeFactory(fpeps)
ham = J1J2(J1,J2,Lx,Ly,grad_by_ad=True,tmpdir=tmpdir,log_every=20)

burn_in = 20
sampler = ExchangeSampler(Lx,Ly,burn_in=burn_in,thresh=1e-28) 
#sampler = DenseSampler(Lx,Ly,nspin,exact=True) 
sampler.amplitude_factory = amp_fac

tnvmc = TNVMC(ham,sampler,optimizer='sr',solve_full=True,solve_dense=False)
#tnvmc.tmpdir = tmpdir 
start = 0 
stop = 50
tnvmc.rate2 = .5
tnvmc.cond2 = 1e-3
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.check = 'energy'

tnvmc.batchsize = int(6e3 + .1)
tnvmc.config = 1,0,1,0,1,0,\
         0,1,0,1,0,1,\
         1,0,1,0,1,0,\
         0,1,0,1,0,1,\
         1,0,1,0,1,0,\
         0,1,0,1,0,1,
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
    print('rate=',tnvmc.rate1,tnvmc.rate2)
    print('cond=',tnvmc.cond1,tnvmc.cond2)
    print('nparams=',len(amp_fac.get_x()))
tnvmc.sampler.config = tnvmc.config
tnvmc.debug = False
tnvmc.run(start,stop,tmpdir=tmpdir)
