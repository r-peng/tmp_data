import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory,
    ExchangeSampler1,
    ExchangeSampler2,
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
Lx,Ly = 4,4
nspin = 8,8
D = 2
J1 = 1.
J2 = 0.5
set_options()
#set_options(deterministic=True)
fpeps = load_tn_from_disc(f'../tmpdir/su_{Lx},{Ly}_rand')
#fpeps = load_tn_from_disc(f'./psi100')

nbatch = 1 
amp_fac = AmplitudeFactory(fpeps)
ham = J1J2(J1,J2,Lx,Ly,nbatch=nbatch)

burn_in = 40
#burn_in = 0
sampler = ExchangeSampler2(Lx,Ly,burn_in=burn_in) 
#sampler = DenseSampler(Lx,Ly,nspin,exact=True) 
sampler.amplitude_factory = amp_fac

tnvmc = TNVMC(ham,sampler,optimizer='rgn',solve_full=False,solve_dense=True)
#tmpdir = './' 
tmpdir = None
tnvmc.tmpdir = tmpdir 
start = 0
stop = 20
tnvmc.rate2 = .5
tnvmc.cond2 = 1e-3
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.check = 'energy'
tnvmc.debug = False

tnvmc.batchsize = int(5e4 + .1)
config = 1,0,1,0,\
               0,1,0,1,\
               1,0,1,0,\
               0,1,0,1
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
    print('rate=',tnvmc.rate1,tnvmc.rate2)
    print('cond=',tnvmc.cond1,tnvmc.cond2)
    print('nparams=',len(amp_fac.get_x()))
#tnvmc.sampler._burn_in(config=config,burn_in=40)
tnvmc.sampler.config = config
tnvmc.config = config
tnvmc.run(start,stop,tmpdir=tmpdir)
