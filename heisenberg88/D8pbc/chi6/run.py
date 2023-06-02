import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    Heisenberg,
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
Lx,Ly = 8,8
nspin = 32,32
D = 8
chi = 6 
J = 1.
h = 0.
step = 26
set_options(deterministic=True,pbc=True,max_bond=chi)
#fpeps = load_tn_from_disc(f'../tmpdir/su_{Lx},{Ly}_rand')
fpeps = load_tn_from_disc(f'./psi{step}')
#scale = 1.05
#for tid in fpeps.tensor_map:
#    tsr = fpeps.tensor_map[tid]
#    tsr.modify(data = tsr.data * scale)
#print(fpeps)

tmpdir = './' 
amp_fac = AmplitudeFactory(fpeps)
#ham = J1J2(J1,J2,Lx,Ly,grad_by_ad=True,tmpdir=tmpdir,log_every=60)
ham = Heisenberg(J,h,Lx,Ly,grad_by_ad=True)

burn_in = 40
sampler = ExchangeSampler(Lx,Ly,burn_in=burn_in,thresh=1e-28) 
#sampler = DenseSampler(Lx,Ly,nspin,exact=True) 
sampler.amplitude_factory = amp_fac

tnvmc = TNVMC(ham,sampler,normalize=True,optimizer='rgn',solve_full=True,solve_dense=False)
#tnvmc.tmpdir = tmpdir 
start = step
stop = step + 1
tnvmc.rate2 = .5
tnvmc.cond2 = 1e-3
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.check = 'energy'

tnvmc.batchsize = int(1e5 + .1)
tnvmc.config = 1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1
if RANK==0:
    print('RANK=',RANK)
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
    print('chi=',chi)
    print('rate=',tnvmc.rate1,tnvmc.rate2)
    print('cond=',tnvmc.cond1,tnvmc.cond2)
    print('nparams=',len(amp_fac.get_x()))
tnvmc.run(start,stop,tmpdir=tmpdir)
