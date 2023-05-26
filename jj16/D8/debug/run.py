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
Lx,Ly = 16,16
nspin = 128,128
D = 8
chi = 8 
J1 = 1.
J2 = 0.5
set_options(deterministic=False,max_bond=chi)
#fpeps = load_tn_from_disc(f'../tmpdir/su_{Lx},{Ly}_rand')
#fpeps = load_tn_from_disc(f'../tmpdir/su1')
fpeps = load_tn_from_disc(f'../tmpdir/init')
#fpeps = load_tn_from_disc(f'psi31')
scale = 1.1
for tid in fpeps.tensor_map:
    tsr = fpeps.tensor_map[tid]
    tsr.modify(data = tsr.data * scale)
#print(fpeps)

amp_fac = AmplitudeFactory(fpeps)
ham = J1J2(J1,J2,Lx,Ly,grad_by_ad=True)

burn_in = 0 
sampler = ExchangeSampler(Lx,Ly,burn_in=burn_in,thresh=1e-28) 
#sampler = DenseSampler(Lx,Ly,nspin,exact=True) 
sampler.amplitude_factory = amp_fac

tnvmc = TNVMC(ham,sampler,normalize=True,optimizer='sr',solve_full=True,solve_dense=False)
tmpdir = './' 
#tnvmc.tmpdir = tmpdir 
start = 0 
stop = 1 
tnvmc.rate2 = .5
tnvmc.cond2 = 1e-3
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.check = 'energy'

tnvmc.batchsize = int(6e3 + .1)
config = []
na,nb = 0,0
for i in range(Lx):
    for j in range(Ly):
        config.append(1 - (i+j) % 2)
        if (i+j)%2==0:
            na += 1
        else:
            nb += 1
tnvmc.config = tuple(config)
if RANK==0:
    print('RANK=',RANK)
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
    print('chi=',chi)
    print('rate=',tnvmc.rate1,tnvmc.rate2)
    print('cond=',tnvmc.cond1,tnvmc.cond2)
    x = amp_fac.get_x()
    print('nparams=',len(x),len(x[np.fabs(x)>1e-5]))
    print(len(tnvmc.config),na,nb)
tnvmc.run(start,stop,tmpdir=tmpdir)
