import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory2D,
    ExchangeSampler2D,
    Hamiltonian2D,
    set_options,
)
from quimb.tensor.tensor_vmc import (
    TNVMC,
    DenseSampler,
    write_tn_to_disc,load_tn_from_disc,
)
import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 8,8
nspin = 32,32
D = 4
chi = 8
J1 = 1.
J2 = 0.5
thresh = 1e-20
set_options()
fpeps = load_tn_from_disc(f'../tmpdir/su_{Lx},{Ly}_rand')
#fpeps = load_tn_from_disc(f'./psi100')

amp_fac = AmplitudeFactory2D(fpeps,max_bond=chi)
model = J1J2(J1,J2,Lx,Ly)
ham = Hamiltonian2D(model)

burn_in = 40
sampler = ExchangeSampler2D(Lx,Ly,burn_in=burn_in) 
#sampler = DenseSampler(Lx,Ly,nspin,exact=True) 
sampler.amplitude_factory = amp_fac

tnvmc = TNVMC(ham,sampler,optimizer='sr',solve_dense=False,solve_full=True)
#tmpdir = './' 
tmpdir = None
start = 0
stop = 1 
tnvmc.rate2 = .5
tnvmc.cond2 = 1e-3
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.check = 'energy'

tnvmc.batchsize = int(5e3 + .1)
tnvmc.config = 1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1
tnvmc.sampler.config = tnvmc.config
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
    print('rate=',tnvmc.rate1,tnvmc.rate2)
    print('cond=',tnvmc.cond1,tnvmc.cond2)
    print('nparams=',len(amp_fac.get_x()))
tnvmc.run(start,stop,tmpdir=tmpdir)
