import numpy as np
from quimb.tensor.tensor_2d_vmc import J1J2,Hamiltonian,ExchangeSampler2
from quimb.tensor.tensor_2d_vmc_gauge import AmplitudeFactory,set_options
from quimb.tensor.tensor_vmc import (
    TNVMC,
    DenseSampler,
    write_tn_to_disc,load_tn_from_disc,
    scale_wfn,
)

import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 8,8
nspin = 32,32
D = 2 
J1 = 1.
J2 = 0.5
chi = 8
#set_options(deterministic=True)
set_options()
fpeps = load_tn_from_disc(f'../../sr_6e4/psi50')
#fpeps = load_tn_from_disc(f'./psi100')
fpeps = scale_wfn(fpeps,1.1)

amp_fac = AmplitudeFactory(fpeps,max_bond=chi)
amp_fac.init_gauge(eps=1e-3,fname='gauge_init')
#print(RANK,amp_fac.gauges)
#exit()
#amp_fac.load_gauge_from_disc('gauge_init')
#print(amp_fac.gauges)
#exit()

model = J1J2(J1,J2,Lx,Ly)
ham = Hamiltonian(model)

burn_in = 40
sampler = ExchangeSampler2(Lx,Ly,burn_in=burn_in) 
#sampler = DenseSampler(Lx*Ly,nspin,exact=True) 
sampler.amplitude_factory = amp_fac

tnvmc = TNVMC(ham,sampler,normalize=True,optimizer='sr',solve_full=True,solve_dense=False)
tmpdir = './' 
start = 0
stop = 50 
tnvmc.rate2 = .5
tnvmc.cond2 = 1e-3
tnvmc.rate1 = 1e-1
tnvmc.cond1 = 1e-3
tnvmc.check = 'energy'

tnvmc.batchsize = int(6e4 + .1)
config = []
for i,j in itertools.product(range(Lx),range(Ly)):
    config.append(1-(i+j)%2)
tnvmc.config = tuple(config)
tnvmc.sampler.config = tnvmc.config
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
    print('rate=',tnvmc.rate1,tnvmc.rate2)
    print('cond=',tnvmc.cond1,tnvmc.cond2)
    print('nparams=',len(amp_fac.get_x()))
tnvmc.run(start,stop,tmpdir=tmpdir)
