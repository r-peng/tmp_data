import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc_subspace import (
    set_options,
    Hubbard,
    AmplitudeFactory,
    ExchangeSampler,
)
from quimb.tensor.fermion.fermion_2d_vmc_ import (
    write_ftn_to_disc,load_ftn_from_disc,
)
from quimb.tensor.tensor_2d_vmc_ import (
    scale_wfn,
    write_tn_to_disc,load_tn_from_disc,
)
from quimb.tensor.vmc import TNVMC

import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 4,4
nelec = 7,7
t = 1.
u = 1.
D = 4 
chi = 16 
step = 30 
data_map = set_options(symmetry='u1',flat=True,deterministic=False,max_bond=chi)
#fpeps_a = load_ftn_from_disc(f'Lx{Lx}Ly{Ly}M2_a_ev')
#fpeps_b = load_ftn_from_disc(f'Lx{Lx}Ly{Ly}M2_b_ev')
#fpeps_j = load_tn_from_disc(f'gutzwiller')
#fpeps_j = load_tn_from_disc(f'uniform')
fpeps_a = load_ftn_from_disc(f'psi{step}_0')
fpeps_b = load_ftn_from_disc(f'psi{step}_1')
fpeps_j = load_tn_from_disc(f'psi{step}_2')

#scale = 1.3
#fpeps_a = scale_wfn(fpeps_a,scale)
#fpeps_b = scale_wfn(fpeps_b,scale)
#fpeps_j = scale_wfn(fpeps_j,scale)

psi = fpeps_a,fpeps_b,fpeps_j
amp_fac = AmplitudeFactory(psi)
ham = Hubbard(t,u,Lx,Ly)

burn_in = 40 
sampler = ExchangeSampler(Lx,Ly,nelec,burn_in=burn_in) 
sampler.amplitude_factory = amp_fac

tmpdir = './' 
#tmpdir = None
start = step 
stop = step + 30 

tnvmc = TNVMC(ham,sampler,normalize=True,optimizer='sr',solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('D=',D)
    print('chi=',chi)
    print('burn in=',burn_in)
    print('nparam=',len(amp_fac.get_x()))
config = 1,2,1,2,\
         2,1,0,1,\
         0,2,1,2,\
         2,1,2,1,
#config = 1,0,1,2,\
#         2,1,2,1,\
#         1,2,1,2,\
#         2,1,2,0,
tnvmc.batchsize = int(6e4+.1) 
tnvmc.config = config 
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.debug = False
tnvmc.sampler.config = config
tnvmc.run(start,stop,tmpdir=tmpdir)
