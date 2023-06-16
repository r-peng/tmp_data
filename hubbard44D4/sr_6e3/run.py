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
u = 8.
D = 4 
chi = 8
step = 21
set_options(symmetry='u1',flat=True,deterministic=False,max_bond=chi)
#fpeps_a = load_ftn_from_disc(f'../tmpdir/su_{Lx},{Ly}_D{D}_a')
#fpeps_b = load_ftn_from_disc(f'../tmpdir/su_{Lx},{Ly}_D{D}_b')
#fpeps_j = load_tn_from_disc(f'../tmpdir/su_{Lx},{Ly}_D{D}_jastrow')
fpeps_a = load_ftn_from_disc(f'psi{step}_a')
fpeps_b = load_ftn_from_disc(f'psi{step}_b')
fpeps_j = load_tn_from_disc(f'psi{step}_boson')

#scale = 0.9
#fpeps_a = scale_wfn(fpeps_a,scale)
#fpeps_b = scale_wfn(fpeps_b,scale)
#fpeps_j = scale_wfn(fpeps_j,scale)

amp_fac = AmplitudeFactory(fpeps_a,fpeps_b,fpeps_j)

ham = Hubbard(t,u,Lx,Ly)

burn_in = 40 
sampler = ExchangeSampler(Lx,Ly,nelec,burn_in=burn_in) 
sampler.amplitude_factory = amp_fac

tmpdir = './' 
start = step 
stop = 30

tnvmc = TNVMC(ham,sampler,normalize=True,optimizer='sr',solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('D=',D)
    print('chi=',chi)
    print('burn in=',burn_in)
    print('nparam=',len(amp_fac.get_x()))
config = 2, 0, 1, 2, \
         0, 1, 2, 1, \
         1, 2, 1, 2, \
         2, 1, 2, 1
tnvmc.batchsize = int(6e4+.1) 
tnvmc.config = config 
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.debug = False
tnvmc.sampler.config = config
tnvmc.run(start,stop,tmpdir=tmpdir)
