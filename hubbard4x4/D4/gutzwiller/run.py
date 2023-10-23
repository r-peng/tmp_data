import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc_subspace import (
    set_options,
    Hamiltonian,
    AmplitudeFactory,
)
from quimb.tensor.fermion.fermion_2d_vmc import (
    Hubbard,
    ExchangeSampler,
)
from quimb.tensor.fermion.fermion_vmc import (
    DenseSampler,
    write_ftn_to_disc,load_ftn_from_disc,
)
from quimb.tensor.tensor_vmc import (
    TNVMC,
    write_tn_to_disc,load_tn_from_disc,
    scale_wfn,
)
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
step = 0
set_options(symmetry='u1',flat=True,deterministic=True)
fpeps = load_ftn_from_disc(f'Lx{Lx}Ly{Ly}N{nelec[0]}M2')
peps = load_tn_from_disc(f'jastrowD2')

psi = fpeps.copy(),fpeps.copy(),peps
amp_fac = AmplitudeFactory(psi)

model = Hubbard(t,u,Lx,Ly,spinless=True)
ham = Hamiltonian(model)

burn_in = 40
#sampler = DenseSampler(Lx*Ly,nelec,spinless=False,exact=True) 
sampler = ExchangeSampler(Lx,Ly,burn_in=burn_in) 
sampler.amplitude_factory = amp_fac

tmpdir = './' 
start = step 
stop = 10 

tnvmc = TNVMC(ham,sampler,normalize=True,optimizer='sr',solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('D=',D)
    print('nparam=',len(amp_fac.get_x()))
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.debug = False
tnvmc.batchsize = int(5e4 + .1)

config = 0, 2, 1, 2, \
         2, 1, 2, 1, \
         1, 2, 1, 2, \
         2, 1, 0, 1
tnvmc.config = config
tnvmc.sampler.config = config
tnvmc.run(start,stop,tmpdir=tmpdir)
