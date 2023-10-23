import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    set_options,
    Hubbard,
    FermionAmplitudeFactory2D,
    FermionExchangeSampler2D,
)
from quimb.tensor.fermion.fermion_vmc import write_ftn_to_disc,load_ftn_from_disc
from quimb.tensor.tensor_2d_vmc import Hamiltonian2D 
from quimb.tensor.tensor_vmc import TNVMC,scale_wfn
import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 4,4
nelec = 7,7
t = 1.
u = 4.
D = 4
chi = 8
step = 0
set_options(symmetry='u1',flat=True)
psi = load_ftn_from_disc('su')
amp_fac = FermionAmplitudeFactory2D(psi,max_bond=chi)

model = Hubbard(t,u,Lx,Ly)
ham = Hamiltonian2D(model)

burn_in = 40
sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=burn_in)
sampler.amplitude_factory = amp_fac

tmpdir = './' 
start = step 
stop = step + 50

tnvmc = TNVMC(ham,sampler,normalize=True,optimizer='sr',solve_full=True,solve_dense=False)
config = 2, 0, 2, 1, \
         1, 2, 1, 2, \
         2, 1, 2, 1, \
         1, 2, 1, 0 
tnvmc.config = config
tnvmc.sampler.config = config
if RANK==0:
    print('SIZE=',SIZE)
    print('nparam=',len(amp_fac.get_x()))
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.debug = False
tnvmc.batchsize = int(5e4 + .1)
tnvmc.check = 'energy'

tnvmc.run(start,stop,tmpdir=tmpdir)
