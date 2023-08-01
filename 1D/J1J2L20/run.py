import numpy as np
from quimb.tensor.tensor_1d_vmc import (
    set_options,
    J1J2,
    Hamiltonian,
    AmplitudeFactory,
    ExchangeSampler,
)
from quimb.tensor.tensor_vmc import (
    DenseSampler,
    TNVMC,
    write_tn_to_disc,load_tn_from_disc,
    scale_wfn,
)
import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
L = 20 
nspin = L//2,L//2
J1 = 1.
J2 = .5
step = 0
set_options()
if step==0:
    psi = load_tn_from_disc('ref/su')
else:
    psi = load_tn_from_disc(f'sr/psi{step}')
amp_fac = AmplitudeFactory(psi)

model = J1J2(J1,J2,L)
ham = Hamiltonian(model)

burn_in = 40
#sampler = DenseSampler(L,nspin,exact=True,thresh=1e-28) 
sampler = ExchangeSampler(L,burn_in=burn_in)
sampler.amplitude_factory = amp_fac

tmpdir = './' 
start = step 
stop = 20

tnvmc = TNVMC(ham,sampler,normalize=True,optimizer='sr',solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('L=',L)
    print('nspin=',nspin)
    print('nparam=',len(amp_fac.get_x()))
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.debug = False
tnvmc.batchsize = int(5e4 + .1)
tnvmc.check = 'energy'

config = tuple([i%2 for i in range(L)]) 
tnvmc.config = config
tnvmc.sampler.config = config
tnvmc.run(start,stop,tmpdir=tmpdir)
