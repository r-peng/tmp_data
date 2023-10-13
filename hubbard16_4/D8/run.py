import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    set_options,
    Hubbard,
    FermionExchangeSampler2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
)
from quimb.tensor.tensor_vmc import (
    TNVMC,
    scale_wfn,
)

import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 4,16
nelec = 56
nelec = (nelec//2,) * 2
t = 1.
u = 8.
D = 8
step = 0 
chi = 8 
set_options(symmetry='u1',flat=True,deterministic=False)
model = Hubbard(t,u,Lx,Ly)

if step==0:
    fpeps = load_ftn_from_disc(f'../suD{D}')
    if RANK==0:
        print(fpeps)
    for tid,tsr in fpeps.tensor_map.items():
        #print(tid,tsr.phase)
        tsr.phase = {}
else:
    fpeps = load_ftn_from_disc(f'psi{step}')
scale = 1.2
fpeps = scale_wfn(fpeps,scale)
af = FermionAmplitudeFactory2D(fpeps,max_bond=chi)
af.model = model 
af.is_tn = True

sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af

tmpdir = './' 
#tmpdir = None
start = step 
stop = step + 10 

tnvmc = TNVMC(sampler,normalize=True,optimizer='sr',pure_newton=False,solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('nparam=',af.nparam)
    print('D=',D)
    print('chi=',chi)
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.debug = False
tnvmc.check = 'energy'
tnvmc.batchsize = int(1e4+1e-6) 
hole = [(1,3),(2,3),(1,6),(2,6),(1,9),(2,9),(1,12),(2,12)]
config = []
for i,j in itertools.product(range(Lx),range(Ly)):
    if (i,j) in hole:
        ci = 0
    elif (i+j)%2==0:
        ci = 1
    else:
        ci = 2
    config.append(ci)
config = tuple(config)
tnvmc.config = config
tnvmc.sampler.config = config
tnvmc.run(start,stop,tmpdir=tmpdir)
