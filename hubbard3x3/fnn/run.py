import numpy as np
from quimb.tensor.nn_core import RBM
from quimb.tensor.tensor_2d_vmc import AmplitudeNN2D
from quimb.tensor.product_2d_vmc import ProductAmplitudeFactory2D
from quimb.tensor.fermion.fermion_2d_vmc import (
    Hubbard2D,
    FermionExchangeSampler2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
    FermionDenseSampler,
)
from quimb.tensor.tensor_vmc import (
    SR,
    scale_wfn,
    write_tn_to_disc,load_tn_from_disc,
)

import itertools,h5py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 3,3
nelec = 3,3
t = 1.
u = 8.
step = 0 
model = Hubbard2D(t,u,Lx,Ly,sep=True)

af = []
if step==0:
    fpeps = load_ftn_from_disc(f'../D4full/psi20')
else:
    fpeps = load_ftn_from_disc(f'psi{step}_0')
for tid,tsr in fpeps.tensor_map.items():
    #print(tid,tsr.phase)
    tsr.phase = {}
af1 = FermionAmplitudeFactory2D(fpeps,model)

lr = [None] * 1
nv = af1.nsite * 2
nh = 50
lr[0] = RBM(nv,nh)
if step==0:
    eps = 1e-2
    lr[0].init(0,eps)
    lr[0].init(1,eps)
    lr[0].init(2,eps)
af2 = AmplitudeNN2D(Lx,Ly,lr)
af2.get_block_dict()
af2.model = model

af = ProductAmplitudeFactory2D((af1,af2),fermion=True)
sampler = FermionDenseSampler(Lx*Ly,nelec,exact=True) 
#sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = 2, 1, 0, \
         1, 2, 0, \
         0, 1, 2,

start = step 
stop = step + 20

tnvmc = SR(sampler,solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
tnvmc.tmpdir = './' 
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.check = 'energy'
tnvmc.batchsize = 5000
tnvmc.progbar = True
tnvmc.run(start,stop,save_wfn=False)
