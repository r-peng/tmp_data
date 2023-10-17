import numpy as np
from quimb.tensor.fermion.product_2d_vmc import (
    set_options,
    ProductAmplitudeFactory2D,
    RBM2D,
)
from quimb.tensor.fermion.fermion_2d_vmc import (
    Hubbard,
    FermionExchangeSampler2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
    FermionDenseSampler,
)
from quimb.tensor.tensor_vmc import (
    TNVMC,
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
set_options(symmetry='u1',flat=True,deterministic=False)
model = Hubbard(t,u,Lx,Ly,sep=True)

af = []
if step==0:
    fpeps = load_ftn_from_disc(f'../D4full/psi39')
else:
    fpeps = load_ftn_from_disc(f'psi{step}_0')
for tid,tsr in fpeps.tensor_map.items():
    #print(tid,tsr.phase)
    tsr.phase = {}
af1 = FermionAmplitudeFactory2D(fpeps)
af1.model = model 
af1.spin = None 

nv = af1.nsite * 2
nh = 100
af2 = RBM2D(Lx,Ly,nv,nh)
if step==0:
    eps = 1e-3
    a = (np.random.rand(nv) * 2 - 1) * eps 
    b = (np.random.rand(nh) * 2 - 1) * eps 
    w = (np.random.rand(nv,nh) * 2 - 1) * eps
    COMM.Bcast(a,root=0)
    COMM.Bcast(b,root=0)
    COMM.Bcast(w,root=0)
    af2.save_to_disc(a,b,w,f'psi{step}_1') 
    af2.a,af2.b,af2.w = a,b,w
else:
    a,b,w = af2.load_from_disc(f'psi{step}_1.hdf5')
    #print(a)
    #print(b)
    #print(w)
    #exit()
af2.model = model

af = ProductAmplitudeFactory2D((af1,af2))
sampler = FermionDenseSampler(Lx*Ly,nelec,exact=True) 
#sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = 0,2,1,2,0,2,1,0,1 

start = step 
stop = step + 20

tnvmc = TNVMC(sampler,normalize=True,optimizer='sr',pure_newton=True,solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('nparam=',af.nparam)
tnvmc.tmpdir = './' 
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.check = 'energy'
tnvmc.batchsize = 2000
tnvmc.run(start,stop,save_wfn=False)
