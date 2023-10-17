import numpy as np
from quimb.tensor.fermion.product_2d_vmc import (
    set_options,
    ProductAmplitudeFactory2D,
    RBM2D,SIGN2D,
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
af0 = FermionAmplitudeFactory2D(fpeps)
af0.model = model 
af0.spin = None 

nv = af0.nsite * 2
nh = 100
af1 = RBM2D(Lx,Ly,nv,nh)
if step==0:
    eps = 1e-3
    a = (np.random.rand(nv) * 2 - 1) * eps 
    b = (np.random.rand(nh) * 2 - 1) * eps 
    w = (np.random.rand(nv,nh) * 2 - 1) * eps
    COMM.Bcast(a,root=0)
    COMM.Bcast(b,root=0)
    COMM.Bcast(w,root=0)
    af1.save_to_disc(a,b,w,f'psi{step}_1') 
    af1.a,af1.b,af1.w = a,b,w
else:
    a,b,w = af1.load_from_disc(f'psi{step}_1.hdf5')
    #print(a)
    #print(b)
    #print(w)
    #exit()
af1.model = model

af2 = SIGN2D(Lx,Ly,afn='cos')
if step==0:
    eps = 1e-3
    w = (np.random.rand(nv) * 2 - 1) * eps 
    COMM.Bcast(w,root=0)
    if RANK==0:
        print(w)
    af2.save_to_disc(w,f'psi{step}_2') 
    af2.w = w
else:
    w = af2.load_from_disc(f'psi{step}_2.npy')
af2.model = model
#exit()

#af = ProductAmplitudeFactory2D((af0,af1,af2))
af = ProductAmplitudeFactory2D((af0,af1))
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
save_wfn = True
tnvmc.run(start,stop,save_wfn=save_wfn)
