import numpy as np
from quimb.tensor.product_2d_vmc import (
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
Lx,Ly = 4,4
nelec = 7
t = 1.
u = 8.
spinless = True
symmetry = 'u1'
step = 0
model = Hubbard(t,u,Lx,Ly,spinless=spinless,symmetry=symmetry)

if step==0:
    fpeps = load_ftn_from_disc(f'../fpeps/psi51')
else:
    fpeps = load_ftn_from_disc(f'psi{step}_0')
for tid,tsr in fpeps.tensor_map.items():
    #print(tid,tsr.phase)
    tsr.phase = {}
af0 = FermionAmplitudeFactory2D(fpeps,spinless=spinless,symmetry=symmetry)
af0.model = model 
af0.spin = None 

nv = af0.nsite 
nh = 100
af1 = RBM2D(Lx,Ly,nv,nh,fermion=False)
if step==0:
    eps = 1e-3
    a,b,w = af1.init(eps,fname=f'psi{step}_1')
    #a,b,w = af1.load_from_disc(f'rbm_only/psi170_1.hdf5')
else:
    a,b,w = af1.load_from_disc(f'psi{step}_1.hdf5')
af1.model = model

#nl = 1
#af2 = SIGN2D(Lx,Ly,nv,nl,afn='tanh',scale=np.pi,fermion=True,phase=True)
#if step==0:
#    w,b = af2.init(nv,1,a=0,b=1,fname=f'psi{step}_2')
#else:
#    w,b = af2.load_from_disc(f'psi{step}_2.npy')
#af2.get_block_dict(w,b)
#af2.model = model

#af = ProductAmplitudeFactory2D((af0,af1,af2),fermion=True)
af = ProductAmplitudeFactory2D((af0,af1),fermion=False)
sampler = FermionDenseSampler(Lx*Ly,nelec,exact=True,spinless=spinless) 
#sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = 0,2,1,2,0,2,1,0,1 

start = step 
stop = step + 100 

tnvmc = TNVMC(sampler,normalize=True,optimizer='sr',pure_newton=True,solve_full=True,solve_dense=True)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('nparam=',af.nparam)
tnvmc.tmpdir = './' 
tnvmc.rate1 = .1
tnvmc.rate2 = 1.
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.check = 'energy'
tnvmc.batchsize = 5000
tnvmc.save_grad_hess = True
save_wfn = True 
tnvmc.run(start,stop,save_wfn=save_wfn)
