import numpy as np
from quimb.tensor.product_2d_vmc import (
    ProductAmplitudeFactory2D,
    RBM1D,
)
from quimb.tensor.fermion.fermion_1d_vmc import (
    Hubbard1D,
    FermionExchangeSampler1D,
    FermionAmplitudeFactory1D,
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
Lx,Ly = 1,12
nelec = 5,5
t = 1.
u = 8.
spinless = False 
symmetry = 'z2'
step = 0 
model = Hubbard1D(t,u,Ly,sep=True,spinless=spinless,symmetry=symmetry)

if step==0:
    fpeps = load_ftn_from_disc(f'../fpeps/psi40')
    config = np.load(f'../fpeps/config39.npy')
else:
    fpeps = load_ftn_from_disc(f'psi{step}_0')
    config = np.load(f'config{step-1}.npy')
for tid,tsr in fpeps.tensor_map.items():
    #print(tid,tsr.phase)
    tsr.phase = {}
af0 = FermionAmplitudeFactory1D(fpeps,spinless=spinless,symmetry=symmetry)
af0.model = model 
af0.spin = None 

nv = af0.nsite * 2 
nh = 200
af1 = RBM1D(Ly,nv,nh,fermion=True)
if step==0:
    eps = 1e-2
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

af = ProductAmplitudeFactory2D((af0,af1),fermion=True)
#sampler = FermionDenseSampler(Lx*Ly,nelec,exact=True,spinless=spinless) 
sampler = FermionExchangeSampler1D(Ly,burn_in=40,nsweep=2)
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:]) 

start = step 
stop = step + 400 

tnvmc = SR(sampler,normalize=True,solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('nparam=',af.nparam)
tnvmc.tmpdir = './' 
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.batchsize = 20000
save_wfn = True 
tnvmc.run(start,stop,save_wfn=save_wfn)
