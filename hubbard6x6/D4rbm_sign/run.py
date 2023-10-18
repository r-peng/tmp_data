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
Lx,Ly = 6,6
nelec = 32
nelec = (nelec//2,) * 2
t = 1.
u = 8.
D = 4 
step = 51 
chi = 16
set_options(symmetry='u1',flat=True,deterministic=False)
model = Hubbard(t,u,Lx,Ly,sep=True)

if step==0:
    fpeps = load_ftn_from_disc(f'../D4rbm/psi48_0')
    if RANK==0:
        print(fpeps)
    for tid,tsr in fpeps.tensor_map.items():
        #print(tid,tsr.phase)
        tsr.phase = {}
    config = np.load(f'../D4rbm/config47.npy')
else:
    fpeps = load_ftn_from_disc(f'psi{step}_0')
    config = np.load(f'config{step-1}.npy')
scale = 1.1
fpeps = scale_wfn(fpeps,scale)
af0 = FermionAmplitudeFactory2D(fpeps,max_bond=chi)
af0.model = model 
af0.is_tn = True
af0.spin = None

nv = af0.nsite * 2
nh = 1000 
af1 = RBM2D(Lx,Ly,nv,nh)
if step==0:
    #eps = 1e-2
    #a = (np.random.rand(nv)*2 - 1)*eps
    #b = (np.random.rand(nh)*2 - 1)*eps
    #w = (np.random.rand(nv,nh)*2 - 1)*eps
    #COMM.Bcast(a,root=0)
    #COMM.Bcast(b,root=0)
    #COMM.Bcast(w,root=0)
    #af1.save_to_disc(a,b,w,f'psi{step}_1')
    a,b,w = af1.load_from_disc(f'../D4rbm/psi48_1.hdf5')
else:
    a,b,w = af1.load_from_disc(f'psi{step}_1.hdf5')
af1.model = model

af2 = SIGN2D(Lx,Ly,afn='cos')
if step==0:
    eps = 1e-2
    w = (np.random.rand(nv) * 2 - 1) * eps
    COMM.Bcast(w,root=0)
    af2.save_to_disc(w,f'psi{step}_2')
    af2.w = w
else:
    w = af2.load_from_disc(f'psi{step}_2.npy')
af2.model = model

af = ProductAmplitudeFactory2D((af0,af1,af2))
sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:])

start = step 
stop = step + 100 

tnvmc = TNVMC(sampler,normalize=True,optimizer='sr',pure_newton=False,solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('nparam=',af.nparam)
    print('D=',D)
    print('chi=',chi)
tnvmc.tmpdir = './' 
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.check = 'energy'
tnvmc.batchsize = int(5e4+1e-6) 
tnvmc.run(start,stop)
