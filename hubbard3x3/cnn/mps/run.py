import numpy as np
from quimb.tensor.product_vmc import relu_init_spin
from quimb.tensor.product_2d_vmc import (
    ProductAmplitudeFactory2D,
    CNN2D1,CNN2D2,
)
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

import itertools,h5py,scipy
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 3,3
nelec = 3,3
t = 1.
u = 8.
step = 0 
deterministic = False 
model = Hubbard2D(t,u,Lx,Ly,sep=True,deterministic=deterministic)

if step==0:
    fpeps = load_ftn_from_disc(f'../../fpeps/psi20')
    config = 0,2,1,2,0,2,1,0,1
    config = np.array([config])
else:
    fpeps = load_ftn_from_disc(f'psi{step}_0')
    config = 0,2,1,2,0,2,1,0,1
    config = np.array([config])
for tid,tsr in fpeps.tensor_map.items():
    #print(tid,tsr.phase)
    tsr.phase = {}
af0 = FermionAmplitudeFactory2D(fpeps,model,deterministic=deterministic)

D = 4
af1 = CNN2D2(Lx,Ly,D,fermion=True)
if step==0:
    eps = 1e-1
    for key in af1.param_keys:
        af1.init(key,eps)
    K = af1.params['in']
    for i in range(K.shape[0]):    
        K[i,:,:] = scipy.linalg.expm(K[i,:,:]-K[i,:,:].T)
    af1.params['in'] = K
    af1.save_to_disc(f'psi{step}_1')
else:
    af1.load_from_disc(f'psi{step}_1.hdf5')
af1.get_block_dict()
af1.model = model

af = ProductAmplitudeFactory2D((af0,af1),fermion=True)
#sampler = FermionDenseSampler(Lx*Ly,nelec,exact=True) 
sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
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
tnvmc.batchsize = 5000
save_wfn = True 
tnvmc.run(start,stop,save_wfn=save_wfn)
