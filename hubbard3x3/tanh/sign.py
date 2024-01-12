import numpy as np
from quimb.tensor.product_2d_vmc import (
    ProductAmplitudeFactory2D,
    NN2D,Dense,
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
    Progbar,
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
chi = None
spinless = False 
symmetry = 'u1'
step = 186 
model = Hubbard2D(t,u,Lx,Ly,sep=True,spinless=spinless,symmetry=symmetry)

if step==0:
    fpeps = load_ftn_from_disc(f'../psi20')
    config = 1,2,0,2,2,1,0,1,0
    config = np.array([config])
else:
    fpeps = load_ftn_from_disc(f'psi{step}_0')
    config = np.load(f'config{step-1}.npy')
for tid,tsr in fpeps.tensor_map.items():
    #print(tid,tsr.phase)
    tsr.phase = {}
af0 = FermionAmplitudeFactory2D(fpeps,model,spinless=spinless,symmetry=symmetry,max_bond=chi)
af0.spin = None 

lr = [None] * 2 
nx,ny = 2*Lx*Ly,20
afn = 'tanh'
lr[0] = Dense(nx,ny,afn,bias=True)
lr[0].scale = 1.
if step==0:
    lr[0].init(0,1/nx)
    lr[0].init(1,1e-1)

nx,ny = ny,1
afn = None
lr[1] = Dense(nx,ny,afn,bias=False,pre_act=False,post_act=False)
if step==0:
    eps = 1e-1
    lr[1].init(0,eps/nx)

input_format = -1,1 
af1 = NN2D(Lx,Ly,lr,input_format=input_format,log=False,fermion=True)
af1.model = model
if step!=0:
    af1.load_from_disc(f'psi{step}_1')
af1.get_block_dict()

af = ProductAmplitudeFactory2D((af0,af1),fermion=True)
#sampler = FermionDenseSampler(Lx*Ly,nelec,exact=True,spinless=spinless) 
sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af1
sampler.config = tuple(config[RANK%config.shape[0],:]) 
ntotal = 5000
n = 0
pg = Progbar(ntotal)
for _ in range(ntotal):
    config,_ = sampler.sample()
    cx = af1.amplitude(config)
    if cx<0:
        n += 1
    pg.update()
print('nminus=',n)
exit()

start = step 
stop = step + 200 

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
