import numpy as np
from quimb.tensor.product_2d_vmc import SumFNN2D,SumAmplitudeFactory2D
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
np.set_printoptions(precision=6,suppress=False)
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 3,3
nelec = 3,3
t = 1.
u = 8.
step = 0 
deterministic = False 
spinless = False
model = Hubbard2D(t,u,Lx,Ly,sep=True,deterministic=deterministic,spinless=spinless)

if step==0:
    fpeps = load_ftn_from_disc(f'../D4full/psi10')
    #fpeps = load_ftn_from_disc(f'../su')
    config = 0,2,1,2,0,2,1,0,1
    config = np.array([config])
else:
    fpeps = load_ftn_from_disc(f'psi{step}_0')
for tid,tsr in fpeps.tensor_map.items():
    #print(tid,tsr.phase)
    tsr.phase = {}
fpeps = scale_wfn(fpeps,2.)
af0 = FermionAmplitudeFactory2D(fpeps,model,from_plq=False,deterministic=deterministic,spinless=spinless)

nv = af0.nsite * 2
nh = (20,) * 3
afn = ('sin',) * 3
scale = np.pi,np.pi,4
bias = True 
af1 = SumFNN2D(Lx,Ly,nv,nh,afn,scale,bias=bias,log=False,fermion=True)
if step==0:
    eps = 1e-2
    for i,ni in enumerate(nh):
        af1.init((i,'w'),eps)
        if bias:
            af1.init((i,'b'),eps)
    af1.init((len(nh),'w'),eps)
    af1.save_to_disc(f'psi{step}_1')
else:
    af1.load_from_disc(f'psi{step}_1.hdf5')
af1.get_block_dict()
af1.model = model

af = SumAmplitudeFactory2D((af0,af1),fermion=True)
#af.check(config)
#exit()
#af = af0
#sampler = FermionDenseSampler(Lx*Ly,nelec,exact=True) 
sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:]) 

start = step 
stop = step + 100 

tnvmc = SR(sampler,normalize=True,solve_full=True,solve_dense=False,maxiter=200)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('nparam=',af.nparam)
    #print('shift=',shift)
tnvmc.tmpdir = './' 
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.batchsize = 5000
save_wfn = True 
tnvmc.run(start,stop,save_wfn=save_wfn)
