import numpy as np
from quimb.tensor.fermion.product_2d_vmc import (
    set_options,
    ProductAmplitudeFactory2D,
    FNN2D,
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
Lx,Ly = 16,4
nelec = 56
nelec = (nelec//2,) * 2
t = 1.
u = 8.
D = 10
step = 11 
chi = 40 
set_options(symmetry='u1',flat=True,deterministic=False)
model = Hubbard(t,u,Lx,Ly,sep=True)

if step==0:
    fpeps = load_ftn_from_disc(f'../D{D}/psi116')
    if RANK==0:
        print(fpeps)
    for tid,tsr in fpeps.tensor_map.items():
        #print(tid,tsr.phase)
        tsr.phase = {}
    config = np.load(f'../D{D}/config115.npy')
else:
    fpeps = load_ftn_from_disc(f'psi{step}_0')
    config = np.load(f'config{step-1}.npy')
scale = 1. 
fpeps = scale_wfn(fpeps,scale)
af0 = FermionAmplitudeFactory2D(fpeps,max_bond=chi)
af0.model = model 
af0.is_tn = True
af0.spin = None

to_spin = False
nl = 4 
nn = (af0.nsite*2,) if to_spin else (af0.nsite,)
nn = nn + (500,500) + nn 
af1 = FNN2D(Lx,Ly,nl,to_spin=to_spin)
if step==0:
    eps = 1e-3
    w = []
    b = []
    for i in range(nl-1):
        wi = (np.random.rand(nn[i],nn[i+1])*.5 - 1.)*eps
        COMM.Bcast(wi,root=0)
        w.append(wi)

        bi = (np.random.rand(nn[i+1])*.5 - 1.)*eps
        COMM.Bcast(bi,root=0)
        b.append(bi)
    wi = np.ones(nn[nl-1])
    w.append(wi)
    af1.save_to_disc(w,b,f'psi{step}_1')
else:
    w,b = af1.load_from_disc(f'psi{step}_1.hdf5')
af1.get_block_dict(w,b)
af1.model = model

af = ProductAmplitudeFactory2D((af0,af1))
sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=40)
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:])

start = step 
stop = step + 10 

tnvmc = TNVMC(sampler,normalize=True,optimizer='sr',pure_newton=True,solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
    print('nparam=',af.nparam)
    print('D=',D)
    print('chi=',chi)
tnvmc.tmpdir = './' 
#tmpdir = None
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.check = 'energy'
tnvmc.batchsize = int(5e4+1e-6) 
tnvmc.run(start,stop)
