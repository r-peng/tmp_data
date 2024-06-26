import numpy as np
from quimb.tensor.product_2d_vmc import (
    ProductAmplitudeFactory2D,
    RBM2D,SIGN2D,
)
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory2D,
    ExchangeSampler2D,
)
from quimb.tensor.tensor_vmc import (
    TNVMC,
    scale_wfn,
    write_tn_to_disc,load_tn_from_disc,
)
import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 6,4
nspin = 12,12
D = 2 
J1 = 1.
J2 = 0.5
chi = 8 
step = 0 
model = J1J2(J1,J2,Lx,Ly)

if step==0:
    psi = load_tn_from_disc(f'../fpeps/psi40')
    config = np.load(f'../fpeps/config39.npy')
else:
    psi = load_tn_from_disc(f'psi{step}_0')
    config = np.load(f'config{step-1}.npy')
scale = 1.1
#psi = scale_wfn(psi,scale)
af0 = AmplitudeFactory2D(psi,max_bond=chi)
af0.model = model

nv = af0.nsite  
nh = 50 
af1 = RBM2D(Lx,Ly,nv,nh)
if step==0:
    eps = 1e-2
    a,b,w = af1.init(eps,fname=f'psi{step}_1')
else:
    a,b,w = af1.load_from_disc(f'psi{step}_1.hdf5')
af1.model = model

nl = 1 
af2 = SIGN2D(Lx,Ly,nv,nl)
if step==0:
    w,b = af2.init(nv,1.,a=0,b=1,fname=f'psi{step}_2')
else:
    w,b = af2.load_from_disc(f'psi{step}_2.hdf5')
af2.get_block_dict(w,b)
af2.model = model
af2.coeff = 1.

af = ProductAmplitudeFactory2D((af0,af1,af2))
#af = ProductAmplitudeFactory2D((af0,af1))
#af = af0
burn_in = 40
sampler = ExchangeSampler2D(Lx,Ly,burn_in=burn_in)
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:])
#print(sampler.config)

start = step 
stop = step + 400

tnvmc = TNVMC(sampler,normalize=True,optimizer='sr',solve_full=True,solve_dense=False)
tnvmc.tmpdir = './' 
if RANK==0:
    print('SIZE=',SIZE)
    print('nparam=',af.nparam)
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.batchsize = int(5e3 + .1)
tnvmc.check = 'energy'
tnvmc.run(start,stop)
