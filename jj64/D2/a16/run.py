import numpy as np
from quimb.tensor.product_2d_vmc import ProductAmplitudeFactory2D
from quimb.tensor.nn_2d import RBM2D
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory2D,
    ExchangeSampler2D,
)
from quimb.tensor.tensor_vmc import (
    SR,
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
    psi = load_tn_from_disc(f'../peps/psi60')
    config = np.load(f'../peps/config59.npy')
else:
    psi = load_tn_from_disc(f'psi{step}_0')
    config = np.load(f'config{step-1}.npy')
scale = 1.1
#psi = scale_wfn(psi,scale)
af0 = AmplitudeFactory2D(psi,model,max_bond=chi)

nv = af0.nsite  
nh = nv * 16 
af1 = RBM2D(Lx,Ly,nv,nh)
if step==0:
    eps = 1e-2
    for ix in range(3):
        af1.init(ix,eps)
else:
    a,b,w = af1.load_from_disc(f'psi{step}_1.hdf5')
af1.model = model
af1.fermion = False
af1.input_format = (-1,1),None
af1.get_block_dict()

af = ProductAmplitudeFactory2D((af0,af1),fermion=False)
#for ix in range(config.shape[0]):
#    af.check(af.parse_config(tuple(config[ix])))
#exit()

burn_in = 40
sampler = ExchangeSampler2D(Lx,Ly,burn_in=burn_in)
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:])

start = step 
stop = step + 400

tnvmc = SR(sampler,normalize=False,solve_full=True,solve_dense=False)
tnvmc.tmpdir = './' 
if RANK==0:
    print('SIZE=',SIZE)
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.batchsize = int(5e3 + .1)
tnvmc.run(start,stop,save_wfn=True)
