import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory2D,
    ExchangeSampler2D,
)
from quimb.tensor.tensor_vmc import (
    Progbar,
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
model = J1J2(J1,J2,Lx,Ly)

psi0 = load_tn_from_disc(f'../peps/psi460')
#config = np.load('../peps/config459.npy')
af0 = AmplitudeFactory2D(psi0,model,max_bond=chi)

try:
    config = np.load(f'RANK{RANK}.npy')
except FileNotFoundError:
    sampler = ExchangeSampler2D(Lx,Ly)
    sampler.af = af0
    sampler.config = tuple(config[RANK%config.shape[0],:])
    sampler._burn_in(burn_in=0)
    batchsize = 5000
    if RANK==0:
        print('generating configs from a=0...')
        pg = Progbar(total=batchsize)
    config = [None] * batchsize
    for i in range(batchsize):
        config[i],_ = sampler.sample()
        if RANK==0:
            pg.update()
    np.save(f'RANK{RANK}.npy',np.array(config))

def amplitude(config,af):
    cx = np.zeros(config.shape[0])
    if RANK==0:
        pg = Progbar(total=config.shape[0])
    for i in range(config.shape[0]):
        cx[i] = af.amplitude(tuple(config[i]))
        if RANK==0:
            pg.update()
    if RANK==0:
        pg.close()
    return cx

if RANK==0:
    print('computing sign for a=0...')
try:
    cx0 = np.load(f'sign0_{RANK}.npy')
except FileNotFoundError:
    cx0 = amplitude(config,af0)
    np.save(f'sign0_{RANK}.npy',cx0)

for a in (1,2,4,8,16):
    if RANK==0:
        print(f'computing sign for a={a}...')
    try:
        cx1 = np.load(f'sign{a}_{RANK}.npy')
    except FileNotFoundError:
        psi1 = load_tn_from_disc(f'../a{a}/psi500_0')
        af1 = AmplitudeFactory2D(psi1,model,max_bond=chi)
        cx1 = amplitude(config,af1)
        np.save(f'sign{a}_{RANK}.npy',cx1)
    sign = np.sign(cx0*cx1)
    pos = len(sign[sign>0])
    neg = len(sign[sign<0])
    sign = np.array([pos,neg])
    sign_all = np.zeros_like(sign)
    COMM.Reduce(sign,sign_all,root=0)
    if RANK==0:
        print(sign_all)
exit()

nv = af0.nsite  
nh = nv * 8 
af1 = RBM2D(Lx,Ly,nv,nh)
if step==0:
    eps = 1e-2
    for ix in range(3):
        af1.init(ix,eps)
else:
    af1.load_from_disc(f'psi{step}_1')
af1.model = model
af1.fermion = False
af1.input_format = (-1,1),None
af1.get_block_dict()

af = ProductAmplitudeFactory2D((af0,af1),fermion=False)
#for ix in range(config.shape[0]):
#    af.check(af.parse_config(tuple(config[ix])))
#exit()

burn_in = 40

start = step 
stop = 500

tnvmc = SR(sampler,normalize=False,solve_full=True,solve_dense=False)
tnvmc.tmpdir = './' 
if RANK==0:
    print('SIZE=',SIZE)
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.batchsize = int(5e3 + .1)
tnvmc.run(start,stop,save_wfn=True)
