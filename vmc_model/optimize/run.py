from quimb.tensor.vmc_model import (
    AmplitudeFactory,
    Sampler,
    DenseSampler,
    init,
)
from quimb.tensor.tensor_vmc_red import SR,RGN

import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

L = 200
samplesize = 2000 
burn_in = 40
every = 10
#rgn = False
rgn = True

#init(1000,'x',eps=0.5)
#exit()

x1 = np.load('../../../data/x.npy')
x2 = np.load('../../x.npy')
x = np.concatenate([x1,x2])[:L]
#if RANK==0:
#    print(x)
af = AmplitudeFactory(x)
sampler = Sampler(af,burn_in=burn_in,every=every)
sampler.config = tuple([0] * L)

if rgn:
    myvmc = RGN(sampler,normalize=False,pure_newton=False) 
    maxiter = 20
else:
    myvmc = SR(sampler,normalize=False) 
    maxiter = 50
myvmc.save_config = False
#myvmc.tmpdir = f'L{L}_{samplesize}/'
myvmc.tmpdir = f'./'
myvmc.batchsize = samplesize
myvmc.cond1 = 1e-3
myvmc.cond2 = 1e-3
myvmc.rate1 = .1
myvmc.rate2 = .5
myvmc.progbar = True
thresh = 1e-6
if RANK==0:
    #print(x)
    print('rgn=',rgn)
    print('L=',x.shape[0])
    print('sample size=',samplesize)
myvmc.run(0,maxiter,save_wfn=False)
if RANK==0:
    print(myvmc.sampler.af.x)
