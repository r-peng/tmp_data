import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    PerturbedAmplitudeFactory2D,
    ExchangeSampler2D,
)
from quimb.tensor.tensor_vmc import (
    SR,
    DenseSampler,
    write_tn_to_disc,load_tn_from_disc,
)
from quimb.tensor.nn_2d import GaugeNN2D,init_gauge_tanh
import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 4,4
nspin = 8,8
D = 2
J1 = 1.
J2 = 0.5
model = J1J2(J1,J2,Lx,Ly)

step = 0
if step==0:
    peps = load_tn_from_disc(f'../rgn_6e4/psi50')
    af = init_gauge_tanh(Lx,Ly,D,scale=.5,const=1,log=False,eps=1e-5) 
nn = GaugeNN2D(af,peps)
af = PerturbedAmplitudeFactory2D(peps,nn,model)

burn_in = 40
sampler = ExchangeSampler2D(Lx,Ly,burn_in=burn_in) 
#sampler = DenseSampler(Lx,Ly,nspin,exact=True) 
sampler.af = af
sampler.config = 1,0,1,0,\
               0,1,0,1,\
               1,0,1,0,\
               0,1,0,1

tnvmc = SR(sampler,solve_full=True,solve_dense=False)
tnvmc.tmpdir = './' 
start = 0 
stop = 50 
tnvmc.rate2 = .5
tnvmc.cond2 = 1e-3
tnvmc.rate1 = .1
tnvmc.cond1 = 1e-3
tnvmc.check = 'energy'
tnvmc.progbar = True

tnvmc.batchsize = 5000 
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
    print('rate=',tnvmc.rate1,tnvmc.rate2)
    print('cond=',tnvmc.cond1,tnvmc.cond2)
tnvmc.run(start,stop,save_wfn=True)
