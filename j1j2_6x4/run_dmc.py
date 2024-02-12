import numpy as np
from quimb.tensor.tensor_2d_vmc import J1J2,AmplitudeFactory2D
from quimb.tensor.tensor_dmc import Walker,Sampler
from quimb.tensor.tensor_vmc import (
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
psi = load_tn_from_disc(f'../../peps/psi460')
scale = 1.1
#psi = scale_wfn(psi,scale)
af = AmplitudeFactory2D(psi,model,max_bond=chi,from_plq=False)
af.dmc = True

tau = 0.1
wk = Walker(af,tau)

sampler = Sampler(wk)
sampler.progbar = True
sampler.clear_every = 10
start = 0
stop = 1000
if start==0:
    config = np.load(f'../init_config6390.npy')
    batchsize = config.shape[0] // SIZE
    sampler.init(config[RANK*batchsize:(RANK+1)*batchsize])
else:
    sampler.load(f'step{start-1}.hdf5')

tmpdir = './'
sampler.run(start,stop,tmpdir=tmpdir)
