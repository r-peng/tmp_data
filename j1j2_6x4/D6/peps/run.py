from quimb.tensor.tensor_vmc import write_tn_to_disc,load_tn_from_disc,SR
from quimb.tensor.tensor_2d_vmc import J1J2,AmplitudeFactory2D,ExchangeSampler2D,get_brickwork_from_peps
import itertools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

D = 6 
chi = 6 

J1 = 1.
J2 = .5 

Lx = 6
Ly = 4

start = 3 
stop = 20
if start==0:
    #psi = load_tn_from_disc(f'../pepsD{D}')
    #psi = get_brickwork_from_peps(psi)
    #write_tn_to_disc(psi,f'../brickD{D}')
    psi = load_tn_from_disc(f'../pepsD{D}')
    config = [(i+j)%2 for i,j in itertools.product(range(Lx),range(Ly))]
else:
    psi = load_tn_from_disc(f'psi{start}')
    config = np.load(f'config{start-1}.npy')
    config = config[RANK%config.shape[0]]

model = J1J2(J1,J2,Lx,Ly)
sampler = ExchangeSampler2D(Lx,Ly,burn_in=20)
sampler.af = AmplitudeFactory2D(psi,model,max_bond=chi)
sampler.config = tuple(config)

vmc = SR(sampler,solve_full=True,solve_dense=False)
vmc.tmpdir = './'
vmc.batchsize = 5000
vmc.rate1 = 0.1
vmc.cond1 = 1e-3
vmc.run(start,stop,save_wfn=True)
