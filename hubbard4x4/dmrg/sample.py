#from pyblock3.hamiltonian import Hamiltonian
#from pyblock3.fcidump import FCIDUMP
#from pyblock3.algebra.mpe import MPE
import pickle
import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    FermionExchangeSampler2D,
    set_options,
)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

set_options(deterministic=True)
t = 1
u = 8
Lx = Ly = 4
n = Lx * Ly
nelec = 7,7

mps = pickle.load(open(f'./chi2000.npy','rb'))
for i in range(n):
    mps.tensors[i] = mps.tensors[i].to_sparse()
smps = mps.to_sliceable()

class DMRG:
    def __init__(self,mps):
        self.mps = mps
    def unsigned_amplitude(self,config):
        return smps.amplitude(config)
    def parse_config(self,config):
        return config
    def prob(self,config):
        return self.unsigned_amplitude(config)**2
af = DMRG(mps)
burn_in = 10
sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=burn_in)
sampler.af = af

config = np.array((2, 0, 2, 1, \
         1, 2, 1, 2, \
         2, 1, 2, 1, \
         1, 2, 1, 0))
for i in range(SIZE):
    sampler.config = tuple(sampler.rng.permutation(config))
sampler.px = np.log(sampler.af.prob(config))

total = 100
configs = []
print(f'RANK{RANK} burn in...')
for i in range(burn_in):
    sampler.sample()
    if RANK==0:
        print(i)
print(f'RANK{RANK} sampling...')
for i in range(total//SIZE):
    config,_ = sampler.sample()
    configs.append(config)
    if RANK==0:
        print(i)
np.save(f'{Lx}x{Ly}Ne{sum(nelec)}_1_RANK{RANK}.npy',np.array(configs))
