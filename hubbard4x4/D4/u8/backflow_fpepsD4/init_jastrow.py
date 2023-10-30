from quimb.tensor.fermion.fermion_2d_vmc import get_fpeps_z2,FermionAmplitudeFactory2D
from quimb.tensor.fermion.fermion_vmc import write_ftn_to_disc,load_ftn_from_disc
from quimb.tensor.tensor_vmc import scale_wfn
import numpy as np
import itertools
Lx,Ly = 4,4
D = 4 
chi = 8 

peps = get_fpeps_z2(Lx,Ly,D,eps=1e-2,normalize=False)
#for i,j in itertools.product(range(Lx),range(Ly)):
#    print(peps[i,j])
#    print(peps[i,j].data)
#    exit()
#peps = scale_wfn(peps,2.)

af = FermionAmplitudeFactory2D(peps,symmetry='z2',max_bond=chi)
rng = np.random.default_rng()

config = 0, 2, 1, 2, \
         2, 1, 2, 1, \
         1, 2, 1, 2, \
         2, 1, 0, 1
print('prob=',af.prob(config))
config = np.array(config)
for i in range(10):
    config_i = tuple(rng.permutation(config))
    print('prob=',af.prob(config_i),config_i)

config = 0, 2, 1, 2, \
         2, 1, 3, 0, \
         1, 2, 1, 2, \
         2, 1, 0, 1
print('prob=',af.prob(config))
config = np.array(config)
for i in range(10):
    config_i = tuple(rng.permutation(config))
    print('prob=',af.prob(config_i),config_i)
write_ftn_to_disc(peps,f'jastrowD{D}',provided_filename=True)
