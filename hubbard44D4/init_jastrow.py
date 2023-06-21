from quimb.tensor.tensor_2d_vmc_ import (
    get_product_state,scale_wfn,
    write_tn_to_disc,load_tn_from_disc,
    AmplitudeFactory,
    set_options,
)
from quimb.tensor.fermion.fermion_2d_vmc_subspace import get_gutzwiller 
import numpy as np
Lx,Ly = 4,4
D = 4 
chi = 16 
set_options(max_bond=chi)

peps = get_product_state(Lx,Ly,bdim=D,pdim=4,eps=.01)
#peps = get_gutzwiller(Lx,Ly,bdim=D,eps=.01,g=0.1)
peps = scale_wfn(peps,.125)

amp_fac = AmplitudeFactory(peps,phys_dim=4)
rng = np.random.default_rng()

config = 2, 0, 1, 2, \
         0, 1, 2, 1, \
         1, 2, 1, 2, \
         2, 1, 2, 1
print('prob=',amp_fac.prob(config))
config = np.array(config)
for i in range(10):
    config_i = tuple(rng.permutation(config))
    print('prob=',amp_fac.prob(config_i),config_i)

config = 2, 0, 1, 2, \
         0, 1, 2, 1, \
         1, 2, 0, 3, \
         2, 1, 2, 1
print('prob=',amp_fac.prob(config))
config = np.array(config)
for i in range(10):
    config_i = tuple(rng.permutation(config))
    print('prob=',amp_fac.prob(config_i),config_i)

config = 2, 0, 1, 2, \
         0, 1, 2, 1, \
         0, 2, 0, 3, \
         3, 1, 2, 1
print('prob=',amp_fac.prob(config))
config = np.array(config)
for i in range(10):
    config_i = tuple(rng.permutation(config))
    print('prob=',amp_fac.prob(config_i),config_i)
#write_tn_to_disc(peps,f'./tmpdir/su_{Lx},{Ly}_D{D}_jastrow',provided_filename=True)
