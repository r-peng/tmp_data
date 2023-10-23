from quimb.tensor.fermion.fermion_2d_vmc_subspace import (
    get_gutzwiller,
    set_options,
    JastrowAmplitudeFactory,
)
from quimb.tensor.tensor_vmc import (
    scale_wfn,
    write_tn_to_disc,load_tn_from_disc,
)
import numpy as np
Lx,Ly = 4,4
D = 2 
chi = 8 
set_options()

coeffs = np.ones(4)
peps = get_gutzwiller(Lx,Ly,coeffs,bdim=D,eps=1e-3,normalize=True)
peps = scale_wfn(peps,1.5)

amp_fac = JastrowAmplitudeFactory(peps,max_bond=chi)
rng = np.random.default_rng()

config = 0, 2, 1, 2, \
         2, 1, 2, 1, \
         1, 2, 1, 2, \
         2, 1, 0, 1
print('prob=',amp_fac.prob(config))
config = np.array(config)
for i in range(10):
    config_i = tuple(rng.permutation(config))
    print('prob=',amp_fac.prob(config_i),config_i)

config = 0, 2, 1, 2, \
         2, 1, 3, 0, \
         1, 2, 1, 2, \
         2, 1, 0, 1
print('prob=',amp_fac.prob(config))
config = np.array(config)
for i in range(10):
    config_i = tuple(rng.permutation(config))
    print('prob=',amp_fac.prob(config_i),config_i)
write_tn_to_disc(peps,f'jastrowD{D}',provided_filename=True)
