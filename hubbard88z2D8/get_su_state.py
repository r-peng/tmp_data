import numpy as np
from quimb.tensor.fermion.fermion_2d_vmc import (
    get_product_state,
    flat2site,
    ExchangeSampler,
    write_ftn_to_disc,load_ftn_from_disc,
    set_options,
)
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D 
from quimb.tensor.fermion.fermion_2d_tebd import SimpleUpdate 

import itertools
Lx,Ly = 8,8
nelec = 28,28
t = 1.
u = 8.
#D = 20 
D = 8 
chi = 2
mu = -1.9
print('mu=',mu)
data_map = set_options(symmetry='z2',flat=True,deterministic=False,max_bond=chi)
ham = Hubbard2D(t,u,Lx,Ly,mu=mu,symmetry='z2')
#exit()

#load_fname = None
#load_fname = f'old/sr_6e4/psi100'
load_fname = f'./tmpdir/su_{Lx},{Ly}_D20_mu1.9'
if load_fname is None:
    config = 2, 0, 2, 1, 2, 1, 2, 1, \
             1, 2, 1, 0, 1, 2, 1, 2, \
             2, 1, 2, 1, 2, 0, 2, 1, \
             1, 2, 1, 2, 1, 2, 1, 0, \
             2, 1, 2, 0, 2, 1, 2, 1, \
             1, 2, 1, 2, 1, 2, 1, 2, \
             0, 1, 2, 1, 2, 1, 2, 1, \
             1, 0, 1, 2, 1, 2, 0, 2, \
    #sampler_opts = {'Lx':Lx,'Ly':Ly,'nelec':nelec}
    #exchange_sampler = ExchangeSampler2D(sampler_opts)
    #config = exchange_sampler.rand_config()
    print(config)
    ixa = []
    ixb = []
    for ix,i in enumerate(config):
        if i in (1,3):
            ixa.append(ix)
        if i in (2,3):
            ixb.append(ix)
    spin_map = {'a':[flat2site(ix,Lx,Ly) for ix in ixa],'b':[flat2site(ix,Lx,Ly) for ix in ixb]}
    print(spin_map)
    fpeps = get_product_state(Lx,Ly,spin_map)
else:
    fpeps = load_ftn_from_disc(load_fname)

su = SimpleUpdate(fpeps,ham,D=D,chi=chi,compute_energy_final=False)
su.evolve(steps=50,tau=1e-5,progbar=True)
write_ftn_to_disc(su.state,f'./tmpdir/su_{Lx},{Ly}_D{D}_mu1.9',provided_filename=True)
#print(fpeps[0,0].data)
#exit()

#norm = fpeps.make_norm()
#norm = norm.contract()
#print('norm=',norm)

#energy = fpeps.compute_local_expectation(ham.terms,normalized=True) 
#print('fpeps energy=',energy)

#ann_a = cre_a.dagger
#ann_b = cre_b.dagger
#na = np.tensordot(cre_a,ann_a,axes=([1],[0]))
#nb = np.tensordot(cre_b,ann_b,axes=([1],[0]))
#terms = {(i,j):na.copy() for i,j in itertools.product(range(Lx),range(Ly))}
#Na = fpeps.compute_local_expectation(terms,normalized=True) 
#terms = {(i,j):nb.copy() for i,j in itertools.product(range(Lx),range(Ly))}
#Nb = fpeps.compute_local_expectation(terms,normalized=True) 
#print('Na,Nb=',Na,Nb)
