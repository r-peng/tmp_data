import numpy as np
from quimb.tensor.product_2d_vmc import ProductAmplitudeFactory2D
from quimb.tensor.product_vmc import config_to_ab
from quimb.tensor.fermion.product_2d_vmc import PEPSJastrow,BackFlow2D
from quimb.tensor.fermion.fermion_2d_vmc import Hubbard,FermionExchangeSampler2D
from quimb.tensor.tensor_vmc import (
    TNVMC,scale_wfn,
    write_tn_to_disc,load_tn_from_disc,
)
import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 4,4
nelec = 7,7
t = 1.
u = 8.
D = 2 
chi = 8 
step = 1200 
model = Hubbard(t,u,Lx,Ly)
mo = np.load(f'{Lx}x{Ly}Ne{sum(nelec)}_mo.npy')

af = [None] * 3
if RANK==0: 
    h1 = model.get_h()
    dm = [np.dot(mo[s,:,:nelec[s]],mo[s,:,:nelec[s]].T) for s in (0,1)]
    e = np.sum(h1*(dm[0]+dm[1]))
    e += u * np.sum(np.diag(dm[0])*np.diag(dm[1])) 
    print('ehf=',e,e/(Lx*Ly))


if step==0:
    psi = load_tn_from_disc(f'jastrowD{D}')
    config = 2, 0, 2, 1, \
         1, 2, 1, 2, \
         2, 1, 2, 1, \
         1, 2, 1, 0
    if RANK==0:
        p = 1
        for ix,conf in enumerate(config_to_ab(config)):
            det = np.where(np.array(conf))
            det = np.linalg.det(mo[ix,det,:nelec[ix]])
            print('det=',np.log(np.abs(det)),conf)
            p *= det 
        p = p**2
        print(p,np.log(p))
    config = np.array([config]) 
else:
    psi = load_tn_from_disc(f'psi{step}_0')
    config = np.load(f'config{step-1}.npy')
scale = 1.
#psi = scale_wfn(psi,scale)
af[0] = PEPSJastrow(psi,phys_dim=4,max_bond=chi)
af[0].model = model
af[0].spin = None
af[0].spinless = False

nv = af[0].nsite * 2 
nl = 3
nn = (af[0].nsite,) * nl
for ix,spin in zip((0,1),('a','b')):
    af[ix+1] = BackFlow2D(Lx,Ly,mo[ix,:,:nelec[ix]],nv,nl,spin,afn='silu')
    if step==0:
        eps = 0.
        w,b = af[ix+1].init(nn,eps,fname=f'psi{step}_1')
    else:
        w,b = af[ix+1].load_from_disc(f'psi{step}_1.hdf5')
    af[ix+1].get_block_dict(w,b)
    af[ix+1].model = model

af = ProductAmplitudeFactory2D(af,fermion=True)
burn_in = 40
sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=burn_in,scheme='random')
sampler.af = af
sampler.config = tuple(config[RANK%config.shape[0],:])

start = step 
stop = step + 400 

tnvmc = TNVMC(sampler,normalize=True,optimizer='sr',solve_full=True,solve_dense=False)
tnvmc.tmpdir = './' 
if RANK==0:
    print('SIZE=',SIZE)
    print('nparam=',af.nparam)
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.batchsize = int(5e4 + .1)
tnvmc.check = 'energy'
tnvmc.run(start,stop)
