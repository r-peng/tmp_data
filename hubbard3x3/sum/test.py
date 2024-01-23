import numpy as np
from quimb.tensor.nn_core import relu_init_normal
from quimb.tensor.nn_2d import Fourier2D
from quimb.tensor.product_2d_vmc import SumAmplitudeFactory2D
from quimb.tensor.fermion.fermion_2d_vmc import (
    Hubbard2D,
    FermionExchangeSampler2D,
    FermionAmplitudeFactory2D,
)
from quimb.tensor.fermion.fermion_vmc import (
    write_ftn_to_disc,load_ftn_from_disc,
    FermionDenseSampler,
)
from quimb.tensor.tensor_vmc import (
    SR,
    scale_wfn,
    write_tn_to_disc,load_tn_from_disc,
)
np.set_printoptions(suppress=True,precision=6)

import itertools,h5py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 3,3
nelec = 3,3
t = 1.
u = 8.
step = 0 
model = Hubbard2D(t,u,Lx,Ly,sep=True)

af = []
if step==0:
    fpeps = load_ftn_from_disc(f'../D4full/psi20')
else:
    fpeps = load_ftn_from_disc(f'psi{step}_0')
for tid,tsr in fpeps.tensor_map.items():
    #print(tid,tsr.phase)
    tsr.phase = {}
af1 = FermionAmplitudeFactory2D(fpeps,model,from_plq=False)

nx = af1.nsite * 2
ny = 10
af2 = Fourier2D(Lx,Ly,nx,ny)
if step==0:
    eps = 1e-4
    w = np.zeros((2,nx,ny))
    b = np.zeros((2,ny))
    w[0],b[0] = relu_init_normal(nx,ny,-1,1)
    w[1],b[1] = relu_init_normal(nx,ny,-1,1)
    w[0] *= eps
    b[0] *= eps
    af2.params[0] = w
    af2.params[1] = b 
af2.const = 0
af2.log = False
af2.phase = False
af2.fermion = True
af2.input_format = (-1,1),'F'
af2.get_block_dict()
af2.model = model

af = SumAmplitudeFactory2D((af1,af2),fermion=True)
x = af.get_x()
gx = np.zeros_like(x)

#sampler = FermionDenseSampler(Lx*Ly,nelec,exact=True) 
sampler = FermionExchangeSampler2D(Lx,Ly,burn_in=0)
sampler.af = af
sampler.config = 2,1,0,0,1,2,2,1,0 
sampler._burn_in()
nsample = 50
eps = 1e-5
thresh = eps * 10
size,remain = af.nparam // SIZE,af.nparam % SIZE
if RANK<remain:
    size += 1
    start = RANK * size
else:
    start = remain * (size+1) + (RANK-remain) * size
for i in range(nsample):
    af.update(x)
    config,_ = sampler.sample()
    config = np.array(config)
    COMM.Bcast(config,root=0)
    config = tuple(config)

    _,_,vx,_,_ = af.compute_local_energy(config)
    config = af.parse_config(config)
    cx = af.amplitude(config)
    
    gxi = np.zeros_like(vx) 
    for j in range(start,start+size):
        x_new = x.copy()
        x_new[j] += eps
        af.update(x_new)
        cx1 = af.amplitude(config)

        gxi[j] = (cx1 - cx) / eps
    COMM.Reduce(gxi,gx,root=0)
    if RANK==0:
        gx /= cx
        error = np.linalg.norm(vx-gx)/np.linalg.norm(vx)
        print(i,error)
        if error>thresh:
        #if True:
            vx1,vx2 = vx[:538],vx[538:]
            gx1,gx2 = gx[:538],gx[538:]
            error1 = np.linalg.norm(vx1-gx1)/np.linalg.norm(vx1)
            error2 = np.linalg.norm(vx2-gx2)/np.linalg.norm(vx2)
            print(config[0],error1,error2)
            #print(vx2)
            #print(gx2)
            #exit()
        #exit()
exit()

start = step 
stop = step + 100

tnvmc = SR(sampler,solve_full=True,solve_dense=False)
if RANK==0:
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('nelec=',nelec)
tnvmc.tmpdir = './' 
tnvmc.rate1 = .1
tnvmc.rate2 = .5
tnvmc.cond1 = 1e-3
tnvmc.cond2 = 1e-3
tnvmc.check = 'energy'
tnvmc.batchsize = 3000
tnvmc.progbar = True
tnvmc.run(start,stop,save_wfn=True)
