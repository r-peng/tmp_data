import numpy as np
from quimb.tensor.fermion.fermion_gauss import SpinlessGaussianFPEPS2
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D,SimpleUpdate 
from quimb.tensor.fermion.fermion_2d_vmc_ import write_ftn_to_disc,load_ftn_from_disc
from quimb.tensor.fermion.fermion_2d_vmc_subspace import config_to_ab 
import itertools,sys,h5py,torch
np.set_printoptions(suppress=True,threshold=sys.maxsize,linewidth=1000)

t = 1.
u = 8.

Lx,Ly = 4,4
nelec = 7,7 
M = 2 
gf = SpinlessGaussianFPEPS2(Lx,Ly,M,nelec[0],occ_b=2)
sites = 13,9,7,6,12,3,8
gf.get_Q(sites=sites)

#ehf = -7.654423401177489
h1 = np.zeros((gf.nsite,)*2)
for i,j in itertools.product(range(Lx),range(Ly)):
    ix1 = gf.site_map[i,j]['site_ix'] 
    if i+1<Lx:
        ix2 = gf.site_map[i+1,j]['site_ix']
        h1[ix1,ix2] = -t
        h1[ix2,ix1] = -t
    if j+1<Ly:
        ix2 = gf.site_map[i,j+1]['site_ix']
        h1[ix1,ix2] = -t
        h1[ix2,ix1] = -t
gf.set_ham(h1)
config = 1,0,1,2,\
         2,1,2,1,\
         1,2,1,2,\
         2,1,2,0,
#config,_ = config_to_ab(config)
#x0 = gf.init(config,eps=.1)
x0 = None
gf.fname = 'init'
#x = gf.run(x0=x0,maxiter=500)
#exit()

f = h5py.File(gf.fname+'.hdf5','r')
x = f['x'][:]
f.close()
symmetry = 'u1'
fpeps = gf.get_fpeps(x,symmetry=symmetry)

#ham0 = Hubbard2D(0.,0.,Lx,Ly,symmetry='u1',spinless=True)
ham1 = Hubbard2D(t,0.,Lx,Ly,symmetry='u1',spinless=True)
su = SimpleUpdate(fpeps,ham1,D=2**M,compute_energy_final=False)
su.print_conv = False 
su.evolve(steps=50,tau=1e-6,progbar=True)
fpeps = su.state
write_ftn_to_disc(fpeps,gf.fname,provided_filename=True)

chi = 200 
energy = fpeps.compute_local_expectation(ham1.terms,normalized=True,max_bond=chi) 
print('fpeps energy=',energy)
