import numpy as np
from quimb.tensor.fermion.fermion_2d_gauss import SpinlessGaussianFPEPS 
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D 
from quimb.tensor.fermion.fermion_vmc import write_ftn_to_disc,load_ftn_from_disc
from quimb.tensor.tensor_vmc import scale_wfn 
import itertools,sys,h5py,torch
np.set_printoptions(suppress=True,threshold=sys.maxsize,linewidth=1000)

t = 1.
u = 0.

Lx,Ly = 3,3
nelec = 3,3
M = 2 
gf = SpinlessGaussianFPEPS(Lx,Ly,M,nelec[0])
#sites = None
sites = []
gf.get_Q(sites=sites)

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
# solve quadratic
gf.set_ham(h1)
fname = f'Lx{Lx}Ly{Ly}N{nelec[0]}M{M}_a'
#gf.fname = fname
#gf.run(maxiter=100)
#exit()

f = h5py.File(fname+'.hdf5','r')
x = f['x'][:]
f.close()

fpeps = gf.get_fpeps(x=x,symmetry='u1') 
scale = 4.
fpeps = scale_wfn(fpeps,scale)
ham = Hubbard2D(t,0.,Lx,Ly,symmetry='u1',flat=True,spinless=True)
chi = None
exps = fpeps.compute_local_expectation(ham.terms,max_bond=chi,normalized=True,return_all=True)
print('norm=',[n for _,n in exps.values()])
e = sum([e/n for e,n in exps.values()])
print('e=',e,e/(Lx * Ly))
write_ftn_to_disc(fpeps,fname,provided_filename=True)
