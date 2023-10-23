import numpy as np
from quimb.tensor.fermion.fermion_2d_gauss import SpinlessGaussianFPEPS 
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D 
from quimb.tensor.fermion.fermion_vmc import load_ftn_from_disc,write_ftn_to_disc
from quimb.tensor.tensor_vmc import scale_wfn 
import itertools,sys,h5py
np.set_printoptions(suppress=True,threshold=sys.maxsize,linewidth=1000)

t = 1.

Lx,Ly = 4,4
nelec = 7,7 
M = 2 
gf = SpinlessGaussianFPEPS(Lx,Ly,M,nelec[0])
sites = None
sites = 9, 5, 3, 7, 1, 6, 12 
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
gf.set_ham(h1)
fname = f'Lx{Lx}Ly{Ly}N{nelec[0]}M{M}'
#gf.fname = fname
#gf.run(maxiter=100)
#exit()

x = None
if fname is not None:
    f = h5py.File(fname+'.hdf5','r')
    x = f['x'][:]
    f.close()
#gf.fname = fname
#gf.run(x0=x,maxiter=100)
#exit()

symmetry = 'u1'
flat = True
fpeps = gf.get_fpeps(x=x,symmetry=symmetry,flat=flat) 
fpeps = scale_wfn(fpeps,10.)
norm = fpeps.make_norm()
n = norm.contract()
print('norm=',n)
#exit()
from pyblock3.algebra.fermion_ops import H1
h1 = H1(h=-t,symmetry=symmetry,flat=flat,spinless=True)
terms = dict()
for i,j in itertools.product(range(Lx),range(Ly)):
    if i+1<Lx:
        terms[(i,j),(i+1,j)] = h1.copy()
    if j+1<Ly:
        terms[(i,j),(i,j+1)] = h1.copy()
e = fpeps.compute_local_expectation(terms,normalized=True)
print('check energy=',e)
write_ftn_to_disc(fpeps,fname,provided_filename=True)

