from quimb.tensor.vmc_model import (
    AmplitudeFactory,
    Sampler,
    ModelDenseSampler,
    make_matrices,
    sample_matrices,
    init,
)
import numpy as np
import itertools,h5py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

Ls = range(5,21,5)
x = np.load(f'x.npy')
for L in Ls:
    if RANK==0:
        print('L=',L)
        q0 = make_matrices(x[:L]) 
    tmpdir = f'exact_var/{L}_'
    try:
        f = h5py.File(tmpdir+f'covariance.hdf5','r')
        f.close()
    except FileNotFoundError:
        #sampler = ModelDenseSampler(L,exact=True,thresh=1e-50)
        #sampler.af = AmplitudeFactory(x[:L])
        sampler = Sampler(AmplitudeFactory(x[:L]),every=L)
        sample_size = 1000 
        tmpdir = None
        deltas_sr,deltas_rgn = sample_matrices(sampler,samplesize=sample_size,tmpdir=tmpdir,exact_variance=False) 
        if RANK>0:
            continue
        _,g,S,H = q0
        S += np.eye(L*2) * 1e-4 
        deltas = np.linalg.solve(S,g)
        print(np.linalg.norm(deltas-deltas_sr))
        deltas = np.linalg.solve(H+S/.5,g)
        print(np.linalg.norm(deltas-deltas_rgn))
   
#        exit()
#        
#        q1 = [None] * 4
#        f = h5py.File(tmpdir+f'step0.hdf5','r')
#        q1[0] = f['E'][:]
#        q1[1] = f['g'][:]
#        q1[2] = f['S'][:]
#        q1[3] = f['H'][:] - q1[0] * q1[2]
#        f.close()
#        for qi,qj in zip(q0,q1):
#            print(np.linalg.norm(qi-qj))

exit()
if RANK>0:
    exit()
dat = np.zeros((len(Ls),len(lab),3))
for ix,L in enumerate(Ls):
    tmpdir = f'exact_var/{L}_'
    f = h5py.File(tmpdir+f'var.hdf5','r')
    for ix2,name in enumerate(lab):
        qi = np.sqrt(f[name][:])
        if qi.size==1:
            dat[ix,ix2,:] = qi 
        else:
            dat[ix,ix2,:] = np.quantile(qi.flatten(),(.25,.5,.75))
    f.close()
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8*len(lab))})
fig,ax = plt.subplots(nrows=len(lab),ncols=1)
for ix,name in enumerate(lab):
    y = dat[:,ix]
    ax[ix].plot(Ls,y[:,1],linestyle='-',marker='o')
    ax[ix].fill_between(Ls,y[:,0],y[:,2],alpha=0.2)
    ax[ix].set_xlabel('L')
    #ax[ix].set_ylabel(name)
    ax[ix].set_ylabel(name[:-3]+'err')
plt.subplots_adjust(left=0.2, bottom=0.05, right=0.99, top=0.99)
fig.savefig(f"exact_var.png", dpi=250)
plt.close(fig)
exit()        




#Ls = 5,
runs = range(10)
#runs = (0,)
sample_size = 19000
burn_in = 100
every = 10

#init(200,f'x',eps=0.5)
#exit()
#print(x)
#exit()
#dir_ = 'tmp/'
dir_ = f'{sample_size}/'
for L in Ls:
    if RANK==0:
        print('L=',L)
    for run in runs:
        if RANK==0:
            print('run=',run)
        af = AmplitudeFactory(x[:L])
        sampler = Sampler(af,burn_in=burn_in,every=every)
        tmpdir = dir_+f'{L}_{run}_'
        try:
            f = h5py.File(tmpdir+f'step0.hdf5','r')
            f.close()
        except FileNotFoundError:
            sample_matrices(sampler,sample_size=sample_size,tmpdir=tmpdir) 
