import numpy as np
import h5py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(precision=6,suppress=True,linewidth=1000)

N = 500
f = h5py.File(f'N{N}.hdf5','r')
A = f['A'][:]
b = f['b'][:]
f.close()
exact = -69.99723721325788 

sigma = 0.01
print('sigma=',sigma)
batchsize = 100 // SIZE
for run in range(batchsize):
#for run in range(1):
    e_old = 0
    niter = 0
    x = np.random.normal(loc=0,scale=1,size=N)
    while True: 
        Ax = np.dot(A,x)
        e = np.dot(x,b) + .5 * np.dot(x,Ax)
        de = np.fabs(e-e_old)
        print(f'\tniter={niter},e={e},de={de},rel_err={abs((e-exact)/exact)}')
        if de < 1e-6:
            break
        Ai = A + np.random.normal(loc=0,scale=sigma,size=(N,N))
        gi = b + Ax + np.random.normal(loc=0,scale=sigma,size=N)
        p = np.linalg.solve(Ai,gi)
        x -= p
        e_old = e
        niter += 1
    print(f'run={run+batchsize*RANK},niter={niter},rel_err={abs((e-exact)/exact)}')
