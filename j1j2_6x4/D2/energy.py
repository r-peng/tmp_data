import numpy as np
import matplotlib.pyplot as plt
from quimb.tensor.tensor_dmc import load,compute_expectation
np.set_printoptions(suppress=True,precision=6,threshold=int(1e5+.1))
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8*3)})

import itertools,h5py
nsite = 24 
shift = -0.4685
kps = 10,20,30,40,50,60
ns = (199,) * 6 
colors = 'r','g','b','y','c','m'
nmin = 10
eq = 50
fig,(ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1)
for kp,n,color in zip(kps,ns,colors):
    fname = f'kp{kp}/step{n}.hdf5'
    _,f,ws,we = load(fname)
    print('kp=',kp)
    Ls = [] 
    e1 = []
    e2 = []
    for L in range(5,len(f),5):
        if len(f)-eq-L-1>nmin:
            Ls.append(L)
            e1.append(compute_expectation(we,ws,f,L,eq=eq,method=1))
            e1[-1] = e1[-1]/nsite+shift

            e2.append(compute_expectation(we,ws,f,L,eq=eq,method=2))
            e2[-1] /= nsite
            e2[-1][:,0] += shift
            print(L,e1[-1],e2[-1][:,0],e2[-1][:,1])
    Ls = np.array(Ls)
    e1 = np.array(e1)
    e2 = np.array(e2)
    ax1.plot(Ls,e1[:,0], linestyle='-', color=color,label=f'kp{kp}')
    ax1.plot(Ls,e1[:,1], linestyle=':', color=color)
    
    ax2.plot(Ls,e2[:,0,0], linestyle='-', color=color)
    ax2.plot(Ls,e2[:,1,0], linestyle=':', color=color)

    ax3.plot(Ls,e2[:,0,1], linestyle='-', color=color)
    ax3.plot(Ls,e2[:,1,1], linestyle=':', color=color)

ax3.set_xlabel('L')
ax1.set_ylabel('E1')
ax2.set_ylabel('E2')
ax3.set_ylabel('stat err')
ax3.set_yscale('log')
ax1.legend()
plt.subplots_adjust(left=0.25, bottom=0.05, right=0.99, top=0.99)
fig.savefig(f"j1j2_6x4_dmc0_{eq}_{nmin}.png", dpi=250)
