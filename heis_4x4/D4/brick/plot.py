import matplotlib.pyplot as plt
import numpy as np
import h5py 
from scipy import optimize
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8)})
#plt.rcParams.update({'font.family':'serif'})
#plt.rcParams.update({'text.usetex':True})

exact = -0.57432544
fig,ax = plt.subplots(nrows=1,ncols=1)

out = open('summary.out', 'r').readlines()
e = []
for l in out:
    if l[:len('step=')]=="step=":
        ls = l.split(',')
        e.append(float(ls[1].split('=')[-1]))
e = np.array(e)
e = np.log10(np.fabs((e-exact)/exact)) 
ax.plot(range(len(e)),e,linestyle='-')
    
ax.set_xlabel('step')
ax.set_ylabel('log10(rel err)')
plt.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.99)
fig.savefig(f"heis_4x4_D4_brick.png", dpi=250)
