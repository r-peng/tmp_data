import matplotlib.pyplot as plt
import numpy as np
import h5py 
from scipy import optimize
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8)})
#plt.rcParams.update({'font.family':'serif'})
#plt.rcParams.update({'text.usetex':True})

exact = -0.48373
ref = -0.48368
fig,ax = plt.subplots(nrows=1,ncols=1)

for tmpdir in ('peps','peps_rgn','brick','brick_rgn'):
    out = open(tmpdir+'/summary.out', 'r').readlines()
    e = []
    for l in out:
        if l[:len('step=')]=="step=":
            ls = l.split(',')
            e.append(float(ls[1].split('=')[-1]))
    e = np.array(e)
    e = np.log10(np.fabs((e-exact)/exact)) 
    ax.plot(range(len(e)),e,linestyle='-',label=tmpdir)

e = np.log10(np.fabs((ref - exact)/exact))
ax.plot(range(100),np.ones(100)*e,linestyle='-',color='k')
    
ax.set_xlabel('step')
ax.set_ylabel('log10(rel err)')
ax.legend()
plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.99)
fig.savefig(f"j1j2_8x8_D6.png", dpi=250)
