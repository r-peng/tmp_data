import matplotlib.pyplot as plt
import numpy as np
import h5py 
from scipy import optimize
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8*2)})
#plt.rcParams.update({'font.family':'serif'})
#plt.rcParams.update({'text.usetex':True})

old = True
#old = False
if old:
    fnames = {
    'sr2e3':('g',':',None,None),
    'sr4e3':('b',':',None,None),
    'sr5e3':('orange',':',None,None),
    'sr6e3':('pink',':',None,None),
    'sr8e3':('y',':',None,None),
    'sr16e3':('c',':',None,None),
    'sr32e3':('gray',':',None,None),
    'rgn_2e3_1e-3_0.5':('g','-','2e3',None),
    'rgn_4e3_1e-3_0.5':('b','-','4e3',25),
    'rgn_5e3_1e-3_0.5':('orange','-','5e3',25),
    'rgn_6e3_1e-3_0.5':('pink','-','6e3',25),
    'rgn_8e3_1e-3_0.5':('y','-','8e3',30),
    'rgn_16e3_1e-3_0.5':('c','-','16e3',30),
    'rgn_32e3_1e-3_0.5':('gray','-','32e3',35),
    }
else:
    fnames = {
    'sr4e3':('b',':',None),
    'rgn_1e3_1e-1_1.0':('r','-','1e3'),
    'rgn_2e3_1e-1_1.0':('g','-','2e3'),
    'rgn_4e3_1e-1_1.0':('b','-','4e3'),
    }

fig,ax = plt.subplots(nrows=2,ncols=1)
for fname,(color,linestyle,label,mk) in fnames.items():
    print(fname)

    out = open(fname+'/out.out', 'r').readlines()
    x = []
    e = []
    err = []
    for l in out:
        if l[:len('step=')]=="step=":
            ls = l.split(',')
            x.append(float(ls[0].split('=')[-1]))
            e.append(float(ls[1].split('=')[-1]))
            err.append(float(ls[3].split('=')[-1]))
    x = x
    e = np.array(e)
    #ax[0].plot(x,e, linestyle=linestyle, color=color,label=label)
    y = np.log10(np.fabs((-.375-e)/-.375))
    ax[0].plot(x,y, linestyle=linestyle, color=color,label=label)
    err = np.log10(np.array(err)/.375)
    err = y - err
    ax[1].plot(x,err, linestyle=linestyle, color=color,label=label)
    if mk is not None:
        ax[0].plot([0,x[mk]],[y[mk],y[mk]], linestyle='--', color=color)
        ax[1].plot([0,x[mk]],[err[mk],err[mk]], linestyle='--',color=color)
    
ax[0].set_xlabel('step')
ax[1].set_xlabel('step')
#ax[2].set_xlabel('step')
#ax[0].set_ylabel('e')
ax[0].set_ylabel('log10(rel err)')
ax[1].set_ylabel('log10(snr)')
ax[0].set_xlim((0,60))
ax[1].set_xlim((0,60))
ax[0].set_ylim((-6.,-1.))
ax[1].set_ylim((-.5,2.5))
#ax3.set_ylim((.01,1.))
#ax4.set_ylim((.0005,.02))
plt.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.99)
#plt.title('N=20,d=10')
ax[0].legend()
#ax[1].legend()
#plt.show()
if old:
    fig.savefig(f"1d_100.png", dpi=250)
    #fig.savefig(f"L100.png", dpi=250)
else:
    fig.savefig(f"L100_new.png", dpi=250)
