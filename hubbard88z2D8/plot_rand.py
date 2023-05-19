import matplotlib.pyplot as plt
import numpy as np
import h5py 
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4*2,4.8*2)})
#plt.rcParams.update({'font.family':'serif'})
#plt.rcParams.update({'text.usetex':True})

fnames = {
    'sr_6e4':('SR_6e4','b',0),
    'rgn_6e4':('RGN','c',40),
    'rgn_26':('RGN','y',26),
    'rgn_30':('RGN','r',30),
    'rgn_50':('RGN','g',50),
}
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
for fname,(label,color,start) in fnames.items():
    print(fname)

    f = h5py.File(fname+'/energy.hdf5','r')
    data = f['data'][:]
    f.close()
    dE = np.array([data[ix+1]-data[ix] for ix in range(len(data)-1)])
    ax1.plot(np.arange(start,start+len(data)),data, linestyle='', marker='o', color=color,label=label)
    ax2.plot(np.arange(start+1,start+len(data)),np.fabs(dE), linestyle='', marker='o', color=color,label=label)

    f = h5py.File(fname+'/gradient.hdf5','r')
    data = f['data'][:]
    f.close()
    ax3.plot(np.arange(start,start+len(data)),data, linestyle='', marker='o', color=color,label=label)

    f = h5py.File(fname+'/err.hdf5','r')
    data = f['data'][:]
    f.close()
    ax4.plot(np.arange(start,start+len(data)),data, linestyle='', marker='o', color=color,label=label)

ax3.set_xlabel('step')
ax4.set_xlabel('step')
ax1.set_ylabel('E')
ax2.set_ylabel('dE')
ax3.set_ylabel('g')
ax4.set_ylabel('statistical error')
ax2.set_yscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
#ax2.set_ylim((.0001,.1))
#ax3.set_ylim((.01,1.))
#ax4.set_ylim((.0005,.02))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.99)
#plt.title('N=20,d=10')
ax1.legend()
#plt.show()
fig.savefig("hubbard_Lx8Ly8D8.png", dpi=250)
