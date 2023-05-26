import matplotlib.pyplot as plt
import numpy as np
import h5py 
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8*3)})
#plt.rcParams.update({'font.family':'serif'})
#plt.rcParams.update({'text.usetex':True})

fig,(ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1)
f = h5py.File('energy.hdf5','r')
data = f['data'][:]
f.close()
dE = np.array([data[ix+1]-data[ix] for ix in range(len(data)-1)])
ax1.plot(np.arange(len(data)),data, linestyle='', marker='o')
ax2.plot(np.arange(1,len(data)),np.fabs(dE), linestyle='', marker='o')

#f = h5py.File('gradient.hdf5','r')
#data = f['data'][:]
#f.close()
#ax3.plot(np.arange(start,start+len(data)),data, linestyle='', marker='o', color=color,label=label)

f = h5py.File('err.hdf5','r')
data = f['data'][:]
f.close()
ax3.plot(np.arange(len(data)),data, linestyle='', marker='o')

ax3.set_xlabel('step')
ax1.set_ylabel('E')
ax2.set_ylabel('dE')
ax3.set_ylabel('statistical error')
ax2.set_yscale('log')
ax3.set_yscale('log')
#ax2.set_ylim((.0001,.1))
#ax3.set_ylim((.01,1.))
#ax4.set_ylim((.0005,.02))
plt.subplots_adjust(left=0.2, bottom=0.05, right=0.99, top=0.99)
fig.savefig("J1J2D8chi4.png", dpi=250)
