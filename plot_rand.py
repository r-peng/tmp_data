import matplotlib.pyplot as plt
import numpy as np
import h5py 
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8*2)})
#plt.rcParams.update({'font.family':'serif'})
#plt.rcParams.update({'text.usetex':True})

dmrg = -0.8413402262
fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1)
fnames = '../D4full_rbm_complex/rbm_only','relu_20_20s','relu_20_50s','relu_20_18','cnn_proj','cnn_mps'
colors = 'r','g','b','y','c','m','k'
labels = 'rbm50','relu20','relu50','relu18','cnn1','cnn2'
for fname,color,label in zip(fnames,colors,labels):
    f = h5py.File(fname+'/energy.hdf5','r')
    data = f['data'][:]
    f.close()
    print(len(data))
    ax1.plot(np.arange(len(data)),data, linestyle='-', color=color,label=label)
    ax2.plot(np.arange(len(data)),np.fabs((data-dmrg)/dmrg), linestyle='-', color=color)
   
ax2.set_xlabel('step')
ax1.set_ylabel('E')
ax2.set_ylabel('relative error')
ax2.set_yscale('log')
#ax2.set_ylim((.0001,.1))
#ax3.set_ylim((.01,1.))
#ax4.set_ylim((.0005,.02))
ax1.legend()
plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.99)
#plt.title('N=20,d=10')
#plt.show()
fig.savefig("hubbard3x3_fnn.png", dpi=250)
