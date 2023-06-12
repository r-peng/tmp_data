import numpy as np
import h5py
for step in range(0,10):
    f = h5py.File(f'step{step}','r')
    print(f.keys())
    #H = f['H'][:] 
    #S = f['S'][:] 
    #E = f['E'][:] 
    #g = f['g'][:] 
    #deltas = f['deltas'][:] 
    f.close()
    
    hess = H - E[0] * S
    w = np.linalg.eigvals(hess)
    w = w.real
    print(np.amin(w),np.amax(w))
