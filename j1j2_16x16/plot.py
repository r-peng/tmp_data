import numpy as np
import scipy.stats

for D in range(4,9):
    for chi in range(2,7):
        out = open(f'D{D}/chi{chi}.out', 'r').readlines()
        energy,err=[],[]
        for l in out:
            if l[:5] == "step=":
                ls = l.split(',')
                energy.append(float(ls[1].split('=')[-1]))
                err.append(float(ls[3].split('=')[-1]))
        energy = np.array(energy)
        err = np.array(err)
        ix = np.argmin(energy)
        print(f'D={D},chi={chi}')        
        print('energy=',energy)
        print('err=',err)
        print('min=',energy[ix],err[ix]) 
        print()
