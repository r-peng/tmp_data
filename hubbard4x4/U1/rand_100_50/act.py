import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(4.8,4.8*2)})

for step in (1,2):
    print('\nstep=',step)
    for i in range(3):
        print('layer=',i)
        amap = dict()
        for RANK in range(30): 
            with open(f'psi{step}_1l{i}RANK{RANK}.pkl','rb') as f:
                amap_ = pickle.load(f)
                for key,cset in amap_.items():
                    if key not in amap:
                        amap[key] = set()
                    amap[key].update(cset)
        n1 = []
        n2 = []
        for key,cset in amap.items():
            n1.append(len(cset))
            n2.append(len(key))
        print('number of pattern=',len(amap))
        print('nconfig=',sum(n1))
    
        fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1)
        ax1.bar(range(len(n1)), n1)
        ax2.bar(range(len(n2)), n2)
        ax1.set_ylabel('nconfig')
        ax2.set_ylabel('nnode')
        ax1.set_yscale('log')
        plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.99)
        fig.savefig(f"step{step}layer{i}.png", dpi=250)
