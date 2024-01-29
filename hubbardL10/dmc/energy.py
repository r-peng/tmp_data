from quimb.tensor.tensor_dmc import load,compute_expectation
import matplotlib.pyplot as plt
import itertools
import numpy as np
np.set_printoptions(suppress=True,precision=6,linewidth=1000)
step = 99
_,ws,we = load(f'step{step}.hdf5')
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8*3)})
shift = 0
dmrg = -5.788665537023 
L = 10

nmin = 10
neqs = 10,20,30,40
colors = [(r,g,b) for r,g,b in itertools.product((.3,.6),repeat=3)]
fig,(ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1)
for color,neq in zip(colors,neqs):
    Ls = list(range(1,len(ws)-neq-nmin))
    e = np.zeros(len(Ls)) 
    err = np.zeros(len(Ls)) 
    for i,L in enumerate(Ls):
        e[i],err[i] = compute_expectation(we,ws,L,eq=neq)
    e += shift
    rel_err = np.fabs((dmrg-e)/dmrg)
    ax1.plot(Ls,e/L, linestyle='-', color=color,label=f'neq={neq}')
    ax2.plot(Ls,rel_err, linestyle='-', color=color,label=f'neq={neq}')
    ax3.plot(Ls,err, linestyle='-', color=color,label=f'neq={neq}')
    print('neq=',neq)
    print(e/L)
    print(rel_err)

ax3.set_xlabel('step')
ax1.set_ylabel('E')
ax2.set_ylabel('relative error')
ax3.set_ylabel('statistical error')
ax2.set_yscale('log')
ax3.set_yscale('log')
#ax2.set_ylim((1e-3,2e-2))
ax1.legend()
plt.subplots_adjust(left=0.2, bottom=0.1, right=0.99, top=0.99)
fig.savefig(f"hubbard_L10Ne8_plain.png", dpi=250)


