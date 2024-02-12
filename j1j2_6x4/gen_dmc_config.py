import numpy as np
from quimb.tensor.tensor_dmc import (
        Walker,Sampler,Hubbard1D,ConstWFN1D,
        Progbar,
)

config = np.load(f'../peps/config459.npy') # 639 configs
config = np.concatenate([config.copy() for _ in range(10)],axis=0)
np.save(f'init_config{config.shape[0]}.npy',config)
print(config.shape)
exit()

L = 10
nelec = 4,4
u = 8
model = Hubbard1D(u,L)

psi = ConstWFN1D(model)
tau = 0.01
wk = Walker(psi,tau)
wk.kp = 10
wk.gamma = 0
#wk.shift = -5.788665537023 

sampler = Sampler(wk)
sampler.progbar = True

M = 50000
#configs = np.zeros((M,L),dtype=int) 
#pg = Progbar(total=M)
#for i in range(M):
#    config = np.zeros((2,L),dtype=int)
#    for ix in range(2):
#        idx = sampler.rng.choice(L,size=nelec[ix],replace=False,shuffle=False)
#        config[ix,idx] = 1
#    configs[i] = config[0] + config[1] * 2
#    pg.update()
#np.save(f'init_config{M}.npy',configs)
#exit()

configs = np.load(f'init_config{M}.npy')
sampler.init(configs)

start = 0
stop = 1000
tmpdir = './'
sampler.run(start,stop,tmpdir=tmpdir)
