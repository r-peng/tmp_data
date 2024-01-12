import numpy as np
#from quimb.tensor.product_2d_vmc import RBM2D

Lx = 3
Ly = 3
nsite  = Lx * Ly
nelec = 3,3

#nv = 12 * 2 + 1 
#nh = 10
#af = RBM2D(Lx,Ly,nv,nh,input_format='bond')

print(2**(-nsite))
print(4**(-nsite))
exit()
for cf in range(10):
    rng = np.random.default_rng()
    config = [None] * 2
    flt = [None] * 2
    for ix,ne in enumerate(nelec):
        site = rng.choice(nsite,size=ne,replace=False) 
        config[ix] = np.zeros(nsite,dtype=int) 
        config[ix][site] = 1
        #print(site,config[ix])
        flt[ix] = sum([ci*2**(i-nsite) for i,ci in enumerate(config[ix])]) 
    print(cf,config,flt)
exit()
config = config[0] + config[1] * 2
print(af.bmap)
v = af.input_bond(config)
print(v)
