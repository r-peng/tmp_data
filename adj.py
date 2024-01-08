import numpy as np
Lx = 3
Ly = 3
nelec = 3,3
def flatten(i,j):
    return i*Lx + j 
def flat2site(ix):
    return ix//Ly,ix%Ly
def adj(config):
    print(config)
    config = np.array(config).reshape((Lx,Ly))
    print(config)

rng = np.random.default_rng()
site = rng.chose
