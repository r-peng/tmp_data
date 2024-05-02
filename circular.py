from pyblock3.algebra.fermion_encoding import get_state_map
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.fermion import SparseFermionTensor
from quimb.tensor.fermion.fermion_core import FermionTensor,FermionTensorNetwork
import numpy as np
symm = 'u1'
state_map = get_state_map(symm)

def get_tsr():
    blks = []
    for s1,s2 in ((0,1),(1,0)):
        q1,ix1,d1 = state_map[s1]
        q2,ix2,d2 = state_map[s2]
        qlabs = q1,q2
        array = np.random.rand(d1,d2) * 2 - 1
        blks.append(SubTensor(reduced=array,q_labels=qlabs))
    return SparseFermionTensor(blocks=blks,pattern='+-')
A = get_tsr()
B = get_tsr()
C = get_tsr()
AB = np.tensordot(A,B,axis=[(1,),(0,)])
ABC = np.tensordot(AB,C,axis=[(1,0),(0,1)]) 
print(ABC)


A = FermionTensor(data=A,inds=('i','j'),tags='A')
B = FermionTensor(data=B,inds=('j','k'),tags='B')
C = FermionTensor(data=C,inds=('k','i'),tags='C')

tn1 = FermionTensorNetwork([A,B,C])
tn2 = FermionTensorNetwork([B,C,A])
tn2 = FermionTensorNetwork([C,A,B])
print(tn1.contract())
print(tn2.contract())
print(tn3.contract())
