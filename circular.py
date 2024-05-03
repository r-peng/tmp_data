from pyblock3.algebra.fermion_ops import vaccum,bonded_vaccum,creation,Hubbard 
from quimb.tensor.fermion.fermion_core import FermionTensor,FermionTensorNetwork
import numpy as np
################### setting up bonded tn #######################################
symm = 'u11'
vac = bonded_vaccum((1,)*2,'+-+',symmetry=symm,flat=False)

cre_a = creation(spin='a',symmetry=symm,flat=False)
cre_b = creation(spin='b',symmetry=symm,flat=False)
dat_a = np.tensordot(vac,cre_a,axes=[(2,),(1,)]) 
dat_b = np.tensordot(vac,cre_b,axes=[(2,),(1,)]) 
dat_ab = np.tensordot(dat_a,cre_b,axes=[(2,),(1,)])
vac1 = vaccum(n=1,symmetry=symm,flat=False)
state_a = np.tensordot(cre_a,vac1,axes=[(1,),(0,)])
state_b = np.tensordot(cre_b,vac1,axes=[(1,),(0,)])
state_ab = np.tensordot(cre_b,state_a,axes=[(1,),(0,)])

A = FermionTensor(data=vac.copy(),inds=('i','j','a'),tags='A')
B = FermionTensor(data=dat_a.copy(),inds=('j','k','b'),tags='B')
C = FermionTensor(data=dat_b.copy(),inds=('k','l','c'),tags='C')
D = FermionTensor(data=dat_ab.copy(),inds=('l','i','d'),tags='D')

tn = FermionTensorNetwork([D,C,B,A])
print(tn)
gate_AB = Hubbard(t=np.random.rand(),u=np.random.rand(),symmetry=symm,flat=False) 
gate_BC = Hubbard(t=np.random.rand(),u=np.random.rand(),symmetry=symm,flat=False) 
gate_CD = Hubbard(t=np.random.rand(),u=np.random.rand(),symmetry=symm,flat=False) 
gate_DA = Hubbard(t=np.random.rand(),u=np.random.rand(),symmetry=symm,flat=False) 
for _ in range(2):
    tn.gate_inds_(gate_AB,inds=('a','b'),contract='split')
    tn.gate_inds_(gate_BC,inds=('b','c'),contract='split')
    tn.gate_inds_(gate_CD,inds=('c','d'),contract='split')
    tn.gate_inds_(gate_DA,inds=('d','a'),contract='split')
print(tn)
A = tn['A']
B = tn['B']
C = tn['C']
D = tn['D']
print(A._phase,A.get_fermion_info()[1])
print(B._phase,B.get_fermion_info()[1])
print(C._phase,C.get_fermion_info()[1])
print(D._phase,D.get_fermion_info()[1])

################## reordering ##################################
bra1 = FermionTensorNetwork([
    FermionTensor(data=vac1.copy(),inds=('a',)),
    FermionTensor(data=state_ab.copy(),inds=('b',)),
    FermionTensor(data=vac1.copy(),inds=('c',)),
    FermionTensor(data=state_ab.copy(),inds=('d',))])
bra2 = FermionTensorNetwork([
    FermionTensor(data=state_a.copy(),inds=('a',)),
    FermionTensor(data=vac1.copy(),inds=('b',)),
    FermionTensor(data=state_b.copy(),inds=('c',)),
    FermionTensor(data=state_ab.copy(),inds=('d',))])
tn0 = tn.copy()
tn1 = FermionTensorNetwork([D,C,B,A])
tn2 = FermionTensorNetwork([C,B,A,D])
tn3 = FermionTensorNetwork([B,A,D,C])
tn4 = FermionTensorNetwork([A,D,C,B])
for bra in [bra1,bra2]:
    print()
    bra = bra.H
    for tn in [tn0,tn1,tn2,tn3,tn4]:
        tn = tn.copy()
        tn.add_tensor_network(bra)
        print(tn.contract())
