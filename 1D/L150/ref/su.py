from quimb.tensor import MPS_rand_state
from quimb.tensor.tensor_1d_tebd import LocalHam1D,TEBD
from quimb.tensor.tensor_vmc import load_tn_from_disc, write_tn_to_disc, get_gate2
from quimb.tensor.tensor_1d_vmc import compute_energy
L = 150
bond_dim = 5
mps = MPS_rand_state(L,bond_dim,phys_dim=2,normalize=True,cyclic=False)

J1 = 1.
J2 = .5
terms = dict()
gate = get_gate2(1.,to_bk=True,to_matrix=True)
order = 'b1,k1,b2,k2'
for d,J in zip((1,2),(J1,J2)):
#for d,J in [(1,J1)]:
    for i in range(L):
        if i+d<L:
            terms[i,i+d] = J * gate 
#e = compute_energy(mps,terms,order)
#print('init energy=', e)

ham = LocalHam1D(L,terms)
su = TEBD(mps,ham,imag=True,split_opts={'max_bond':bond_dim})
su.update_to(20.,dt=1e-1,progbar=True)
psi = su.pt

#e = compute_energy(psi,terms,order)
#print('final energy=', e)
write_tn_to_disc(psi,'su')
