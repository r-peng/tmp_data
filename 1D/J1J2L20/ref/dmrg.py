from quimb.tensor.tensor_vmc import write_tn_to_disc, load_tn_from_disc, get_gate2
from quimb.tensor.tensor_1d_vmc import compute_energy, build_mpo
from quimb.tensor import MPS_rand_state
L = 20
mps = load_tn_from_disc('su')
#D = 5 
#mps = MPS_rand_state(L,D,phys_dim=2,normalize=True,cyclic=False)
#write_tn_to_disc(mps,'rand')

J1 = 1.
J2 = .5
terms = dict()
gate = get_gate2(1.,to_matrix=True)
order = 'b1,k1,b2,k2'
for d,J in zip((1,2),(J1,J2)):
    for i in range(L):
        if i+d<L:
            terms[i,i+d] = J * gate 
e = compute_energy(mps,terms,order)
print('energy from plq =', e)

mpo = build_mpo(L,terms,order)
norm = mps.make_norm()
n = norm.contract(tags=all)
for i in range(L):
    norm[mps.site_tag(i),'BRA'].reindex_({mps.site_ind(i):mpo.upper_ind(i)})
    norm[mps.site_tag(i),'KET'].reindex_({mps.site_ind(i):mpo.lower_ind(i)})
e = norm
e.add_tensor_network(mpo)
e = e.contract(tags=all)
print('energy from mpo =',e / n,n)

from quimb.tensor.tensor_dmrg import DMRG2
bond_dim = 50 
dmrg = DMRG2(mpo,bond_dims=bond_dim,p0=mps)

# generate initial state for VMC
#dmrg.solve(tol=1e-4,verbosity=1,max_sweeps=1)
#print(dmrg.state)
#write_tn_to_disc(dmrg.state,'init')

# converge DMRG
dmrg.solve(tol=1e-4,verbosity=1)
print(dmrg.state)
write_tn_to_disc(dmrg.state,'dmrg')

