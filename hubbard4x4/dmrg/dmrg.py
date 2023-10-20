from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mpe import MPE
t = 1
u = 8
Lx = Ly = 4
n = Lx * Ly
nelec = 7,7

def flat2site(ix):
    return ix // Ly, ix % Ly
def flatten(i,j):
    return i * Ly + j
def build_hubbard(cutoff=1E-9):
    fcidump = FCIDUMP(pg='c1', n_sites=n, n_elec=sum(nelec), twos=0, ipg=0, orb_sym=[0] * n)
    hamil = Hamiltonian(fcidump, flat=True)

    def generate_terms(n_sites, c, d):
        for ix1 in range(0, n_sites):
            i,j = flat2site(ix1)
            if i + 1 < Lx:
                ix2 = flatten(i+1,j)
                for s in [0, 1]:
                    yield -t * c[ix1, s] * d[ix2, s]
                    yield -t * c[ix2, s] * d[ix1, s]
            if j + 1 < Ly:
                ix2 = flatten(i,j+1)
                for s in [0, 1]:
                    yield -t * c[ix1, s] * d[ix2, s]
                    yield -t * c[ix2, s] * d[ix1, s]
            yield u * (c[ix1, 0] * c[ix1, 1] * d[ix1, 1] * d[ix1, 0])

    return hamil, hamil.build_mpo(generate_terms, cutoff=cutoff)

hamil,mpo = build_hubbard()
bond_dim = 200
mpo, _ = mpo.compress(cutoff=1E-9, norm_cutoff=1E-9)
print('MPO (compressed) = ', mpo.show_bond_dims())
mps = hamil.build_mps(bond_dim)

dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0], dav_thrds=[1E-4], iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print("Energy = %20.12f" % ener)
print('energy per site=',ener / n)
