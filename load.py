import numpy as np
import itertools

from quimb.tensor.fermion.fermion_core import FermionTensor,FermionTensorNetwork
from quimb.tensor.fermion.fermion_2d import FPEPS
from quimb.tensor.fermion.fermion_vmc import write_ftn_to_disc 
from pyblock3.algebra.fermion_symmetry import U1
from pyblock3.algebra.fermion import FlatFermionTensor 

Lx,Ly = 4,4
path ='peps_npy_out_4x4_h0.125_D8_E0.621/' 

tsrs = dict()
for (i,j) in itertools.product(range(Lx),range(Ly)):
    print(i,j)
    # pattern
    pattern = ''
    if j>0:
        pattern += '+' 
    if i>0:
        pattern += '+'
    if j<Ly-1:
        pattern += '-'
    if i<Lx-1:
        pattern += '-'
    pattern += '+'
    print(pattern)
    sign = np.array([{'+':1,'-':-1}[c] for c in pattern]) 

    extra_leg = [] 
    dq = []
    blk_dict = dict()
    sh_dict = dict()
    for p in range(4):
        # load
        data = np.load(path+f'tensor-{i}-{j}-{p}-data.npy')
        qlab = np.load(path+f'tensor-{i}-{j}-{p}-q_labels.npy')
        shape = np.load(path+f'tensor-{i}-{j}-{p}-shapes.npy')
        #print(qlab)
        #print(shape)

        # remove extra leg
        extra_leg.append(qlab[:,-1])
        qlab = qlab[:,:-1]
        shape = shape[:,:-1]

        # parse blocks
        sec = np.cumsum(np.prod(shape,axis=1))[:-1]
        data = np.split(data,sec)
        assert len(data)==qlab.shape[0]
        assert len(data)==shape.shape[0]
        for k in range(len(data)):
            qlab_k = tuple([U1(n).to_flat() for n in qlab[k,:]])
            sh_k = tuple(shape[k,:])
            dq.append(np.dot(qlab[k,:],sign))  

            data_k = np.reshape(data[k],sh_k,order='F')
            if p in (0,3):
                if qlab_k not in blk_dict:
                    sh_dict[qlab_k] = sh_k 
                    blk_dict[qlab_k] = np.zeros(sh_k)
                blk_dict[qlab_k] += data_k
            else:
                if qlab_k not in blk_dict:
                    sh_dict[qlab_k] = sh_k[:-1]+(2,)
                    blk_dict[qlab_k] = np.zeros(sh_k[:-1]+(2,))
                blk_dict[qlab_k][...,p-1] += data_k[...,0]
    extra_leg = np.concatenate(extra_leg)
    assert np.linalg.norm(extra_leg-extra_leg[0])<1e-6

    data = []
    qlab = []
    shape = []
    for qlab_k,data_k in blk_dict.items():
        data.append(data_k.flatten())
        qlab.append(np.array(qlab_k))
        shape.append(np.array(sh_dict[qlab_k]))
    data = np.concatenate(data)
    qlab = np.stack(qlab,axis=0)
    shape = np.stack(shape,axis=0)

    tsrs[i,j] = FlatFermionTensor(qlab,shape,data,pattern=pattern,symmetry=U1) 
    print(tsrs[i,j].dq)
    for dqi in dq:
        try: 
            assert U1(dqi)==tsrs[i,j].dq 
        except:
            print(U1(dqi),tsrs[i,j].dq) 
    print(tsrs[i,j])
#exit()

tn = FermionTensorNetwork([])
xrange = range(Lx)
yrange = range(Ly-1,-1,-1)
scheme = 'cr'
if scheme=='rc':
    order = [(i,j) for i in xrange for j in yrange]
else:
    order = [(i,j) for j in yrange for i in xrange]
for i,j in order:
    inds = []
    if j>0:
        inds.append(f'I{i},{j-1}_I{i},{j}')
    if i<Lx-1:
        inds.append(f'I{i},{j}_I{i+1},{j}')
    if j<Ly-1:
        inds.append(f'I{i},{j}_I{i},{j+1}')
    if i>0:
        inds.append(f'I{i-1},{j}_I{i},{j}')
    inds.append(f'k{i},{j}')
    T = FermionTensor(data=tsrs[Lx-1-i,j],inds=inds,tags=(f'I{i},{j}',f'ROW{i}',f'COL{j}'))
    tn.add_tensor(T,virtual=True)
    _,site = T.get_fermion_info() 
    print((i,j),site)
print(tn)
tn.view_as_(FPEPS,inplace=True,
            site_tag_id='I{},{}',
            row_tag_id='ROW{}',
            col_tag_id='COL{}',
            Lx=Lx,
            Ly=Ly,
            site_ind_id='k{},{}')
tn.reorder('row',inplace=True)
write_ftn_to_disc(tn,f'4x4D8_{scheme}',provided_filename=True)
