
def _remove(T,bd):
    idx = T.inds.index(bd)
    if idx==0:
        T.modify(data=T.data[0],inds=T.inds[1:])
        return T
    if idx==len(T.inds)-1:
        T.modify(data=T.data[...,0],inds=T.inds[:-1])
        return T
    data = np.zeros(T.shape[idx])
    data[0] = 1
    data = np.tensordot(T.data,data,axes=([idx],[0]))
    inds = T.inds[:idx] + T.inds[idx+1:]
    T.modify(data=data,inds=inds)
def _add(T,bd):
    data = np.tensordot(np.ones(1),T.data,axes=0)
    inds = (bd,) + T.inds
    T.modify(data=data,inds=inds)
def convert(peps,typ='v',fname=None):
    if typ=='v':
        irange = range(peps.Lx-1)
        jrange = range(peps.Ly)
        def get_next(i,j):
            return i+1,j
    else:
        irange = range(peps.Lx)
        jrange = range(peps.Ly-1)
        def get_next(i,j):
            return i,j+1
    for i,j in itertools.product(irange,jrange):
        if (i+j)%2==0:
            continue
        i2,j2 = get_next(i,j)
        T1,T2 = peps[i,j],peps[i2,j2]
        bonds = tuple(T1.bonds(T2))
        if len(bonds)==0:
            bd = f'I{i},{j}_I{i2},{j2}' 
            _add(T1,bd)
            _add(T2,bd)
        elif len(bonds)==1:
            bd = bonds[0]
            _remove(T1,bd)
            _remove(T2,bd)
        else:
            raise ValueError
    if fname is None:
        return peps
    import matplotlib.pyplot as plt
    #plt.rcParams.update({'font.size':16})
    fig,ax = plt.subplots(nrows=1,ncols=1)
    fix = {peps.site_tag(i,j):(i,j) for i,j in itertools.product(range(peps.Lx),range(peps.Ly))}
    peps.draw(show_inds=False,show_tags=False,fix=fix,ax=ax)
    #fig.subplots_adjust(left=0.15, bottom=0.15, right=0.99, top=0.95)
    fig.savefig(fname)
    return peps 
