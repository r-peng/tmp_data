import numpy as np
import itertools
import scipy.linalg
from quimb.utils import progbar as Progbar
np.set_printoptions(suppress=True,precision=6,linewidth=1000)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

H = np.array([[0., 1.], [1., 0.]])
def wf(x):
    return x/np.linalg.norm(x)
def deriv_wf2(x,eps=1e-4):
    g= np.zeros([2, 2])
    g[:,0] = (wf(x+np.array([eps,0.]))-wf(x))/eps
    g[:,1] = (wf(x+np.array([0.,eps]))-wf(x))/eps
    return g
def deriv_wf(x):
    norm = np.linalg.norm(x)
    xhat = x/norm
    return (np.eye(2) - np.outer(xhat,xhat))/ norm 
def make_matrices(x):
    L = len(x)
    ovlp = np.zeros((L,2,L,2))
    g = [deriv_wf(xi) for xi in x]
    #g = [deriv_wf2(xi) for xi in x]
    xg = [np.dot(xi,gi) for gi,xi in zip(g,x)]
    for i in range(L):
        ovlp[i,:,i,:] = np.dot(g[i].T,g[i])
        for j in range(i+1,L):
            ovlp_ij = np.outer(xg[i],xg[j])
            ovlp[i,:,j,:] = ovlp_ij 
            ovlp[j,:,i,:] = ovlp_ij.T

    hess = np.zeros((L,2,L,2))
    xHg = [np.dot(np.dot(xi,H),gi) for gi,xi in zip(g,x)]
    e = [np.dot(np.dot(xi,H),xi) for xi in x] 
    E = sum(e)
    for i in range(L):
        hess[i,:,i,:] = np.dot(g[i].T,np.dot(H,g[i]))
        hess[i,:,i,:] += (E-e[i]) * ovlp[i,:,i,:] 
        for j in range(i+1,L):
            hess_ij = np.outer(xHg[i],xg[j])   
            hess_ij += np.outer(xg[i],xHg[j])   
            hess_ij += (E-e[i]-e[j]) * ovlp[i,:,j,:]
            hess[i,:,j,:] = hess_ij
            hess[j,:,i,:] = hess_ij.T
    return ovlp.reshape(L*2,L*2), hess.reshape(L*2,L*2)
class AmplitudeFactory:
    def __init__(self,x):
        self.x = x
        self.Hx = [np.dot(H,xi) for xi in x]
        self.nsite = len(x)
        self.nparam = 2 * self.nsite
    def get_x(self):
        return np.array(self.x).flatten()
    def log_prob(self,config):
        return np.log(np.array([xi[ci] for xi,ci in zip(self.x,config)])**2).sum()
    def compute_local_energy(self,config):
        #cx = np.array([xi[ci] for xi,ci in zip(self.x,config)]).prod()
        ex = sum([Hxi[ci]/xi[ci] for Hxi,xi,ci in zip(self.Hx,self.x,config)])
        vx = np.zeros((self.nsite,2))
        for i,(xi,ci) in enumerate(zip(self.x,config)):
            vx[i,ci] = 1/xi[ci]
        vx = vx.flatten()

        Hvx = np.zeros((self.nsite,2))
        for i,(Hxi,xi,ci) in enumerate(zip(self.Hx,self.x,config)):
            dat = np.zeros(2)
            dat[ci] = 1
            Hvx[i] = np.dot(H,dat)/xi[ci]
            Hvx += (ex-Hxi[ci]/xi[ci]) * dat / xi[ci]
        Hx = Hx.flatten()
        return ex,vx,Hvx
class Sampler:
    def __init__(self,af,burn_in=0,seed=None,every=1):
        self.af = af
        self.L = self.af.nsite
        self.burn_in = burn_in
        self.rng = np.random.default_rng(seed)
        self.every = every 
    def sample(self):
        for _ in range(self.every):
            step = self.rng.choice([-1,1])
            sweep = range(self.L) if step==1 else range(self.L-1,-1,-1)
            for i in sweep:
                i_old = self.config[i]
                i_new = 1-i_old
                p = self.af.x[i]**2
                acceptance = p[i_new]/p[i_old]
                if acceptance < self.rng.uniform(): # reject
                    continue

                config_new = list(self.config)
                config_new[i] = 1-self.config[i]
                self.config = tuple(config_new)
        return self.config
class VMC:
    def __init__(self):
        return
    def sample(self,sampler,samplesize,tmpdir,progbar=False):
        nparam = sampler.af.nparam
        # burn in
        for i in range(sampler.burn_in):
            sampler.sample()

        # sample
        batchsize = sample_size // SIZE
        sample_size = batchsize * SIZE
        self.e = np.zeros(batchsize)
        self.v = np.zeros((batchsize,nparam))
        self.Hv = np.zeros((batchsize,nparam))
        if RANK==0 and probar:
            pg = Progbar(total=batchsize) 
        for i in range(batchsize):
            config = sampler.sample()
            self.e[i],self.v[i],self.Hv[i] = sampler.af.compute_local_energy(config)
            if RANK==0 and probar:
                pg.update() 

        # collect 
        if RANK>0:
            COMM.Send(self.e,dest=0,tag=0)
            COMM.Send(self.v,dest=0,tag=1)
            COMM.Send(self.Hv,dest=0,tag=2)
            return
        e = [self.e.copy()] 
        v = [self.v.copy()]
        Hv = [self.Hv.copy()]
        for worker in range(1,SIZE):
            COMM.Recv(self.e,source=worker,tag=0)
            COMM.Recv(self.v,source=worker,tag=1)
            COMM.Recv(self.Hv,source=worker,tag=2)
            e.append(self.e.copy())
            v.append(self.v.copy())
            Hv.append(self.Hv.copy())
        e = np.concatenate(e)
        v = np.concatenate(v,axis=0)
        Hv = np.concatenate(Hv,axis=0)
        np.save(tmpdir+f'e.npy',self.e)
        np.save(tmpdir+f'v.npy',self.v)
        np.save(tmpdir+f'Hv.npy',self.Hv)
    def check_matrices(self,tmpdir,x):
        e = np.load(tmpdir+'e.npy')
        v = np.load(tmpdir+'v.npy')
        Hv = np.load(tmpdir+'Hv.npy')
        samplesize = len(e)

        emean = e.sum()/samplesize
        vmean = np.sum(v,axis=0)/samplesize
        g = np.dot(e,v)/samplesize - emean * vmean 
        S = np.dot(v.T,v)/samplesize - np.outer(vmean,vmean)
        H = np.dot(v.T,Hv)/samplesize - np.outer(g,vmean) - emean * S 

        x = [wf(xi) for xi in x] 
        ovlp,hess = make_matrices(x)

        return S,H
    def extract(self,tmpdir): 
        e = np.load(tmpdir+'e.npy')
        v = np.load(tmpdir+'v.npy')
        Hv = np.load(tmpdir+'Hv.npy')
        samplesize = len(e)

        emean = e.sum()/samplesize
        estd = np.sqrt(((e-emean)**2).sum()/samplesize)

        vmean = np.sum(v,axis=0)/samplesize
        vstd = np.sqrt(np.sum((v-vmean)**2,axis=0)/samplesize)

        ev = e * v
        evmean = np.sum(ev,axis=0)/samplesize
        evstd = np.sqrt(np.sum((ev-evmean)**2,axis=0)/samplesize)

        Hvmean = np.sum(Hv,axis=0)/samplesize
        Hvstd = np.sqrt(np.sum((Hv-Hvmean)**2,axis=0)/samplesize)
class ExactVMC:
    def __init__(self):
        return
    def sample(self,x,tmpdir,progbar=False):
        af = AmplitudeFactory(x)
        samples = list(itertools.product((0,1),repeat=af.nsite))
        nparam = af.nparam
         
        ntotal = len(samples)
        batchsize,remain = ntotal//SIZE,ntotal%SIZE
        count = np.array([batchsize]*SIZE)
        if remain > 0:
            count[-remain:] += 1
        disp = np.concatenate([np.array([0]),np.cumsum(count[:-1])])
        start = disp[RANK]
        stop = start + count[RANK]

        esum = 0
        norm = 0
        vsum = np.zeros(nparam) 
        Hvsum = np.zeros(nparam)
        evsum = np.zeros(nparam) 
        ssum = np.zeros((nparam,nparam))
        hsum = np.zeros((nparam,nparam)) 
        if RANK==SIZE-1 and probar:
            pg = Progbar(total=batchsize) 
        for ix in range(start,stop):
            config = samples[ix]
            log_px = af.log_prob(config) 
            wx = np.exp(log_px)
            norm += wx

            ex,vx,Hvx = af.compute_local_energy(config)
            esum += ex * wx
            vsum += vx * wx
            evsum += ex * vx * wx
            Hvsum += Hvx * wx 
            ssum += np.outer(vx,vx) * wx
            hvsum += np.outer(vx,Hvx) * wx

            if RANK==SIZE-1 and probar:
                pg.update()

        norm = np.array(norm)
        n = np.zeros_like(norm)
        COMM.Reduce(norm,n,op=MPI.SUM,root=0)
        n = n[0]

        esum = np.array(esum)
        e = np.zeros_like(e)
        COMM.Reduce(esum,e,op=MPI.SUM,root=0)
        e = e[0] / n

        v = np.zeros_like(vsum)
        COMM.Reduce(vsum,v,op=MPI.SUM,root=0)
        v /= n

        ev = np.zeros_like(evsum)
        COMM.Reduce(evsum,ev,op=MPI.SUM,root=0)
        ev /= n
        g = ev - e * v

        Hv = np.zeros_like(Hvsum)
        COMM.Reduce(Hvsum,Hv,op=MPI.SUM,root=0)
        Hv /= n

        S = np.zeros_like(ssum)
        COMM.Reduce(ssum,S,op=MPI.SUM,root=0)
        S = S/n - np.outer(v,v)

        H = np.zeros_like(hsum)
        COMM.Reduce(hsum,H,op=MPI.SUM,root=0)
        H = H/n - np.outer(g,v) - e * S
        if RANK>0:
            return

        x = [wf(xi) for xi in x] 
        ovlp,hess = make_matrices(x)
        print('ovlp')
        print(S)
        print(ovlp)
        print('hess')
        print(H)
        print(hess)
if __name__ == '__main__':
    print('check analytical derivative')
    c = wf(np.random.rand(2))
    print(c)
    g1 = deriv_wf(c)
    print(g1)
    for eps in (1e-4,1e-5,1e-6):
        g2 = deriv_wf2(c,eps=eps)
        print(eps,np.linalg.norm(g1-g2))

    #Lmax = 16
    Lmax = 10
    x = [wf(np.random.rand(2) * 2 - 1) for _ in range(Lmax + 1)]
    tr = []
    for L in range(1,Lmax + 1):


    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':16})
    plt.rcParams.update({'figure.figsize':(6.4,4.8)})

    c = wf(np.array([.1,.5]))
    x = [c.copy() for _ in range(50)]

    fig,ax = plt.subplots(nrows=1,ncols=1)
    colors = [(r,g,b) for r,g,b in itertools.product((.3,.6,.9),repeat=3)]
    for run in range(10):
        print(run)
        x = [wf(np.random.rand(2) * 2 - 1) for _ in range(50)]
        tr = []
        for L in range(1,50):
        #for L in range(1,5):
            #print(L)
            ovlp,hess = make_matrices(x[:L])
            #print(hess)
            #print(ovlp)
            ovlp += 1.e-5 * np.eye(ovlp.shape[0])
            #print(np.linalg.eigvals(ovlp))
            eigs = scipy.linalg.eigvalsh(hess, ovlp)
            # hess = np.linalg.inv(ovlp) @ hess
            # trace = np.trace(hess)
            sumeigs = np.sum(np.abs(eigs))
            maxeig = np.max(np.abs(eigs))
            #print(L,sumeigs/maxeig)
            tr.append(sumeigs/maxeig)
            #print(eigs)
            #  print(np.sum(eigs))/np.max(np.abs(eigs))
            #print(np.max(np.abs(eigs)))
        ax.plot(range(1,50),tr, linestyle='-', color=colors[run],label=f'run={run}')
    ax.set_xlabel('L')
    ax.set_ylabel('norm_trace')
    ax.legend()
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.99, top=0.99)
    fig.savefig(f"vmc_model.png", dpi=250)
    
    
