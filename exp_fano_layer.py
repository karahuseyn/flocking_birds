# exp_fano_layer.py -- #4 (non-decaying octonion memory) + #5 (slow/fast) via a
# CYCLIC OCTONIONIC FANO PATH as an intermediate layer.
#
# Idea: the 7 Fano points = the 7 imaginary octonion units e1..e7, cyclically ordered
# by the Singer automorphism e_i -> e_{i+1 mod 7}. Token at position p takes role
# R[p mod 7]. The last-7 flock then occupies all 7 Fano points exactly once -> a
# holographic 7-slot circular register H = sum_k R[(p) mod 7] (x) embK[token].
# Because octonions are a division algebra and {e_j, x} generate an associative
# subalgebra (Artin), unbind R[j]^-1 (x) H recovers slot j EXACTLY (+structured
# crosstalk), cleaned up by nearest-embedding. Two readouts:
#   (a) ECHO prior (#5 slow rhythm): unbind the NEXT slot's role -> what occupied
#       this Fano point one cycle (7 tokens) ago = period-7 structural prior.
#   (b) SLOW topic (#5): once per 7-cycle, commit the cycle centroid to a slowly
#       moving topic state (non-decaying within a cycle).
# All gradient-free. base64-verified.
import os, re, time, base64
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

def _qmul(x, y):
    a0,a1,a2,a3 = x[...,0],x[...,1],x[...,2],x[...,3]
    b0,b1,b2,b3 = y[...,0],y[...,1],y[...,2],y[...,3]
    return np.stack([a0*b0-a1*b1-a2*b2-a3*b3, a0*b1+a1*b0+a2*b3-a3*b2,
                     a0*b2-a1*b3+a2*b0+a3*b1, a0*b3+a1*b2-a2*b1+a3*b0], -1)
def octo_mul(a, b):
    p,q = a[...,:4],a[...,4:]; r,s = b[...,:4],b[...,4:]
    cs=s.copy(); cs[...,1:]*=-1; cr=r.copy(); cr[...,1:]*=-1
    return np.concatenate([_qmul(p,r)-_qmul(cs,q), _qmul(s,p)+_qmul(q,cr)], -1)
def unit(v): return v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-12)
def _conj(a): o=a.copy(); o[...,1:]*=-1; return o
def _inv(a): return _conj(a)/((a*a).sum(-1,keepdims=True)+1e-12)
def _fano_role(steps,K):
    acc=np.zeros(8); acc[0]=1.0
    for g in steps:
        e=np.zeros(8); e[g]=1.0; acc=octo_mul(acc,e)
    return np.tile(acc,(K,1))
def fano_cycle_roles(K):
    """7 cyclic Fano-point roles: e1..e7 tiled across K octonion blocks."""
    R=[]
    for i in range(1,8):
        e=np.zeros(8); e[i]=1.0; R.append(np.tile(e,(K,1)))
    return R                       # R[s] is role for Fano point/slot s (0..6)
VSUF=("ed","ing","es","ize","ise","ate","fy")

def build(text, vocab_size=8000, K=12, window=5, shift=5.0):
    words=re.findall(r"[a-z']+", text.lower())
    vocab=[w for w,_ in Counter(words).most_common(vocab_size)]
    wi={w:i for i,w in enumerate(vocab)}; W=len(vocab)
    ids=np.array([wi[w] for w in words if w in wi], dtype=np.int64)
    C=csr_matrix((W,W),dtype=np.float32)
    for d in range(1,window+1):
        a,b=ids[:-d],ids[d:]; data=np.full(len(a),np.float32(1.0/d),np.float32)
        Cd=coo_matrix((data,(a,b)),shape=(W,W)).tocsr(); C=C+Cd+Cd.T
    tot=C.sum(); Pa=np.asarray(C.sum(1)).ravel()/tot; Cx=C.tocoo()
    pmi=np.log(Cx.data/tot/(Pa[Cx.row]*Pa[Cx.col]+1e-30)+1e-12)-np.log(shift)
    keep=pmi>0
    P=coo_matrix((pmi[keep],(Cx.row[keep],Cx.col[keep])),shape=(W,W)).tocsr()
    D=8*K
    U,S,_=svds(P,k=min(D,W-1)); o=np.argsort(S)[::-1]; U,S=U[:,o],S[o]
    if U.shape[1]<D: U=np.pad(U,((0,0),(0,D-U.shape[1]))); S=np.pad(S,(0,D-len(S)))
    emb=unit(U*np.sqrt(S))
    idl=ids.tolist(); tri=defaultdict(Counter); bi=defaultdict(Counter)
    for i in range(len(idl)-1): bi[idl[i]][idl[i+1]]+=1
    for i in range(len(idl)-2): tri[(idl[i],idl[i+1])][idl[i+2]]+=1
    act=np.array([any(vocab[i].endswith(s) for s in VSUF) and len(vocab[i])>4 for i in range(W)])
    return dict(vocab=vocab,wi=wi,W=W,K=K,emb=emb,embK=emb.reshape(W,K,8),
                tri=tri,bi=bi,act=act,RS=_fano_role([1,2],K),RP=_fano_role([3,4],K),
                RC=fano_cycle_roles(K))

def generate(M, seed, n=80, temp=0.5, decay=0.8, drift=0.18,
             w_subj=1.5, w_pred=3.5, w_alt=2.0, rep_pen=2.0,
             w_cohesion=2.0, w_align=1.0, flock=7, veto=True, rng_seed=1,
             w_goal=0.0, w_fano=0.0, w_slow=0.0, slow_r=0.5):
    vocab,wi,W,emb,embK=M["vocab"],M["wi"],M["W"],M["emb"],M["embK"]
    tri,bi,act,RS,RP,RC=M["tri"],M["bi"],M["act"],M["RS"],M["RP"],M["RC"]
    K=M["K"]; rng=np.random.default_rng(rng_seed)
    out=[wi[w] for w in seed.lower().split() if w in wi] or [int(rng.integers(W))]
    s=unit(emb[out].mean(0)); cvec=s.copy(); SK=np.zeros((K,8))
    goal=unit(emb[out].mean(0)); slow=unit(emb[out].mean(0))
    for x in out: SK=decay*SK+octo_mul(RP if act[x] else RS, embK[x])
    recent={}; since=0; seen=set()
    for _ in range(n):
        c=tri.get((out[-2],out[-1])) if len(out)>=2 else None
        if not c: c=bi.get(out[-1])
        if c:
            cand=np.array(list(c)); fr=np.array([c[x] for x in cand],float)
            if veto and len(cand)>1:
                k=np.array([(out[-1],int(x)) not in seen for x in cand])
                if k.any(): cand,fr=cand[k],fr[k]
            nz=lambda x:(x-x.min())/(np.ptp(x)+1e-9) if len(cand)>1 else x*0
            es=unit(octo_mul(_inv(RS),SK).reshape(-1)); ep=unit(octo_mul(_inv(RP),SK).reshape(-1))
            want=1.0 if since>=2 else -0.5
            al=np.array([want if act[x] else 0.0 for x in cand])
            rp=np.array([recent.get(int(x),0) for x in cand],float)
            fl=out[-flock:]; cen=unit(emb[fl].mean(0))
            mot=unit(emb[fl[-1]]-emb[fl[0]]) if len(fl)>1 else cen
            coh=nz(emb[cand]@cen); ali=nz(emb[cand]@mot)
            gl=nz(emb[cand]@goal) if w_goal else 0.0
            # ---- CYCLIC FANO PATH intermediate layer -------------------------
            fa=0.0
            if w_fano:
                base=len(out)-len(fl)
                H=np.zeros((K,8))
                for kk,tok in enumerate(fl):
                    H=H+octo_mul(RC[(base+kk)%7], embK[tok])     # 7-slot holographic register
                ns=len(out)%7                                    # next token's Fano slot
                pred=unit(octo_mul(_inv(RC[ns]), H).reshape(-1)) # unbind -> echo from 1 cycle ago
                fa=nz(emb[cand]@pred)
            sl=nz(emb[cand]@slow) if w_slow else 0.0
            sc=(np.log(fr)+nz(emb[cand]@cvec)+1.5*nz(emb[cand]@s)
                +w_subj*nz(emb[cand]@es)+w_pred*nz(emb[cand]@ep)+w_alt*al
                +w_cohesion*coh+w_align*ali+w_goal*gl+w_fano*fa+w_slow*sl-rep_pen*rp)
            p=np.exp(sc/temp); p/=p.sum(); nxt=int(rng.choice(cand,p=p))
        else:
            nxt=int(rng.integers(W))
        if out: seen.add((out[-1],nxt))
        out.append(nxt); recent={k:v*0.6 for k,v in recent.items()}; recent[nxt]=recent.get(nxt,0)+1
        since=0 if act[nxt] else since+1
        cvec=0.85*cvec+0.15*emb[nxt]; s=unit((1-drift)*s+drift*emb[nxt])
        SK=decay*SK+octo_mul(RP if act[nxt] else RS, embK[nxt])
        if w_slow and len(out)%7==0:                              # once per Fano cycle
            cyc=unit(emb[out[-7:]].mean(0)); slow=unit((1-slow_r)*slow+slow_r*cyc)
    return [vocab[i] for i in out], out

def metrics(M, ids):
    emb=M["emb"]; E=emb[ids]; win=5
    cents=[unit(E[i:i+win].mean(0)) for i in range(0,len(E)-win,2)]
    lc=np.mean([cents[i]@cents[i+1] for i in range(len(cents)-1)]) if len(cents)>1 else 0.0
    a=unit(E[:12].mean(0)); b=unit(E[-12:].mean(0)); se=1.0-float(a@b)
    g=unit(E.mean(0)); anc=float(g@a)
    bg=list(zip(ids[:-1],ids[1:])); rep=1.0-len(set(bg))/max(len(bg),1)
    return lc, se, anc, rep

if __name__=="__main__":
    t0=time.time()
    TEXT=open("corpus_books.txt",encoding="utf-8",errors="ignore").read()
    M=build(TEXT, vocab_size=8000); print("built %.0fs"%(time.time()-t0))
    seeds=["the king","she looked at the","in the morning","the meaning of",
           "the old man","they walked through the","the war had"]
    configs=[("baseline          ", dict()),
             ("fano echo 2.5     ", dict(w_fano=2.5)),
             ("slow cycle 2.5    ", dict(w_slow=2.5)),
             ("fano+slow         ", dict(w_fano=2.0,w_slow=2.0)),
             ("fano+slow+goal3   ", dict(w_fano=2.0,w_slow=2.0,w_goal=3.0))]
    lines=[]
    for name,cfg in configs:
        LC=SE=ANC=REP=0.0; sample=""
        for si,sd in enumerate(seeds):
            toks,ids=generate(M, sd, n=80, rng_seed=si+1, **cfg)
            lc,se,anc,rep=metrics(M, ids); LC+=lc; SE+=se; ANC+=anc; REP+=rep
            if si==1: sample=" ".join(toks)
        k=len(seeds)
        lines.append("%s localcoh=%.3f drift=%.3f anchor=%.3f rep=%.3f"%(name,LC/k,SE/k,ANC/k,REP/k))
        lines.append("   sample: "+sample[:230])
    print("B64FANO:"+base64.b64encode("\n".join(lines).encode()).decode())
