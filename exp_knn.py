# exp_knn.py -- #2: kNN-LM-style retrieval to break the last-2-token Markov ceiling.
#
# Datastore: for every corpus position t, key = unit(mean emb of last L tokens),
# value = ids[t] (the token that actually followed). At generation time we encode
# the current context the same way, find the k nearest keys (cosine), turn their
# next-tokens into a distribution p_knn = softmax(sim/tau), and INTERPOLATE with the
# n-gram+flock distribution:  p = (1-lam)*p_ngram + lam*p_knn.
# Tokens only kNN proposes can still be sampled -> the Markov ceiling is broken,
# and continuations are borrowed from real, coherent text. Gradient-free.
# (Khandelwal et al. 2020, arXiv:1911.00172.)  base64-verified.
import time, base64
import numpy as np
import exp_fano_layer as X      # reuse octonion algebra + build + metrics

def build_datastore(M, ids, L=4):
    """keys[t] = unit(mean emb[ids[t-L..t-1]]), values[t] = ids[t]."""
    emb = M["emb"]; E = emb[ids]                       # (N, D)
    N, D = E.shape
    cs = np.vstack([np.zeros((1, D), np.float32), np.cumsum(E, 0)])
    keys = np.empty((N, D), np.float32); vals = ids.astype(np.int64)
    for t in range(N):
        lo = max(0, t - L)
        keys[t] = (cs[t] - cs[lo]) / max(t - lo, 1)
    keys = X.unit(keys).astype(np.float32)
    return keys, vals

def build_datastore_octo(M, ids, decay=0.8):
    """Octonion-state key: key[t] = unit(SK_t) where SK rolls the SAME role-binding
    recurrence used at generation. Order- and role-aware (subject/predicate), so
    neighbours share STRUCTURAL context, not just a bag of recent words."""
    embK=M["embK"]; act=M["act"]; RS=M["RS"]; RP=M["RP"]; K=M["K"]; N=len(ids)
    keys=np.empty((N,8*K),np.float32); SK=np.zeros((K,8))
    for t in range(N):
        keys[t]=SK.reshape(-1)
        x=int(ids[t]); SK=decay*SK+X.octo_mul(RP if act[x] else RS, embK[x])
    return X.unit(keys).astype(np.float32), ids.astype(np.int64)

def generate_knn(M, DS, seed, n=80, temp=0.5, decay=0.8, drift=0.18,
                 w_subj=1.5, w_pred=3.5, w_alt=2.0, rep_pen=2.0,
                 w_cohesion=2.0, w_align=1.0, flock=7, veto=True, rng_seed=1,
                 w_goal=0.0, knn_lambda=0.0, knn_k=64, knn_tau=0.08, L=4,
                 knn_key="bow"):
    vocab,wi,W,emb,embK=M["vocab"],M["wi"],M["W"],M["emb"],M["embK"]
    tri,bi,act,RS,RP=M["tri"],M["bi"],M["act"],M["RS"],M["RP"]
    keys,vals=DS; rng=np.random.default_rng(rng_seed)
    out=[wi[w] for w in seed.lower().split() if w in wi] or [int(rng.integers(W))]
    s=unit=X.unit; s=X.unit(emb[out].mean(0)); cvec=s.copy(); SK=np.zeros((M["K"],8))
    goal=X.unit(emb[out].mean(0))
    for x in out: SK=decay*SK+X.octo_mul(RP if act[x] else RS, embK[x])
    recent={}; since=0; seen=set()
    for _ in range(n):
        c=tri.get((out[-2],out[-1])) if len(out)>=2 else None
        if not c: c=bi.get(out[-1])
        pmix={}
        # ---- n-gram + flock distribution over its candidate set --------------
        if c:
            cand=np.array(list(c)); fr=np.array([c[x] for x in cand],float)
            if veto and len(cand)>1:
                k=np.array([(out[-1],int(x)) not in seen for x in cand])
                if k.any(): cand,fr=cand[k],fr[k]
            nz=lambda x:(x-x.min())/(np.ptp(x)+1e-9) if len(cand)>1 else x*0
            es=X.unit(X.octo_mul(X._inv(RS),SK).reshape(-1)); ep=X.unit(X.octo_mul(X._inv(RP),SK).reshape(-1))
            want=1.0 if since>=2 else -0.5
            al=np.array([want if act[x] else 0.0 for x in cand])
            rp=np.array([recent.get(int(x),0) for x in cand],float)
            fl=out[-flock:]; cen=X.unit(emb[fl].mean(0))
            mot=X.unit(emb[fl[-1]]-emb[fl[0]]) if len(fl)>1 else cen
            coh=nz(emb[cand]@cen); ali=nz(emb[cand]@mot)
            gl=nz(emb[cand]@goal) if w_goal else 0.0
            sc=(np.log(fr)+nz(emb[cand]@cvec)+1.5*nz(emb[cand]@s)
                +w_subj*nz(emb[cand]@es)+w_pred*nz(emb[cand]@ep)+w_alt*al
                +w_cohesion*coh+w_align*ali+w_goal*gl-rep_pen*rp)
            pg=np.exp(sc/temp); pg/=pg.sum()
            for tk,pv in zip(cand,pg): pmix[int(tk)]=(1-knn_lambda)*pv
        # ---- kNN retrieval distribution --------------------------------------
        if knn_lambda>0:
            if knn_key=="octo":
                q=X.unit(SK.reshape(-1)).astype(np.float32)   # octonion context fingerprint
            else:
                fl=out[-L:]; q=X.unit(emb[fl].mean(0)).astype(np.float32)
            sims=keys@q                                   # cosine over datastore
            idx=np.argpartition(sims,-knn_k)[-knn_k:]
            ws=np.exp(sims[idx]/knn_tau); ws/=ws.sum()
            pk={}
            for j,wv in zip(idx,ws):
                t=int(vals[j])
                if veto and (out[-1],t) in seen: continue
                pk[t]=pk.get(t,0.0)+float(wv)
            z=sum(pk.values()) or 1.0
            for t,pv in pk.items(): pmix[t]=pmix.get(t,0.0)+knn_lambda*pv/z
        if pmix:
            toks=np.array(list(pmix)); pr=np.array(list(pmix.values()),float); pr/=pr.sum()
            nxt=int(rng.choice(toks,p=pr))
        else:
            nxt=int(rng.integers(W))
        seen.add((out[-1],nxt))
        out.append(nxt); recent={k:v*0.6 for k,v in recent.items()}; recent[nxt]=recent.get(nxt,0)+1
        since=0 if act[nxt] else since+1
        cvec=0.85*cvec+0.15*emb[nxt]; s=X.unit((1-drift)*s+drift*emb[nxt])
        SK=decay*SK+X.octo_mul(RP if act[nxt] else RS, embK[nxt])
    return [vocab[i] for i in out], out

if __name__=="__main__":
    t0=time.time()
    TEXT=open("corpus_books.txt",encoding="utf-8",errors="ignore").read()[:9_000_000]
    M=X.build(TEXT, vocab_size=8000)
    import re
    words=re.findall(r"[a-z']+", TEXT.lower())
    ids=np.array([M["wi"][w] for w in words if w in M["wi"]], dtype=np.int64)
    keys,vals=build_datastore(M, ids, L=4)
    print("built %.0fs  datastore N=%d  %.0fMB"%(time.time()-t0, len(vals), keys.nbytes/1e6))
    seeds=["the king","she looked at the","in the morning","the meaning of",
           "the old man","they walked through the","the war had"]
    cfgs=[("baseline (ngram)  ", dict(knn_lambda=0.0)),
          ("+knn lam0.25      ", dict(knn_lambda=0.25)),
          ("+knn lam0.50      ", dict(knn_lambda=0.50)),
          ("+knn0.35 +goal3   ", dict(knn_lambda=0.35, w_goal=3.0))]
    L=[]
    for name,cfg in cfgs:
        a=b=an=r=0.0; sample=""
        for si,sd in enumerate(seeds):
            toks,idl=generate_knn(M,(keys,vals),sd,n=80,rng_seed=si+1,**cfg)
            lc,se,anc,rep=X.metrics(M,idl); a+=lc;b+=se;an+=anc;r+=rep
            if si==1: sample=" ".join(toks)
        k=len(seeds)
        L.append("%s coh=%.3f drift=%.3f anchor=%.3f rep=%.3f"%(name,a/k,b/k,an/k,r/k))
        L.append("   sample: "+sample[:230])
    print("B64KNN:"+base64.b64encode("\n".join(L).encode()).decode())
