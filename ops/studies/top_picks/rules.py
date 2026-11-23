import pandas as pd, numpy as np, sys
t=pd.read_parquet("aplus.parquet")
K=int(sys.argv[1]) if len(sys.argv)>1 else 8
MIN=int(sys.argv[2]) if len(sys.argv)>2 else 9
c=t[t.n_day>=MIN].copy()
rng=np.random.default_rng(1)
rules={
 "current(score,p,dist50)":[("score",0),("p",0),("dist50",0)],
 "p":[("p",0)],"e_pct":[("e_pct",0)],"e":[("e",0)],
 "rr hi":[("rr",0)],"rr lo":[("rr",1)],
 "rsi2 lo":[("rsi2",1)],"depth hi":[("depth_atr",0)],"depth lo":[("depth_atr",1)],
 "atr lo":[("atr_pct",1)],"atr hi":[("atr_pct",0)],
 "rs hi":[("rs_pct",0)],"dist200 hi":[("dist200",0)],"dist200 lo":[("dist200",1)],
 "dist50 hi":[("dist50",0)],"dist50 lo":[("dist50",1)],
 "ibs lo":[("ibs",1)],"vol hi":[("vol_ratio",0)],"vol lo":[("vol_ratio",1)],
 "adx hi":[("adx14",0)],"adx lo":[("adx14",1)],"ret5 lo":[("ret5",1)],"ret5 hi":[("ret5",0)],
 "etf first,score":[("is_etf",0),("score",0)],"stock first,score":[("is_etf",1),("score",0)],
 "slope hi":[("slope200",0)],"streak hi":[("streak",0)],"streak lo":[("streak",1)],
}
def stats(g):
    d=g.groupby("signal_date").agg(w=("win","mean"),r=("ret","mean"))
    return len(g),g.win.mean(),g.ret.mean(),d.w.mean(),d.r.mean(),(d.r<0).mean(), g.ret[g.ret<0].sum()/max(1e-9,-0+abs(g.ret[g.ret<0].sum()))
def pf(g):
    gp=g.ret[g.ret>0].sum(); gl=-g.ret[g.ret<0].sum(); return gp/gl if gl>0 else np.inf
rows=[]
for oos in(False,True):
    cc=c[c.oos==oos]
    n,w,r,dw,dr,dl,_=stats(cc)
    rows.append(("ALL (any pick)",oos,n,w,r,dw,dr,pf(cc)))
    for name,keys in rules.items():
        s=cc.sort_values(["signal_date"]+[k for k,_ in keys],ascending=[True]+[bool(a) for _,a in keys])
        top=s.groupby("signal_date").head(K)
        n,w,r,dw,dr,dl,_=stats(top)
        rows.append((name,oos,n,w,r,dw,dr,pf(top)))
df=pd.DataFrame(rows,columns=["rule","oos","n","win","avg","day_win","day_avg","pf"])
a=df[~df.oos].set_index("rule"); b=df[df.oos].set_index("rule")
out=a[["n","win","avg","pf"]].join(b[["n","win","avg","pf"]],lsuffix="_pre",rsuffix="_oos")
for col in("win_pre","win_oos"): out[col]=(100*out[col]).round(1)
for col in("avg_pre","avg_oos"): out[col]=(100*out[col]).round(2)
out["pf_pre"]=out.pf_pre.round(2); out["pf_oos"]=out.pf_oos.round(2)
print(f"K={K}, days with >= {MIN} A+; pre days {c[~c.oos].signal_date.nunique()}, oos days {c[c.oos].signal_date.nunique()}")
print(out.to_string())
