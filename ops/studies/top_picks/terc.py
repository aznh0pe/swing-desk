import pandas as pd, numpy as np
t=pd.read_parquet("aplus.parquet"); c=t[t.n_day>=9].copy()
feats=["score","p","e_pct","rr","rsi2","depth_atr","atr_pct","rs_pct","dist200","dist50","ibs","vol_ratio","adx14","ret5","slope200","streak"]
print("baseline day-weighted: ", {o:(round(100*g.groupby('signal_date').win.mean().mean(),1), round(100*g.groupby('signal_date').ret.mean().mean(),2)) for o,g in c.groupby('oos')})
rows=[]
for f in feats:
    c["q"]=c.groupby("signal_date")[f].rank(pct=True,method="first")
    c["b"]=pd.cut(c.q,[0,1/3,2/3,1.0001],labels=["lo","mid","hi"])
    r=[f]
    for oos in(False,True):
        g=c[c.oos==oos].groupby("b",observed=True).agg(w=("win","mean"),a=("ret","mean"))
        r+= [round(100*g.w[k],1) for k in("lo","mid","hi")]+[round(100*g.a[k],2) for k in("lo","mid","hi")]
    rows.append(r)
cols=["feat"]+[f"{p}_{m}_{k}" for p in("pre","oos") for m in("win","avg") for k in("lo","mid","hi")]
print(pd.DataFrame(rows,columns=cols).to_string(index=False))
print(c.groupby(["oos","strategy"]).agg(n=("win","size"),win=("win","mean"),avg=("ret","mean")).round(3).to_string())
