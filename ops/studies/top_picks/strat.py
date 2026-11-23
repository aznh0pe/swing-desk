import pandas as pd, numpy as np, sys
sys.path.insert(0,"/home/claude/swing-desk")
from engine import strategies as S, data as D
t=pd.read_parquet("../bt_base/trades.parquet")
t["cond"]=[S.condition_of(a,b,c) for a,b,c in zip(t.spy_above200,t.vix_ratio,t.spy_ret5)]
a=pd.read_parquet("aplus.parquet")
a["cond"]=[S.condition_of(x,y,z) for x,y,z in zip(a.spy_above200,a.vix_ratio,a.spy_ret5)]
try:
    u=D.load_universe("/home/claude/work/data_now"); a["sector"]=a.ticker.map(lambda k: (u.loc[k,"sector"] if k in u.index and isinstance(u.loc[k,"sector"],str) else (u.loc[k,"name"] if k in u.index else None)))
except Exception as e:
    print("no universe",e); a["sector"]=None
print("conditions on crowded days:", a[a.n_day>=9].drop_duplicates("signal_date").groupby(["oos","cond"]).size().to_dict())
# walk-forward strategy record: all A+ trades of the strategy whose exit_date < signal day (overall and in the same condition)
a=a.sort_values("signal_date").reset_index(drop=True)
hist=a[["strategy","cond","exit_date","win","ret"]].copy()
N0=30
def prior(day,strat,cond=None):
    h=hist[(hist.exit_date<day)]
    base=h.win.mean() if len(h) else 0.75
    h=h[h.strategy==strat]
    if cond is not None: h=h[h.cond==cond]
    return (h.win.sum()+N0*base)/(len(h)+N0)
days=a[a.n_day>=9].signal_date.unique()
rec={}
for d in days:
    g=a[a.signal_date==d]
    for s in g.strategy.unique():
        rec[(d,s)]=(prior(d,s),prior(d,s,g.cond.iloc[0]))
c=a[a.n_day>=9].copy()
c["sr"]=[rec[(d,s)][0] for d,s in zip(c.signal_date,c.strategy)]
c["src"]=[rec[(d,s)][1] for d,s in zip(c.signal_date,c.strategy)]
c.to_parquet("crowded.parquet")
def pick(g,keys,K,per_strat=None,per_sec=None):
    g=g.sort_values([k for k,_ in keys],ascending=[bool(x) for _,x in keys])
    out=[];ns={};nc={}
    for i,r in g.iterrows():
        if per_strat and ns.get(r.strategy,0)>=per_strat: continue
        if per_sec and r.sector and nc.get(r.sector,0)>=per_sec: continue
        out.append(i); ns[r.strategy]=ns.get(r.strategy,0)+1
        if r.sector: nc[r.sector]=nc.get(r.sector,0)+1
        if len(out)>=K: break
    return out
CUR=[("score",0),("p",0),("dist50",0)]
variants={
 "A current":(CUR,None,None),
 "A2 current + sector<=2":(CUR,None,2),
 "B strat record, score":([("sr",0)]+CUR,None,2),
 "E strat record in condition, score":([("src",0)]+CUR,None,2),
 "C current, <=3/strategy":(CUR,3,2),
 "D record, <=3/strategy":([("sr",0)]+CUR,3,2),
 "F cond record, <=3/strategy":([("src",0)]+CUR,3,2),
 "G cond record, <=2/strategy":([("src",0)]+CUR,2,2),
}
for K in(8,3):
    print(f"\nK={K}")
    for name,(keys,ps,pc) in variants.items():
        line=f"{name:36s}"
        for oos in(False,True):
            cc=c[c.oos==oos]; idx=[]
            for d,g in cc.groupby("signal_date"): idx+=pick(g,keys,K,ps,pc)
            top=cc.loc[idx]; dd=top.groupby("signal_date").ret.mean()
            gp=top.ret[top.ret>0].sum(); gl=-top.ret[top.ret<0].sum()
            line+=f" | {'oos' if oos else 'pre'} n {len(top):4d} win {100*top.win.mean():5.1f} avg {100*top.ret.mean():5.2f} pf {gp/gl:4.2f} losing days {100*(dd<0).mean():4.1f}% worst5 {100*dd.nsmallest(5).mean():6.2f}"
        print(line)
    for oos in(False,True):
        cc=c[c.oos==oos]; dd=cc.groupby("signal_date").ret.mean(); dw=cc.groupby("signal_date").win.mean()
        print(f"  baseline any pick {'oos' if oos else 'pre'}: win {100*dw.mean():.1f} avg {100*dd.mean():.2f} losing days {100*(dd<0).mean():.1f}% worst5 {100*dd.nsmallest(5).mean():.2f}")
