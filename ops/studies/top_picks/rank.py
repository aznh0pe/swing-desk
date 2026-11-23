import pandas as pd, numpy as np, sys
sys.path.insert(0,"/home/claude/swing-desk")
from engine import data as D
c=pd.read_parquet("crowded.parquet")
w=D.load_history("/home/claude/work/data_hist"); ret=w["close"].pct_change()
print("history", ret.index.min().date(), ret.index.max().date(), ret.shape)
CUR=[("score",0),("p",0),("dist50",0)]
def corr_ok(day,tk,chosen,thr):
    if not chosen or tk not in ret.columns: return True
    r=ret.loc[:day].tail(60)
    for o in chosen:
        if o in r.columns:
            cv=r[tk].corr(r[o])
            if cv==cv and cv>=thr: return False
    return True
def pick(g,K,per_strat=None,per_sec=2,thr=None):
    g=g.sort_values([k for k,_ in CUR],ascending=[False]*3)
    out=[];why={};ns={};nc={};tks=[]
    for i,r in g.iterrows():
        if per_strat and ns.get(r.strategy,0)>=per_strat: why[i]="strat";continue
        if per_sec and r.sector and nc.get(r.sector,0)>=per_sec: why[i]="sector";continue
        if thr and not corr_ok(r.signal_date,r.ticker,tks,thr): why[i]="corr";continue
        out.append(i); tks.append(r.ticker); ns[r.strategy]=ns.get(r.strategy,0)+1
        if r.sector: nc[r.sector]=nc.get(r.sector,0)+1
        if len(out)>=K: break
    return out,why
def rep(name,top):
    line=f"{name:34s}"
    for oos in(False,True):
        tt=top[top.oos==oos]; dd=tt.groupby("signal_date").ret.mean()
        gp=tt.ret[tt.ret>0].sum(); gl=-tt.ret[tt.ret<0].sum()
        line+=f" | {'oos' if oos else 'pre'} n {len(tt):4d} win {100*tt.win.mean():5.1f} avg {100*tt.ret.mean():5.2f} pf {gp/max(gl,1e-9):4.2f} lose-days {100*(dd<0).mean():4.1f}% worst5 {100*dd.nsmallest(5).mean():6.2f}"
    print(line)
res={}
for name,kw in {"A2 sector<=2":{}, "C +<=3/strategy":{"per_strat":3}, "C4 +<=4/strategy":{"per_strat":4}, "H sector + corr<0.85":{"thr":0.85},
                "I <=3/strat + corr<0.85":{"per_strat":3,"thr":0.85}, "J <=3/strat + corr<0.75":{"per_strat":3,"thr":0.75}}.items():
    idx=[];pos={};whys={}
    for d,g in c.groupby("signal_date"):
        o,wy=pick(g,8,**kw); idx+=o; whys.update(wy)
        for k,i in enumerate(o): pos[i]=k+1
    res[name]=(idx,pos,whys); rep(name,c.loc[idx])
# rank position quality under A2
idx,pos,_=res["A2 sector<=2"]; a=c.loc[idx].assign(pos=pd.Series(pos))
a["band"]=pd.cut(a.pos,[0,3,5,8],labels=["1-3","4-5","6-8"])
print(a.groupby(["oos","band"],observed=True).agg(n=("win","size"),win=("win","mean"),avg=("ret","mean")).round(3).to_string())
# what the strategy cap drops vs what replaces it
ia=set(res["A2 sector<=2"][0]); ic=set(res["C +<=3/strategy"][0])
rep("dropped by <=3/strategy",c.loc[sorted(ia-ic)]); rep("added instead",c.loc[sorted(ic-ia)])
ii=set(res["I <=3/strat + corr<0.85"][0]); rep("dropped by corr (vs C)",c.loc[sorted(ic-ii)]); rep("added instead (vs C)",c.loc[sorted(ii-ic)])
