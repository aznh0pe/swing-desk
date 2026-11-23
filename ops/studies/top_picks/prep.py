import pandas as pd, numpy as np
t=pd.read_parquet("../bt_base/trades.parquet")
t=t[(t.score>=50)&(t.p>=0.76)].copy()
t["rr"]=(t.target-t.limit)/(t.limit-t.stop)
t=t[t.rr>=0.2]
# one per ticker/day: best score
t=t.sort_values(["signal_date","ticker","score"],ascending=[True,True,False]).drop_duplicates(["signal_date","ticker"])
t["n_day"]=t.groupby("signal_date")["ticker"].transform("size")
t["oos"]=t.signal_date>="2022-01-01"
t["ret_f"]=np.where(t.filled,t.ret,0.0)
t.to_parquet("aplus.parquet")
d=t.drop_duplicates("signal_date")
print("A+ setups",len(t),"days",len(d))
for lo,hi in((1,8),(9,20),(21,50),(51,150),(151,9999)):
    m=(t.n_day>=lo)&(t.n_day<=hi); f=t[m&t.filled]
    for oos in(False,True):
        g=t[m&(t.oos==oos)]; ff=g[g.filled]
        print(f"n_day {lo}-{hi} oos={oos}: days {g.signal_date.nunique()}, setups {len(g)}, fill {g.filled.mean():.2f}, win {ff.win.mean():.3f}, avg {100*ff.ret.mean():.2f}%")
print(t.filled.dtype, t.win.dtype)
