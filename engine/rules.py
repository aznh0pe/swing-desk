"""Data-driven strategy specs: how research ideas become order plans.

A spec is plain JSON, so a new idea can be written, tested and promoted without code changes:

{
  "id": "R_ETF_3LL",                         R_ + capitals/digits/underscores
  "label": "ETF Three Lower Lows",
  "family": "Mean reversion",
  "about": "One or two plain-English sentences.",
  "source": {"title": "...", "url": "https://..."},
  "universe": "etf",                         all | stock | etf
  "conditions": ["close > sma200", "low < low[1]", "low[1] < low[2]", "low[2] < low[3]"],
  "entry": {"kind": "limit_next", "price": "close - {k}*atr10"},     or {"kind": "close"}
  "stop": {"atr": "{stop}"},                  or {"price": "ll20 - 0.25*atr10"}
  "target": {"atr": 1.5},                     or {"price": "sma5", "min_atr": 0.3, "cap_atr": 4}
  "exit": "sma5",                             sma5 | prev_high | high7 | rsi2_70 | none
  "max_hold": "{hold}", "valid_days": 1,
  "regime": [],                               optional gates: spy200, breadth, vixcalm
  "params": {"k": 0.3, "stop": 3.0, "hold": 7},
  "grid": {"k": [0, 0.3, 0.6], "hold": [5, 8]}
}

Expressions may use indicator frames (FRAME_NAMES), market series (MARKET_NAMES), calendar fields
(dow 0=Mon..4=Fri, tdom = trading day of the month from 1, tdom_rev = trading days left in the
month, 0 on its last session), numbers, + - * / and parentheses, lags x[n] (n sessions ago),
rmin / rmax / rmean / rsum(x, n), min(a, b), max(a, b) and abs(x). Conditions are comparisons
joined with and / or / not; all conditions must hold. {name} placeholders are filled from
params (defaults) or the grid (variants tried on 2008-2021 data only).
"""
from __future__ import annotations

import ast
import itertools
import json
import re

import numpy as np
import pandas as pd

from . import calendar as cal

FRAME_NAMES = {"open", "high", "low", "close", "volume", "sma5", "sma10", "sma20", "sma50", "sma100", "sma200",
               "ema10", "ema20", "rsi2", "rsi3", "rsi14", "atr10", "atr14", "atr_pct", "ibs", "pctb", "min7c",
               "max7c", "hh10", "hh20", "hh252", "ll20", "mean4", "streak", "slope200", "slope50", "dist200",
               "dist50", "v20", "v5", "dv20", "vol_ratio", "ret5", "ret63", "ret126", "adx14", "vwap20",
               "vstd20", "gap_up", "rs_pct", "sector_up", "is_etf", "is_stock", "is_lev", "range"}
MARKET_NAMES = {"spy", "spy_sma200", "spy_sma50", "spy_above200", "spy_dist200", "spy_ret5", "spy_rsi2",
                "breadth50", "breadth200", "vix", "vix_ratio"}
CAL_NAMES = {"dow", "tdom", "tdom_rev"}
FUNCS = {"rmin": 2, "rmax": 2, "rmean": 2, "rsum": 2, "min": 2, "max": 2, "abs": 1}
EXITS = {"sma5", "prev_high", "high7", "rsi2_70", "none"}
UNIVERSES = {"all", "stock", "etf"}
REGIMES = {"spy200", "breadth", "vixcalm"}
ID_RE = re.compile(r"^R_[A-Z0-9_]{2,40}$")
PH_RE = re.compile(r"\{(\w+)\}")
MAX_COMBOS = 36


# ----------------------------------------------------------------------------- params
def subst(x, params: dict):
    """Fill {name} placeholders. Numbers stay numbers."""
    if isinstance(x, str):
        def rep(m):
            k = m.group(1)
            if k not in params:
                raise ValueError(f"placeholder {{{k}}} has no value in params")
            return repr(params[k]) if isinstance(params[k], str) else str(params[k])
        out = PH_RE.sub(rep, x)
        return out
    return x


def _num(x, params, cast=float):
    v = subst(x, params)
    return cast(float(v))


def combos(spec: dict) -> list[dict]:
    base = dict(spec.get("params") or {})
    grid = spec.get("grid") or {}
    keys = [k for k, v in grid.items() if isinstance(v, list) and v]
    if not keys:
        return [base]
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        out.append({**base, **dict(zip(keys, vals))})
    return out[:MAX_COMBOS]


# ----------------------------------------------------------------------------- evaluator
class Evaluator:
    """Evaluates rule expressions over the indicator frames, caching every sub-result."""

    def __init__(self, ind: dict):
        self.ind = ind
        self.c = ind["close"]
        self.cache: dict = {}
        self._cal = None

    # fields
    def _bcast(self, s: pd.Series) -> pd.DataFrame:
        v = s.reindex(self.c.index).astype(float).values
        return pd.DataFrame(np.repeat(v[:, None], self.c.shape[1], axis=1), index=self.c.index, columns=self.c.columns)

    def _calendar(self) -> pd.DataFrame:
        if self._cal is None:
            idx = self.c.index
            fut, d = [], idx[-1].date()
            for _ in range(25):  # finish the current month with the exchange calendar
                d = cal.next_trading_day(d)
                fut.append(pd.Timestamp(d))
            alld = pd.DatetimeIndex(list(idx) + fut)
            ym = pd.Series(alld.year * 100 + alld.month, index=alld)
            tdom = ym.groupby(ym.values).cumcount() + 1
            size = ym.groupby(ym.values).transform("size")
            self._cal = pd.DataFrame({"dow": alld.dayofweek.astype(float), "tdom": tdom.values.astype(float),
                                      "tdom_rev": (size - tdom).values.astype(float)}, index=alld).loc[idx]
        return self._cal

    def field(self, name: str) -> pd.DataFrame:
        key = "f:" + name
        if key in self.cache:
            return self.cache[key]
        ind = self.ind
        if name == "range":
            f = ind["high"] - ind["low"]
        elif name in ("is_etf", "is_stock", "is_lev"):
            f = ind[name].astype(float)
        elif name in FRAME_NAMES and name in ind:
            f = ind[name]
        elif name in MARKET_NAMES and name in ind["market"].columns:
            f = self._bcast(ind["market"][name])
        elif name in CAL_NAMES:
            f = self._bcast(self._calendar()[name])
        else:
            raise ValueError(f"unknown field '{name}'")
        self.cache[key] = f
        return f

    # expressions
    def value(self, expr: str):
        expr = expr.strip()
        if expr in self.cache:
            return self.cache[expr]
        tree = ast.parse(expr, mode="eval")
        v = self._eval(tree.body)
        if len(self.cache) > 80:
            for k in [k for k in self.cache if not k.startswith("f:")][:40]:
                self.cache.pop(k, None)
        self.cache[expr] = v
        return v

    def mask(self, expr: str) -> pd.DataFrame:
        v = self.value(expr)
        if isinstance(v, (bool, np.bool_)):
            return pd.DataFrame(bool(v), index=self.c.index, columns=self.c.columns)
        if isinstance(v, pd.DataFrame):
            if v.dtypes.apply(lambda d: d == bool).all():
                return v
            return v.fillna(0) != 0
        raise ValueError(f"'{expr}' is not a condition")

    def frame(self, expr: str) -> pd.DataFrame:
        v = self.value(expr)
        if isinstance(v, pd.DataFrame):
            return v.astype(float)
        return pd.DataFrame(float(v), index=self.c.index, columns=self.c.columns)

    def _eval(self, n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.Name):
            return self.field(n.id)
        if isinstance(n, ast.Subscript):
            sl = n.slice
            if not (isinstance(sl, ast.Constant) and isinstance(sl.value, int) and 0 <= sl.value <= 260):
                raise ValueError("a lag must be a whole number of sessions, e.g. close[1]")
            base = self._eval(n.value)
            return base.shift(sl.value) if isinstance(base, pd.DataFrame) else base
        if isinstance(n, ast.UnaryOp):
            v = self._eval(n.operand)
            if isinstance(n.op, ast.USub):
                return -v
            if isinstance(n.op, ast.UAdd):
                return v
            if isinstance(n.op, ast.Not):
                return ~self._as_bool(v)
        if isinstance(n, ast.BinOp):
            a, b = self._eval(n.left), self._eval(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                if isinstance(b, pd.DataFrame):
                    b = b.replace(0, np.nan)
                elif b == 0:
                    return a * np.nan
                return a / b
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS:
            fn = n.func.id
            if len(n.args) != FUNCS[fn] or n.keywords:
                raise ValueError(f"{fn}() takes {FUNCS[fn]} argument(s)")
            args = [self._eval(a) for a in n.args]
            if fn in ("rmin", "rmax", "rmean", "rsum"):
                x, w = args
                w = int(w)
                if not 1 <= w <= 260 or not isinstance(x, pd.DataFrame):
                    raise ValueError(f"{fn}(x, n) needs a field and 1 <= n <= 260")
                r = x.rolling(w, min_periods=w)
                return {"rmin": r.min, "rmax": r.max, "rmean": r.mean, "rsum": r.sum}[fn]()
            if fn == "abs":
                return abs(args[0])
            a, b = args
            if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
                fa = a if isinstance(a, pd.DataFrame) else pd.DataFrame(a, index=self.c.index, columns=self.c.columns)
                return np.minimum(fa, b) if fn == "min" else np.maximum(fa, b)
            return min(a, b) if fn == "min" else max(a, b)
        if isinstance(n, ast.Compare):
            left = self._eval(n.left)
            res = None
            for op, comp in zip(n.ops, n.comparators):
                right = self._eval(comp)
                if isinstance(op, ast.Lt):
                    r = left < right
                elif isinstance(op, ast.LtE):
                    r = left <= right
                elif isinstance(op, ast.Gt):
                    r = left > right
                elif isinstance(op, ast.GtE):
                    r = left >= right
                elif isinstance(op, ast.Eq):
                    r = left == right
                elif isinstance(op, ast.NotEq):
                    r = left != right
                else:
                    raise ValueError("unsupported comparison")
                r = self._as_bool(r)
                res = r if res is None else (res & r)
                left = right
            return res
        if isinstance(n, ast.BoolOp):
            vals = [self._as_bool(self._eval(v)) for v in n.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(n.op, ast.And) else (out | v)
            return out
        raise ValueError(f"unsupported expression element: {type(n).__name__}")

    def _as_bool(self, v):
        if isinstance(v, pd.DataFrame):
            return v if v.dtypes.apply(lambda d: d == bool).all() else (v.fillna(0) != 0)
        return pd.DataFrame(bool(v), index=self.c.index, columns=self.c.columns)


_EVALS: dict = {}


def evaluator(ind: dict) -> Evaluator:
    key = id(ind)
    ev = _EVALS.get(key)
    if ev is None or ev.ind is not ind:
        _EVALS.clear()
        ev = _EVALS[key] = Evaluator(ind)
    return ev


# ----------------------------------------------------------------------------- plans
def build_plan(spec: dict, ind: dict, P: dict, b: dict, params: dict | None = None, name: str | None = None) -> dict:
    pr = {**(spec.get("params") or {}), **(params or {})}
    ev = evaluator(ind)
    c = ind["close"]
    uni = spec.get("universe", "all")
    if uni == "stock":
        m = b["is_stock"].copy()
    elif uni == "etf":
        m = b["is_etf"].copy()
    else:
        m = b["is_stock"] | b["is_etf"]
    m = m & b["liquid"]
    for cond in spec["conditions"]:
        m = m & ev.mask(subst(cond, pr))
    reg = spec.get("regime") or []
    if isinstance(reg, str):
        reg = [reg]
    if reg:
        mk = ind["market"]
        gate = pd.Series(True, index=c.index)
        if "spy200" in reg:
            gate &= mk["spy_above200"] >= 0.5
        if "breadth" in reg:
            gate &= mk["breadth50"].fillna(0.5) >= 0.35
        if "vixcalm" in reg:
            gate &= mk["vix_ratio"].fillna(1.0) <= 1.35
        m = m & pd.DataFrame(np.repeat(gate.values[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
    en = spec.get("entry") or {"kind": "limit_next", "price": "close"}
    kind = en.get("kind", "limit_next")
    limit = c.copy() if kind == "close" else ev.frame(subst(en.get("price", "close"), pr))
    st = spec.get("stop") or {"atr": 3.0}
    stop = ({"kind": "atr", "mult": _num(st["atr"], pr)} if "atr" in st
            else {"kind": "level", "frame": ev.frame(subst(st["price"], pr))})
    tg = spec.get("target") or {"atr": 1.0}
    target = ({"kind": "atr", "mult": _num(tg["atr"], pr)} if "atr" in tg
              else {"kind": "level", "frame": ev.frame(subst(tg["price"], pr)),
                    "min_atr": _num(tg.get("min_atr", 0.25), pr), "cap_atr": _num(tg.get("cap_atr", 4.0), pr)})
    dyn = str(subst(spec.get("exit", "none"), pr)).strip("'\"")
    return {"name": name or spec["id"], "entry_kind": "close" if kind == "close" else "limit_next",
            "valid_days": _num(spec.get("valid_days", 1), pr, int), "max_hold": _num(spec.get("max_hold", 7), pr, int),
            "dyn_exit": dyn if dyn in EXITS else "none", "stop": stop, "target": target,
            "mask": m.fillna(False).astype(bool), "limit": limit}


def strategy_entry(spec: dict, params: dict, enabled: bool) -> dict:
    """The P['strategies'][id] record the scanner reads for a promoted research strategy."""
    pr = {**(spec.get("params") or {}), **(params or {})}
    en = spec.get("entry") or {}
    return {"enabled": bool(enabled), "custom": True, "universe": spec.get("universe", "all"),
            "entry_kind": "close" if en.get("kind") == "close" else "limit_next",
            "max_hold": _num(spec.get("max_hold", 7), pr, int), "valid_days": _num(spec.get("valid_days", 1), pr, int),
            "dyn_exit": str(subst(spec.get("exit", "none"), pr)).strip("'\"")}


# ----------------------------------------------------------------------------- checks
def _names(tree) -> tuple[set, set]:
    names, funcs = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            funcs.add(n.func.id)
        elif isinstance(n, ast.Name):
            names.add(n.id)
    return names - funcs, funcs


def validate(spec: dict) -> list[str]:
    errs = []
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    if not ID_RE.match(str(spec.get("id", ""))):
        errs.append("id must look like R_SOME_NAME (capitals, digits, underscores)")
    for k in ("label", "family", "about"):
        if not isinstance(spec.get(k), str) or not spec.get(k).strip():
            errs.append(f"'{k}' is required")
    if spec.get("universe", "all") not in UNIVERSES:
        errs.append("universe must be all, stock or etf")
    conds = spec.get("conditions")
    if not isinstance(conds, list) or not conds or len(conds) > 14:
        errs.append("conditions must be a list of 1-14 strings")
        conds = []
    try:
        cs = combos(spec)
    except Exception as e:
        cs, _ = [{}], errs.append(f"grid: {e}")
    n_combos = 1
    for v in (spec.get("grid") or {}).values():
        n_combos *= max(1, len(v) if isinstance(v, list) else 1)
    if n_combos > MAX_COMBOS:
        errs.append(f"grid has {n_combos} combinations; keep it to {MAX_COMBOS} or fewer")
    pr = cs[0] if cs else {}
    exprs = list(conds)
    en = spec.get("entry") or {}
    if en.get("kind", "limit_next") not in ("limit_next", "close"):
        errs.append("entry.kind must be limit_next or close")
    if en.get("kind", "limit_next") == "limit_next":
        exprs.append(en.get("price", "close"))
    st, tg = spec.get("stop") or {}, spec.get("target") or {}
    if not ("atr" in st or "price" in st):
        errs.append("stop needs 'atr' or 'price'")
    elif "price" in st:
        exprs.append(st["price"])
    if not ("atr" in tg or "price" in tg):
        errs.append("target needs 'atr' or 'price'")
    elif "price" in tg:
        exprs.append(tg["price"])
    reg = spec.get("regime") or []
    if isinstance(reg, str):
        reg = [reg]
    if any(r not in REGIMES for r in reg):
        errs.append("regime gates must be among spy200, breadth, vixcalm")
    allowed = FRAME_NAMES | MARKET_NAMES | CAL_NAMES
    for e in exprs:
        try:
            tree = ast.parse(str(subst(e, pr)), mode="eval")
        except Exception as ex:
            errs.append(f"can't read '{e}': {ex}")
            continue
        names, funcs = _names(tree)
        bad = names - allowed
        if bad:
            errs.append(f"unknown field(s) {sorted(bad)} in '{e}'")
        badf = funcs - set(FUNCS)
        if badf:
            errs.append(f"unknown function(s) {sorted(badf)} in '{e}'")
    try:
        for c in cs:
            dyn = str(subst(spec.get("exit", "none"), c)).strip("'\"")
            if dyn not in EXITS:
                errs.append(f"exit must be one of {sorted(EXITS)}")
                break
            _num(spec.get("max_hold", 7), c, int)
            if "atr" in st:
                _num(st["atr"], c)
            if "atr" in tg:
                _num(tg["atr"], c)
    except Exception as e:
        errs.append(f"params: {e}")
    return errs


def signature(spec: dict) -> str:
    """Rule fingerprint used to spot a duplicate idea under a different name."""
    pr = (combos(spec) or [{}])[0]
    conds = sorted(re.sub(r"\s+", "", str(subst(c, pr))) for c in spec.get("conditions") or [])
    en = spec.get("entry") or {}
    return json.dumps([spec.get("universe", "all"), en.get("kind", "limit_next"), conds])


def describe(spec: dict) -> str:
    """One plain-English line of the rules, for the research log."""
    pr = (combos(spec) or [{}])[0]
    en = spec.get("entry") or {}
    ent = "buy near the close" if en.get("kind") == "close" else f"buy limit at {subst(en.get('price', 'close'), pr)} next session"
    st, tg = spec.get("stop") or {}, spec.get("target") or {}
    stop = f"stop {subst(st['atr'], pr)} ATR" if "atr" in st else f"stop at {subst(st.get('price'), pr)}"
    tgt = f"target {subst(tg['atr'], pr)} ATR" if "atr" in tg else f"target {subst(tg.get('price'), pr)}"
    return (f"{spec.get('universe', 'all')} | when " + " and ".join(str(subst(c, pr)) for c in spec.get("conditions") or [])
            + f" | {ent} | {stop}, {tgt}, exit {subst(spec.get('exit', 'none'), pr)}, max {subst(spec.get('max_hold', 7), pr)} days")
