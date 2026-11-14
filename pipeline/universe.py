"""Scan universe: S&P 500 + liquid growth names + index/sector/thematic ETFs (the core tier), and in "wide"
mode every other big, liquid US-listed stock plus more ETFs (the extension tier), screened daily."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import pathlib
import urllib.request

SP500_SOURCES = [
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
]

# Liquid, widely-followed names outside the S&P 500 (kept short on purpose).
EXTRA_STOCKS = {
    "ASML": ("ASML Holding", "Information Technology"), "TSM": ("Taiwan Semiconductor", "Information Technology"),
    "ARM": ("Arm Holdings", "Information Technology"), "SHOP": ("Shopify", "Information Technology"),
    "MELI": ("MercadoLibre", "Consumer Discretionary"), "PDD": ("PDD Holdings", "Consumer Discretionary"),
    "BABA": ("Alibaba", "Consumer Discretionary"), "SNOW": ("Snowflake", "Information Technology"),
    "NET": ("Cloudflare", "Information Technology"), "MDB": ("MongoDB", "Information Technology"),
    "ZS": ("Zscaler", "Information Technology"), "TEAM": ("Atlassian", "Information Technology"),
    "RBLX": ("Roblox", "Communication Services"), "SOFI": ("SoFi Technologies", "Financials"),
    "AFRM": ("Affirm", "Financials"), "ROKU": ("Roku", "Communication Services"),
    "DKNG": ("DraftKings", "Consumer Discretionary"), "NU": ("Nu Holdings", "Financials"),
    "SE": ("Sea Ltd", "Communication Services"), "CPNG": ("Coupang", "Consumer Discretionary"),
    "PINS": ("Pinterest", "Communication Services"), "ONON": ("On Holding", "Consumer Discretionary"),
    "CAVA": ("CAVA Group", "Consumer Discretionary"), "DUOL": ("Duolingo", "Consumer Discretionary"),
    "CRDO": ("Credo Technology", "Information Technology"), "RKLB": ("Rocket Lab", "Industrials"),
    "HIMS": ("Hims & Hers Health", "Health Care"), "CELH": ("Celsius Holdings", "Consumer Staples"),
    "ELF": ("e.l.f. Beauty", "Consumer Staples"), "CHWY": ("Chewy", "Consumer Discretionary"),
    "MRVL": ("Marvell Technology", "Information Technology"), "SPOT": ("Spotify", "Communication Services"),
    "NVO": ("Novo Nordisk", "Health Care"), "MSTR": ("Strategy (MicroStrategy)", "Information Technology"),
}

ETFS = {
    # broad
    "SPY": "S&P 500", "QQQ": "Nasdaq-100", "IWM": "Russell 2000", "DIA": "Dow 30", "RSP": "S&P 500 Equal Weight",
    "MDY": "S&P MidCap 400", "IJR": "S&P SmallCap 600", "SPYM": "S&P 500 (low price, formerly SPLG)", "QQQM": "Nasdaq-100 (low price)",
    # sectors
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care", "XLI": "Industrials",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communication Services",
    # industries / themes
    "SMH": "Semiconductors", "SOXX": "Semiconductors (iShares)", "XBI": "Biotech", "IBB": "Biotech (iShares)",
    "KRE": "Regional Banks", "KBE": "Banks", "XHB": "Homebuilders", "ITB": "Home Construction", "XRT": "Retail",
    "XOP": "Oil & Gas E&P", "OIH": "Oil Services", "GDX": "Gold Miners", "GDXJ": "Junior Gold Miners",
    "XME": "Metals & Mining", "COPX": "Copper Miners", "URA": "Uranium", "IGV": "Software", "CIBR": "Cybersecurity",
    "ITA": "Aerospace & Defense", "XAR": "Aerospace & Defense (EW)", "IYT": "Transportation", "PAVE": "Infrastructure",
    "TAN": "Solar", "ICLN": "Clean Energy", "JETS": "Airlines", "ARKK": "ARK Innovation", "VNQ": "REITs",
    # commodities / bonds / international
    "GLD": "Gold", "SLV": "Silver", "USO": "Oil", "TLT": "20+yr Treasuries", "IEF": "7-10yr Treasuries",
    "HYG": "High Yield Bonds", "EEM": "Emerging Markets", "EFA": "Developed ex-US", "FXI": "China Large Cap",
    "KWEB": "China Internet", "EWZ": "Brazil", "EWJ": "Japan", "INDA": "India", "IBIT": "Bitcoin",
}
LEVERAGED = {"TQQQ": "Nasdaq-100 3x", "UPRO": "S&P 500 3x", "SOXL": "Semiconductors 3x", "TNA": "Russell 2000 3x",
             "TECL": "Technology 3x", "FAS": "Financials 3x", "LABU": "Biotech 3x"}
INDEXES = {"^VIX": "CBOE Volatility Index"}


# ----------------------------------------------------------------------------- wide universe (since Nov 14, 2026)
# The desk scanned S&P 500 + 33 extras + 68 ETFs (605 tickers). That left most of the liquid US market unseen,
# and A+ setups scale with how many liquid names are watched. The wide universe adds every US-listed common
# stock and ADR that is big and liquid enough to trade (screened daily, with hysteresis so the list is stable)
# and more distinct ETFs. Every item carries a tier: "core" (the original universe) or "ext". Market context
# (breadth, relative-strength ranks) is always measured on the core tier, so core signals are identical with
# or without the extension, and the extension can be judged, paused or narrowed on its own record.
DEFAULT_MODE = "core"      # "core" | "wide"; the workflow input or UNIVERSE env overrides it

NASDAQ_SCREENER = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true"
ISHARES = {  # fallback when the screener is unreachable: index-fund holdings (no market cap or volume)
    "IWB": "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund",
    "IJH": "https://www.ishares.com/us/products/239763/ishares-core-sp-midcap-etf/1467271812596.ajax?fileType=csv&fileName=IJH_holdings&dataType=fund",
}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# entry and stay thresholds (hysteresis keeps the list from flapping around a line)
EXT_RULES = {"cap_in": 2.0e9, "cap_stay": 1.0e9, "price_in": 8.0, "price_stay": 5.0,
             "dv_in": 25e6, "dv_stay": 12e6, "keep_days": 90, "max_ext": 1600}

NASDAQ_SECTOR = {  # Nasdaq's sector names -> the GICS names the engine keys sector ETFs and caps on
    "Technology": "Information Technology", "Health Care": "Health Care", "Finance": "Financials",
    "Consumer Discretionary": "Consumer Discretionary", "Consumer Staples": "Consumer Staples",
    "Industrials": "Industrials", "Energy": "Energy", "Utilities": "Utilities", "Real Estate": "Real Estate",
    "Basic Materials": "Materials", "Telecommunications": "Communication Services",
}
ISHARES_SECTOR = {"Communication": "Communication Services", "Information Technology": "Information Technology",
                  "Health Care": "Health Care", "Financials": "Financials", "Industrials": "Industrials",
                  "Consumer Discretionary": "Consumer Discretionary", "Consumer Staples": "Consumer Staples",
                  "Energy": "Energy", "Utilities": "Utilities", "Real Estate": "Real Estate", "Materials": "Materials"}
_NOT_COMMON = ("warrant", " right", "rights", " unit", "units", "preferred", "depositary shares each",
               "% ", "notes due", "debenture", "acquisition corp", " etn", "trust preferred", "subordinated")

# More distinct, liquid ETFs (no clones of ones already listed). The engine's own liquidity filter
# (20-day dollar volume, price) decides day by day which are tradable.
ETFS_EXT = {
    # countries / regions
    "EWT": "Taiwan", "EWY": "South Korea", "EWG": "Germany", "EWU": "United Kingdom", "EWC": "Canada",
    "EWA": "Australia", "EWW": "Mexico", "MCHI": "China (MSCI)", "EWH": "Hong Kong", "VGK": "Europe",
    "EZU": "Eurozone", "VWO": "Emerging Markets (Vanguard)", "ASHR": "China A-Shares", "EPI": "India (earnings)",
    "ARGT": "Argentina", "EWP": "Spain", "EWI": "Italy", "EWQ": "France", "EWL": "Switzerland", "ILF": "Latin America",
    # styles / factors / size
    "IWF": "Russell 1000 Growth", "IWD": "Russell 1000 Value", "IWO": "Russell 2000 Growth", "IWN": "Russell 2000 Value",
    "MTUM": "Momentum", "QUAL": "Quality", "USMV": "Minimum Volatility", "VLUE": "Value Factor", "SCHD": "Dividend Equity",
    "VIG": "Dividend Appreciation", "IJH": "S&P MidCap 400 (iShares)", "VB": "Small Cap (Vanguard)", "VO": "Mid Cap (Vanguard)",
    "IWB": "Russell 1000", "SPHB": "S&P 500 High Beta", "SPLV": "S&P 500 Low Volatility",
    # industries / themes
    "XSD": "Semiconductors (EW)", "IHI": "Medical Devices", "IHF": "Health Care Providers", "XPH": "Pharmaceuticals",
    "XHE": "Health Care Equipment", "KIE": "Insurance", "IAI": "Broker-Dealers", "IYR": "Real Estate (iShares)",
    "XES": "Oil Equipment & Services", "AMLP": "Energy MLPs", "FDN": "Internet", "SKYY": "Cloud Computing",
    "HACK": "Cybersecurity (HACK)", "BOTZ": "Robotics & AI", "LIT": "Lithium & Battery", "REMX": "Rare Earths",
    "SIL": "Silver Miners", "SILJ": "Junior Silver Miners", "ARKG": "ARK Genomic", "ARKW": "ARK Next Gen Internet",
    "FINX": "FinTech", "XTN": "Transportation (EW)", "MOO": "Agribusiness", "NLR": "Nuclear Energy",
    "URNM": "Uranium Miners", "BLOK": "Blockchain", "IGM": "Tech-Software Sector", "XSW": "Software & Services (EW)",
    "IYW": "US Technology", "VGT": "Information Technology (Vanguard)", "XLG": "S&P 500 Top 50", "DRIV": "Autonomous & EV",
    "PPH": "Pharma (VanEck)", "IYF": "US Financials", "IYH": "US Healthcare", "FCG": "Natural Gas Producers",
    # bonds / commodities / currencies / crypto
    "LQD": "Investment Grade Corporates", "EMB": "EM Bonds (USD)", "TIP": "TIPS", "JNK": "High Yield (SPDR)",
    "IAU": "Gold (iShares)", "PPLT": "Platinum", "CPER": "Copper", "UNG": "Natural Gas", "DBC": "Commodities",
    "DBA": "Agriculture", "UUP": "US Dollar", "ETHA": "Ethereum", "FBTC": "Bitcoin (Fidelity)",
}
# The same index in two wrappers: only the best-ranked one is featured, the other is listed as its twin.
TWINS = {"SPYM": "SPY", "QQQM": "QQQ", "SOXX": "SMH", "IBB": "XBI", "KBE": "KRE", "ITB": "XHB", "XAR": "ITA",
         "GDXJ": "GDX", "IAU": "GLD", "FBTC": "IBIT", "JNK": "HYG", "IJH": "MDY", "HACK": "CIBR", "IYR": "VNQ",
         "VGT": "XLK", "IYW": "XLK", "IYF": "XLF", "IYH": "XLV", "SILJ": "SIL", "VWO": "EEM", "MCHI": "FXI"}


def _fetch(url, timeout=60, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _num(x):
    try:
        return float(str(x).replace("$", "").replace(",", "").replace("%", "").strip())
    except Exception:
        return None


def _plain_symbol(sym: str) -> bool:
    return bool(sym) and sym.isalpha() and sym.isupper() and len(sym) <= 5


def screener_rows() -> list[dict]:
    """Every US-listed stock from Nasdaq's screener: symbol, name, last sale, volume, market cap, sector."""
    hdr = {"User-Agent": UA, "Accept": "application/json, text/plain, */*",
           "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
    js = json.loads(_fetch(NASDAQ_SCREENER, timeout=60, headers=hdr))
    data = (js or {}).get("data") or {}
    rows = data.get("rows") or (data.get("table") or {}).get("rows") or []
    out = []
    for r in rows:
        sym = (r.get("symbol") or "").strip().upper()
        name = (r.get("name") or "").strip()
        px, vol, cap = _num(r.get("lastsale")), _num(r.get("volume")), _num(r.get("marketCap"))
        if not _plain_symbol(sym) or px is None:
            continue
        low = " " + name.lower() + " "
        if any(k in low for k in _NOT_COMMON):
            continue
        out.append({"ticker": sym, "name": name[:60], "price": px, "dv": (px * vol) if vol else 0.0,
                    "cap": cap or 0.0, "sector": NASDAQ_SECTOR.get((r.get("sector") or "").strip())})
    return out


def ishares_rows() -> list[dict]:
    out, seen = [], set()
    for etf, url in ISHARES.items():
        try:
            txt = _fetch(url, timeout=60)
            lines = txt.splitlines()
            start = next(i for i, ln in enumerate(lines) if ln.startswith("Ticker,"))
            for r in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
                sym = (r.get("Ticker") or "").strip().upper()
                if not _plain_symbol(sym) or sym in seen or (r.get("Asset Class") or "Equity") != "Equity":
                    continue
                seen.add(sym)
                out.append({"ticker": sym, "name": (r.get("Name") or sym).title()[:60], "price": _num(r.get("Price")),
                            "dv": None, "cap": None, "sector": ISHARES_SECTOR.get((r.get("Sector") or "").strip())})
        except Exception:
            continue
    return out


def _own_liquidity(daily_path: pathlib.Path | None) -> dict:
    """Median 20-day dollar volume and last close per ticker from the feed's own daily file (steadier than one
    day's screener volume)."""
    if not daily_path or not daily_path.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(daily_path, usecols=["date", "ticker", "close", "volume"])
        last = df["date"].max()
        df = df[df["date"] >= str((pd.Timestamp(last) - pd.Timedelta(days=32)).date())]
        df["dv"] = df["close"] * df["volume"]
        g = df.groupby("ticker")
        return {t: {"dv": float(d), "price": float(c)} for (t, d), c in zip(g["dv"].median().items(), g["close"].last())}
    except Exception:
        return {}


def build_ext(core: set, previous_items: list[dict], today: str, daily_path: pathlib.Path | None = None,
              rows: list[dict] | None = None) -> tuple[list[dict], dict]:
    """The extension tier: big, liquid US-listed stocks outside the core. Returns (items, stats)."""
    R = EXT_RULES
    src = "nasdaq-screener"
    if rows is None:
        try:
            rows = screener_rows()
        except Exception as e:  # noqa: BLE001
            rows, src = [], f"screener failed: {e!r}"[:120]
    if len(rows) < 2000:  # blocked or broken: index-fund holdings, else the last good list
        alt = ishares_rows() if not rows else []
        if len(alt) > 600:
            rows, src = alt, "ishares-holdings"
        else:
            keep = [i for i in previous_items if i.get("tier") == "ext" and i.get("type") == "stock"]
            return keep, {"source": f"kept previous ({src})", "ext": len(keep), "added": 0, "dropped": 0}
    own = _own_liquidity(daily_path)
    prev = {i["ticker"]: i for i in previous_items if i.get("tier") == "ext" and i.get("type") == "stock"}
    cutoff = (dt.date.fromisoformat(today) - dt.timedelta(days=R["keep_days"])).isoformat()
    items, seen = [], set()
    for r in rows:
        t = r["ticker"]
        if t in core or t in seen:
            continue
        seen.add(t)
        o = own.get(t) or {}
        dv = o.get("dv") if o.get("dv") else r.get("dv")
        px = o.get("price") or r.get("price")
        cap = r.get("cap")
        was = prev.get(t)
        if src == "ishares-holdings":  # no cap or volume: membership of the Russell 1000 / S&P 400 is the screen
            ok_in = ok_stay = (px or 0) >= R["price_stay"] and (dv is None or dv >= R["dv_stay"])
        else:
            ok_in = (cap or 0) >= R["cap_in"] and (px or 0) >= R["price_in"] and (dv or 0) >= R["dv_in"]
            ok_stay = (cap or 0) >= R["cap_stay"] and (px or 0) >= R["price_stay"] and (dv or 0) >= R["dv_stay"]
        if ok_in or (was and ok_stay):
            last_ok = today
        elif was and str(was.get("last_ok", "")) >= cutoff:
            last_ok = was["last_ok"]  # slipped below the line: kept for 90 days so open positions stay covered
        else:
            continue
        items.append({"ticker": t, "name": r["name"] or (was or {}).get("name") or t, "type": "stock",
                      "sector": r.get("sector") or (was or {}).get("sector"), "index": "EXT", "tier": "ext",
                      "cap_b": round(cap / 1e9, 2) if cap else (was or {}).get("cap_b"),
                      "dv_m": round(dv / 1e6, 1) if dv else None, "last_ok": last_ok})
    for t, was in prev.items():  # gone from the source today (halt, rename): same 90-day grace
        if t not in seen and t not in core and str(was.get("last_ok", "")) >= cutoff:
            items.append(was)
    items.sort(key=lambda i: -(i.get("dv_m") or 0))
    items = items[:R["max_ext"]]
    now = {i["ticker"] for i in items}
    return items, {"source": src, "rows": len(rows), "ext": len(items), "added": len(now - set(prev)),
                   "dropped": len(set(prev) - now), "added_tickers": sorted(now - set(prev))[:40],
                   "dropped_tickers": sorted(set(prev) - now)[:40]}


def build(previous: pathlib.Path | None = None, mode: str | None = None, today: str | None = None,
          daily_path: pathlib.Path | None = None, rows: list[dict] | None = None) -> list[dict]:
    """The scan universe. mode "core" = the original list; "wide" adds the extension tier."""
    mode = mode or DEFAULT_MODE
    today = today or dt.date.today().isoformat()
    old = []
    if previous and previous.exists():
        try:
            old = json.loads(previous.read_text()).get("items", [])
        except Exception:
            old = []
    items, seen = [], set()
    sp = []
    for src in SP500_SOURCES:
        try:
            rows_sp = list(csv.DictReader(io.StringIO(_fetch(src))))
            for r in rows_sp:
                sym = (r.get("Symbol") or "").strip().upper().replace(".", "-")
                if sym:
                    sp.append({"ticker": sym, "name": r.get("Security") or sym, "type": "stock",
                               "sector": r.get("GICS Sector"), "index": "SP500", "tier": "core"})
            if len(sp) > 400:
                break
        except Exception:
            sp = []
    if len(sp) < 400:  # keep last good list
        sp = [dict(i, tier="core") for i in old if i.get("index") == "SP500"]
    for i in sp:
        if i["ticker"] not in seen:
            items.append(i)
            seen.add(i["ticker"])
    for t, (name, sec) in EXTRA_STOCKS.items():
        if t not in seen:
            items.append({"ticker": t, "name": name, "type": "stock", "sector": sec, "index": "EXTRA", "tier": "core"})
            seen.add(t)
    stats = {"mode": mode}
    if mode == "wide":
        ext, st = build_ext(set(seen), old, today, daily_path, rows)
        stats.update(st)
        for i in ext:
            if i["ticker"] not in seen:
                items.append(i)
                seen.add(i["ticker"])
    for t, name in ETFS.items():
        items.append({"ticker": t, "name": name, "type": "etf", "sector": None, "index": "ETF", "tier": "core"})
        seen.add(t)
    if mode == "wide":
        for t, name in ETFS_EXT.items():
            if t not in seen:
                items.append({"ticker": t, "name": name, "type": "etf", "sector": None, "index": "ETF", "tier": "ext"})
                seen.add(t)
    for it in items:
        if it["ticker"] in TWINS and TWINS[it["ticker"]] in seen:
            it["twin_of"] = TWINS[it["ticker"]]
    for t, name in LEVERAGED.items():
        items.append({"ticker": t, "name": name, "type": "etf_lev", "sector": None, "index": "ETF", "tier": "core"})
    for t, name in INDEXES.items():
        items.append({"ticker": t, "name": name, "type": "index", "sector": None, "index": "INDEX", "tier": "core"})
    build.last_stats = stats
    return items


build.last_stats = {}
