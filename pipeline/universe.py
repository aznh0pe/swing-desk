"""Scan universe: S&P 500 + liquid growth names + index/sector/thematic ETFs."""
from __future__ import annotations

import csv
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
    "MDY": "S&P MidCap 400", "IJR": "S&P SmallCap 600", "SPLG": "S&P 500 (low price)", "QQQM": "Nasdaq-100 (low price)",
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


def _fetch(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def build(previous: pathlib.Path | None = None) -> list[dict]:
    items, seen = [], set()
    sp = []
    for src in SP500_SOURCES:
        try:
            rows = list(csv.DictReader(io.StringIO(_fetch(src))))
            for r in rows:
                sym = (r.get("Symbol") or "").strip().upper().replace(".", "-")
                if sym:
                    sp.append({"ticker": sym, "name": r.get("Security") or sym, "type": "stock",
                               "sector": r.get("GICS Sector"), "index": "SP500"})
            if len(sp) > 400:
                break
        except Exception:
            sp = []
    if len(sp) < 400 and previous and previous.exists():  # keep last good list
        try:
            old = json.loads(previous.read_text()).get("items", [])
            sp = [i for i in old if i.get("index") == "SP500"]
        except Exception:
            pass
    for i in sp:
        if i["ticker"] not in seen:
            items.append(i)
            seen.add(i["ticker"])
    for t, (name, sec) in EXTRA_STOCKS.items():
        if t not in seen:
            items.append({"ticker": t, "name": name, "type": "stock", "sector": sec, "index": "EXTRA"})
            seen.add(t)
    for t, name in ETFS.items():
        items.append({"ticker": t, "name": name, "type": "etf", "sector": None, "index": "ETF"})
    for t, name in LEVERAGED.items():
        items.append({"ticker": t, "name": name, "type": "etf_lev", "sector": None, "index": "ETF"})
    for t, name in INDEXES.items():
        items.append({"ticker": t, "name": name, "type": "index", "sector": None, "index": "INDEX"})
    return items
