"""Download prices from Yahoo Finance (free, no key) and the S&P 500 list from Wikipedia."""
import io
import time

import pandas as pd
import requests
import yfinance as yf

from .compute import DOW30, INDICES, SECTORS

WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500():
    html = requests.get(WIKI, headers={"User-Agent": "Mozilla/5.0 KCDT"}, timeout=30).text
    t = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    syms = [s.replace(".", "-") for s in t["Symbol"]]  # BRK.B -> BRK-B (Yahoo style)
    return syms, dict(zip(syms, t["Security"]))


def _download(tickers, period):
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=False)
    return df["Close"], df["High"]


def prices(tickers, period="2y", chunk=50, pause=3):
    """Download in small batches with pauses; Yahoo rate-limits big bursts from cloud IPs."""
    closes, highs = [], []
    todo = list(tickers)
    for attempt in range(4):
        missing = []
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            try:
                c, h = _download(part, period)
            except Exception as e:
                print("batch failed:", e)
                missing += part
                time.sleep(pause * 10)
                continue
            got = [t for t in part if t in c and not c[t].dropna().empty]
            closes.append(c[got]); highs.append(h[got])
            missing += [t for t in part if t not in got]
            time.sleep(pause)
        print(f"attempt {attempt + 1}: {len(tickers) - len(missing)}/{len(tickers)} downloaded")
        if not missing:
            break
        todo = missing
        time.sleep(60 * (attempt + 1))
    if len(missing) > len(tickers) * 0.05:
        raise RuntimeError(f"{len(missing)} tickers missing, e.g. {missing[:10]}")
    close, high = pd.concat(closes, axis=1), pd.concat(highs, axis=1)
    return close.loc[:, ~close.columns.duplicated()], high.loc[:, ~high.columns.duplicated()]


def fetch_all():
    universe, names = sp500()
    extra = [t for t, _ in INDICES] + [t for t, _ in SECTORS] + DOW30
    tickers = sorted(set(universe + extra))
    close, high = prices(tickers)
    return close, high, universe, names
