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


def prices(tickers, period="2y", tries=3):
    for i in range(tries):
        try:
            df = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
            close, high = df["Close"], df["High"]
            missing = [t for t in tickers if t not in close or close[t].dropna().empty]
            if len(missing) > len(tickers) * 0.05:
                raise RuntimeError(f"{len(missing)} tickers missing, e.g. {missing[:10]}")
            return close, high
        except Exception as e:  # Yahoo rate-limits now and then; back off and retry
            if i == tries - 1:
                raise
            print("download failed, retrying:", e)
            time.sleep(30 * (i + 1))


def fetch_all():
    universe, names = sp500()
    extra = [t for t, _ in INDICES] + [t for t, _ in SECTORS] + DOW30
    tickers = sorted(set(universe + extra))
    close, high = prices(tickers)
    return close, high, universe, names
