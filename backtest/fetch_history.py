"""Download daily history since 2003 for the temperature backtest -> backtest/data/*.csv.gz

Runs in GitHub Actions (Yahoo is reachable there); the analysis itself is offline.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kcdt.compute import DOW30  # noqa: E402
from kcdt.fetch import prices  # noqa: E402

SECTOR9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]  # all trade since 1998
TICKERS = ["^GSPC", "^VIX", "^NDX", "QQQ", "SPY"] + SECTOR9 + DOW30
OUT = pathlib.Path(__file__).resolve().parent / "data"


def main():
    import yfinance as yf
    yf_download = yf.download

    def since_2003(tickers, period=None, **kw):  # period is ignored: take everything since 2003
        return yf_download(tickers, start="2003-01-01", **kw)

    yf.download = since_2003
    px = prices(TICKERS, chunk=10)
    OUT.mkdir(exist_ok=True)
    for f in ("Close", "High"):
        df = px[f].sort_index()
        df.index = df.index.strftime("%Y-%m-%d")
        df.round(4).to_csv(OUT / f"{f.lower()}.csv.gz")
    c = px["Close"]
    print("rows", len(c), c.index.min(), c.index.max())
    print(c.apply(lambda s: s.first_valid_index()).to_string())


if __name__ == "__main__":
    main()
