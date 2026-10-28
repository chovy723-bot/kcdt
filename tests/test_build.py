"""Offline check: synthetic prices + fundamentals -> build() -> rendered page."""
import numpy as np
import pandas as pd

from kcdt.compute import DOW30, INDICES, SECTORS, build, wants_fundamentals

GICS = ["Information Technology", "Communication Services", "Consumer Discretionary",
        "Financials", "Health Care", "Industrials", "Energy"]


def fake_prices(tickers, days=500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-27", periods=days)
    rets = rng.normal(0.0005, 0.018, (days, len(tickers)))
    close = pd.DataFrame(100 * np.exp(rets.cumsum(0)), index=idx, columns=tickers)
    close["^VIX"] = 15 + 5 * rng.random(days)
    high = close * (1 + rng.random(close.shape) * 0.01)
    low = close * (1 - rng.random(close.shape) * 0.01)
    opn = close.shift(1).fillna(close)
    vol = pd.DataFrame(rng.integers(20_000, 5_000_000, close.shape), index=idx, columns=tickers).astype(float)
    return {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol}


def fake_meta(universe):
    return pd.DataFrame({
        "n": [f"Company {t}" for t in universe],
        "gics": [GICS[i % len(GICS)] for i in range(len(universe))],
        "ndx": [i < 100 for i in range(len(universe))],
        "sp500": True, "sp400": False, "sp600": False,
    }, index=universe)


def fake_fund(tickers, seed=2):
    rng = np.random.default_rng(seed)
    out = {}
    for t in tickers:
        rev = sorted(rng.uniform(1e8, 1e10, 6).tolist(), reverse=bool(rng.integers(0, 2)))
        out[t] = dict(mcap=float(rng.uniform(3e8, 3e12)), beta=float(rng.uniform(0.3, 2.5)),
                      inst=float(rng.uniform(0, 1)), sq=float(rng.uniform(-0.2, 0.8)),
                      eq=float(rng.uniform(-0.5, 1.5)), pe=float(rng.uniform(5, 80)), ysec=None,
                      ipo=1.2e12, edate=1.8e9, revq=rev, epsq=[1.5, 1.2, 1.0, 0.8, 0.7], reva=[5e9, 4e9])
    return out


def test_build():
    universe = [f"S{i:04d}" for i in range(1100)] + DOW30[:5]
    tickers = sorted(set(universe + DOW30 + [t for t, _ in INDICES] + [t for t, _ in SECTORS]))
    px = fake_prices(tickers)
    # a partial "today" row where only VIX has traded
    extra = pd.Timestamp("2026-10-28")
    for k in px:
        px[k].loc[extra] = np.nan
    px["Close"].loc[extra, "^VIX"] = 16.0
    meta = fake_meta(universe)
    want = wants_fundamentals(px, meta)
    assert len(want) >= 100
    d = build(px, meta, fake_fund(want))
    assert d["asof"] == "2026-10-27"
    assert len(d["hist"]) == 60 and len(d["stocks"]) == 1105
    assert all(0 <= h["score"] <= 16 for h in d["hist"])
    assert d["breadth"]["ndx"]["n"] == 100 and d["nfund"] == len(want)
    assert len(d["minis"]) == 4 and len(d["minis"][0]["c"]) == 60
    ranks = [s["rsr"] for s in d["stocks"]]
    assert min(ranks) == 1 and max(ranks) == 99
    return d


if __name__ == "__main__":
    import pathlib
    from kcdt.__main__ import render
    d = test_build()
    pathlib.Path("site").mkdir(exist_ok=True)
    pathlib.Path("site/index.html").write_text(render(d), encoding="utf-8")
    print("ok", d["asof"], d["hist"][-1]["score"], len(d["stocks"]), d["breadth"])
