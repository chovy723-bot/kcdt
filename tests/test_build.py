"""Offline check: synthetic prices + fundamentals -> build() -> rendered page."""
import numpy as np
import pandas as pd

from kcdt.compute import DOW30, INDICES, SECTORS, build, stock_pages, wants_fundamentals

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
        rev = sorted(rng.uniform(1e8, 1e10, 8).tolist(), reverse=bool(rng.integers(0, 2)))
        out[t] = dict(mcap=float(rng.uniform(3e8, 3e12)), beta=float(rng.uniform(0.3, 2.5)),
                      inst=float(rng.uniform(0, 1)), sq=float(rng.uniform(-0.2, 0.8)),
                      eq=float(rng.uniform(-0.5, 1.5)), pe=float(rng.uniform(5, 80)), ysec=None,
                      ipo=1.2e12, edate=1.8e9, revq=rev, epsq=[1.5, 1.2, 1.0, 0.8, 0.7], reva=[5e9, 4e9, 3e9],
                      epsa=[4.0, 3.0, 2.5], epsh=[1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4],
                      efc=[0.3, 0.2], rfc=[0.25, None])
    return out


def test_build():
    universe = [f"S{i:04d}" for i in range(1100)] + DOW30[:5]
    tickers = sorted(set(universe + DOW30 + [t for t, _ in INDICES] + [t for t, _ in SECTORS] + ["^NDX", "^VXN"]))
    px = fake_prices(tickers)
    # a partial "today" row where only VIX has traded
    extra = pd.Timestamp("2026-10-28")
    for k in px:
        px[k].loc[extra] = np.nan
    px["Close"].loc[extra, "^VIX"] = 16.0
    meta = fake_meta(universe)
    want = wants_fundamentals(px, meta)
    assert len(want) >= 100
    fng = dict(score=42.0, rating="fear", prev=40.1, w1=35.0, m1=55.2, y1=60.0, ts="2026-10-27",
               hist=[["2026-%02d-%02d" % (1 + i // 28, 1 + i % 28), 30 + (i * 7) % 50] for i in range(250)], live=True)
    d = build(px, meta, fake_fund(want), fng)
    assert d["asof"] == "2026-10-27"
    assert len(d["hist"]) == 60 and len(d["stocks"]) == 1105
    assert all(0 <= h["score"] <= 16 for h in d["hist"])
    assert d["breadth"]["ndx"]["n"] == 100 and d["nfund"] == len(want)
    assert len(d["minis"]) == 4 and len(d["minis"][0]["c"]) == 60
    ranks = [s["rsr"] for s in d["stocks"]]
    assert min(ranks) == 1 and max(ranks) == 99
    f0 = next(s for s in d["stocks"] if s.get("mcap") is not None)
    assert f0["g3e"] == 100.0 and f0["yr"] is True and f0["e0"] == 30.0 and f0["r1"] is None  # min of 1.5/0.7, 1.2/0.6, 1.0/0.5
    assert d["fng"]["score"] == 42.0
    assert len(d["tech"]["hist"]) == 60 and 0 <= d["tech"]["hist"][-1]["score"] <= 16
    assert set(d["panic"]) == {"on", "day"}
    pages = stock_pages(px, d["stocks"])
    nv = pages[d["stocks"][0]["s"]]
    assert len(pages) == 1105 and len(nv["d"]) == 260 and nv["d"][-1] == "2026-10-27"
    assert nv["m200"][-1] is not None and len(nv["v"]) == 260
    return d, pages


if __name__ == "__main__":
    import pathlib
    from kcdt import __main__ as m
    d, pages = test_build()
    m.OUT = pathlib.Path("site")
    m.OUT.mkdir(exist_ok=True)
    (m.OUT / "index.html").write_text(m.render(d), encoding="utf-8")
    m.write_stock_pages(pages)
    print("ok", d["asof"], d["hist"][-1]["score"], len(d["stocks"]), d["breadth"])
