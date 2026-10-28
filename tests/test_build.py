"""Offline check: synthetic prices -> build() -> rendered page."""
import numpy as np
import pandas as pd

from kcdt.compute import DOW30, INDICES, SECTORS, build


def fake_prices(tickers, days=500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-27", periods=days)
    rets = rng.normal(0.0004, 0.015, (days, len(tickers)))
    close = pd.DataFrame(100 * np.exp(rets.cumsum(0)), index=idx, columns=tickers)
    close["^VIX"] = 15 + 5 * rng.random(days)
    high = close * (1 + rng.random(close.shape) * 0.01)
    return close, high


def test_build():
    universe = [f"S{i:03d}" for i in range(500)] + DOW30[:5]
    tickers = sorted(set(universe + DOW30 + [t for t, _ in INDICES] + [t for t, _ in SECTORS]))
    close, high = fake_prices(tickers)
    d = build(close, high, universe)
    assert len(d["hist"]) == 60 and len(d["stocks"]) == 505
    assert all(0 <= h["score"] <= 16 for h in d["hist"])
    assert len(d["spx"]["close"]) == 252 and len(d["sectors"]) == 12
    ranks = [s["rsr"] for s in d["stocks"]]
    assert min(ranks) == 1 and max(ranks) == 99
    return d


if __name__ == "__main__":
    import pathlib
    from kcdt.__main__ import render
    d = test_build()
    pathlib.Path("site").mkdir(exist_ok=True)
    pathlib.Path("site/index.html").write_text(render(d), encoding="utf-8")
    print("ok", d["asof"], d["hist"][-1]["score"], len(d["stocks"]))
