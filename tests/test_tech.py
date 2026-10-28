import numpy as np
import pandas as pd

from kcdt.tech import TECH_RULES, panic, tech_indicator_frame, tech_score


def test_tech_and_panic():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range(end="2026-10-27", periods=400)
    cols = ["^NDX", "^VXN"] + [f"N{i}" for i in range(100)]
    close = pd.DataFrame(100 * np.exp(rng.normal(0.0005, 0.02, (400, len(cols))).cumsum(0)), index=idx, columns=cols)
    close["^VXN"] = 20 + 5 * rng.random(400)
    high = close * 1.01
    ind = tech_indicator_frame(close, high, [f"N{i}" for i in range(100)]).dropna()
    s, pts = tech_score(ind.iloc[-1])
    assert 0 <= s <= 16 and set(pts) == {k for k, *_ in TECH_RULES}
    spx = pd.DataFrame({"vix5": [20, 30, 31, 32], "rv21": [20, 26, 27, 28]})
    assert panic(spx) == {"on": True, "day": 3}
    assert panic(spx.iloc[:1]) == {"on": False, "day": 0}
