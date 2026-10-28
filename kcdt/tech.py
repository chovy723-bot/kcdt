"""Tech temperature (Nasdaq-100 version of the 8-indicator score) and the panic hint.

Thresholds come from backtest/analyze_tech.py: each tech yellow / red sits at the same historical
rarity (2004–2015) as the S&P one. Needs ^NDX and ^VXN in the price download.
"""
import numpy as np
import pandas as pd

from .compute import points

# key, label, yellow, red, direction ("high" = bigger is worse), unit
TECH_RULES = [
    ("vix5", "VXN 5 日均", 23, 27, "high", ""),
    ("vs50", "Nasdaq-100 vs 50 日線", 0, -4, "low", "%"),
    ("pct50", "Nasdaq-100 成份股 % 企 50 日線上", 50, 30, "low", "%"),
    ("dist52", "Nasdaq-100 成份股平均距 52 週高", -15, -25, "low", "%"),
    ("pct200", "Nasdaq-100 成份股 % 企 200 日線上", 55, 40, "low", "%"),
    ("ma50_200", "Nasdaq-100 50 日線 vs 200 日線", 1, 0, "low", "%"),
    ("rv21", "Nasdaq-100 21 日波幅（年化）", 21, 27, "high", "%"),
    ("spx52", "Nasdaq-100 距 52 週高", -7, -11, "low", "%"),
]

# Panic: VIX 5-day average and S&P 21-day volatility both at the S&P temperature's red level.
PANIC_VIX5, PANIC_RV21 = 25, 25


def tech_indicator_frame(close, high, members):
    """The 8 tech indicators as daily series (percent units). `members` = Nasdaq-100 tickers."""
    ix, ix_h = close["^NDX"], high["^NDX"]
    ma50, ma200 = ix.rolling(50).mean(), ix.rolling(200).mean()
    b = close[[t for t in members if t in close]]
    bh = high[b.columns]
    m50, m200 = b.rolling(50).mean(), b.rolling(200).mean()
    h252 = bh.rolling(252).max().where(bh.notna().rolling(252).sum() >= 252)
    df = pd.DataFrame(index=close.index)
    df["vix5"] = close["^VXN"].rolling(5).mean()
    df["vs50"] = (ix / ma50 - 1) * 100
    df["pct50"] = (b > m50).where(m50.notna()).mean(axis=1) * 100
    df["dist52"] = (b / h252 - 1).mean(axis=1) * 100
    df["pct200"] = (b > m200).where(m200.notna()).mean(axis=1) * 100
    df["ma50_200"] = (ma50 / ma200 - 1) * 100
    df["rv21"] = np.log(ix).diff().rolling(21).std() * np.sqrt(252) * 100
    df["spx52"] = (ix / ix_h.rolling(252).max() - 1) * 100
    return df


def tech_score(row):
    """row: one day of tech_indicator_frame -> (score 0–16, {key: points})."""
    pts = {k: points(float(row[k]), y, r, s) for k, _n, y, r, s, _u in TECH_RULES}
    return sum(pts.values()), pts


def panic(ind):
    """ind: the S&P indicator_frame. Returns {"on": bool, "day": consecutive panic days (0 if off)}."""
    on = (ind["vix5"] >= PANIC_VIX5) & (ind["rv21"] >= PANIC_RV21)
    run = on.groupby((~on).cumsum()).cumsum()
    return {"on": bool(on.iloc[-1]), "day": int(run.iloc[-1])}


def rebound(qqq):
    """Rebound confirmation: QQQ closes above its 20-day line and that line is higher than 5 days ago.
    qqq: daily closes. Returns {"on", "day" (consecutive days on), "ma20", "close"}."""
    m20 = qqq.rolling(20).mean()
    on = (qqq > m20) & (m20.diff(5) > 0)
    run = on.groupby((~on).cumsum()).cumsum()
    return {"on": bool(on.iloc[-1]), "day": int(run.iloc[-1]),
            "ma20": round(float(m20.iloc[-1]), 2), "close": round(float(qqq.iloc[-1]), 2)}
