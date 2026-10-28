"""Turn daily price history into the dashboard's data object.

Inputs are wide DataFrames (index = trading dates, columns = tickers) of
split/dividend-adjusted closes and highs. Nothing here touches the network,
so it can be tested with any price history.
"""
import numpy as np
import pandas as pd

DOW30 = ["AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
         "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
         "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"]

SECTORS = [("XLK", "科技"), ("SMH", "半導體"), ("XLC", "通訊"), ("XLY", "非必需消費"),
           ("XLF", "金融"), ("XLI", "工業"), ("XLV", "醫療"), ("XLE", "能源"),
           ("XLP", "必需消費"), ("XLU", "公用"), ("XLB", "原材料"), ("XLRE", "房地產")]

INDICES = [("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq 綜合"), ("^DJI", "道指"),
           ("IWM", "羅素 2000 (IWM)"), ("RSP", "等權 S&P (RSP)"), ("^VIX", "VIX")]

# key, label, yellow, red, direction ("high" = bigger is worse), unit
RULES = [
    ("vix5", "VIX 5 日均", 20, 25, "high", ""),
    ("vs50", "標普 vs 50 日線", 0, -3, "low", "%"),
    ("pct50", "30 隻龍頭股 % 企 50 日線上", 50, 30, "low", "%"),
    ("dist52", "30 隻龍頭股平均距 52 週高", -10, -20, "low", "%"),
    ("pct200", "30 隻龍頭股 % 企 200 日線上", 60, 40, "low", "%"),
    ("ma50_200", "標普 50 日線 vs 200 日線", 1, 0, "low", "%"),
    ("rv21", "標普 21 日波幅（年化）", 18, 25, "high", "%"),
    ("spx52", "標普距 52 週高", -5, -10, "low", "%"),
]

HIST_DAYS = 60


def points(v, yellow, red, side):
    if side == "high":
        return 2 if v >= red else 1 if v >= yellow else 0
    return 2 if v <= red else 1 if v <= yellow else 0


def indicator_frame(close, high, vix):
    """All 8 indicators as daily series (percent units)."""
    spx, spx_h = close["^GSPC"], high["^GSPC"]
    ma50, ma200 = spx.rolling(50).mean(), spx.rolling(200).mean()
    dow = close[[t for t in DOW30 if t in close]]
    dow_h = high[dow.columns]
    df = pd.DataFrame(index=close.index)
    df["vix5"] = vix.rolling(5).mean()
    df["vs50"] = (spx / ma50 - 1) * 100
    df["pct50"] = (dow > dow.rolling(50).mean()).mean(axis=1) * 100
    df["dist52"] = (dow / dow_h.rolling(252).max() - 1).mean(axis=1) * 100
    df["pct200"] = (dow > dow.rolling(200).mean()).mean(axis=1) * 100
    df["ma50_200"] = (ma50 / ma200 - 1) * 100
    df["rv21"] = np.log(spx).diff().rolling(21).std() * np.sqrt(252) * 100
    df["spx52"] = (spx / spx_h.rolling(252).max() - 1) * 100
    return df


def stock_table(close, high, universe):
    px = close[[t for t in universe if t in close]].dropna(axis=1, thresh=253)
    hi = high[px.columns]
    last, prev = px.iloc[-1], px.iloc[-2]
    ma50, ma200 = px.rolling(50).mean(), px.rolling(200).mean()
    hi252 = hi.rolling(252).max().iloc[-1]
    ret = {k: (last / px.iloc[-1 - k] - 1) * 100 for k in (21, 63, 126, 252)}
    rs = 0.4 * ret[63] + 0.2 * ret[126] + 0.2 * ret[252] + 0.2 * ret[21]
    rs_rank = (rs.rank(pct=True) * 98 + 1).round()
    rows = []
    for t in px.columns:
        c, m50, m200 = last[t], ma50[t].iloc[-1], ma200[t].iloc[-1]
        if np.isnan(c) or np.isnan(m200):
            continue
        tt = [c > m50, m50 > m200, c > m200, m200 > ma200[t].iloc[-21], c >= hi252[t] * 0.75]
        rows.append(dict(
            s=t, c=round(float(c), 2), chg=round(float((c / prev[t] - 1) * 100), 2),
            r21=round(float(ret[21][t]), 1), r63=round(float(ret[63][t]), 1),
            r126=round(float(ret[126][t]), 1), r252=round(float(ret[252][t]), 1),
            d50=round(float((c / m50 - 1) * 100), 1), d200=round(float((c / m200 - 1) * 100), 1),
            dh=round(float((c / hi252[t] - 1) * 100), 1),
            tt=int(sum(bool(x) for x in tt)), rsr=int(rs_rank[t]), dow=t in DOW30))
    return rows, px, ma50, ma200


def build(close, high, universe, sp500_names=None):
    """close/high: DataFrames with every ticker needed. universe: S&P 500 tickers."""
    close, high = close.sort_index(), high.sort_index()
    ind = indicator_frame(close, high, close["^VIX"]).dropna().tail(HIST_DAYS)
    stocks, px, ma50, ma200 = stock_table(close, high, universe)
    above50 = (px > ma50).where(ma50.notna()).mean(axis=1) * 100
    above200 = (px > ma200).where(ma200.notna()).mean(axis=1) * 100

    hist = []
    for d, row in ind.iterrows():
        v = {k: round(float(row[k]), 2) for k, *_ in RULES}
        p = {k: points(v[k], y, r, s) for k, _n, y, r, s, _u in RULES}
        hist.append(dict(d=d.strftime("%m-%d"), score=sum(p.values()), v=v, p=p,
                         b50=round(float(above50[d]), 1), b200=round(float(above200[d]), 1)))

    def chg(t, k):
        s = close[t].dropna()
        return round(float((s.iloc[-1] / s.iloc[-1 - k] - 1) * 100), 2)

    def ytd(t):
        s = close[t].dropna()
        base = s[s.index.year < s.index[-1].year]
        return round(float((s.iloc[-1] / base.iloc[-1] - 1) * 100), 2) if len(base) else None

    sectors = [dict(t=t, n=n, d1=chg(t, 1), d5=chg(t, 5), d10=chg(t, 10), h6=chg(t, 126), ytd=ytd(t))
               for t, n in SECTORS if t in close]
    idx = [dict(n=n, l=round(float(close[t].dropna().iloc[-1]), 2), p=round(float(close[t].dropna().iloc[-2]), 2))
           for t, n in INDICES if t in close]
    if sp500_names:
        for s in stocks:
            s["n"] = sp500_names.get(s["s"], "")
    return dict(
        asof=close["^GSPC"].dropna().index[-1].strftime("%Y-%m-%d"),
        hist=hist, stocks=stocks, idx=idx, sectors=sectors,
        spx=dict(close=[round(float(x)) for x in close["^GSPC"].dropna().tail(252)]),
        rules=[dict(k=k, n=n, y=y, r=r, side=s, u=u) for k, n, y, r, s, u in RULES],
        nuniv=len(stocks))
