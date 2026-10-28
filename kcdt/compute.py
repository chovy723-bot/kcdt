"""Turn daily price history (and cached fundamentals) into the dashboard's data object.

Price inputs are a dict of wide DataFrames {"Open","High","Low","Close","Volume"}
(index = trading dates, columns = tickers), split/dividend-adjusted. Nothing here
touches the network, so it can be tested with any price history.
"""
import math

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

MINIS = [("^GSPC", "S&P 500"), ("^IXIC", "NASDAQ"), ("^DJI", "道指"), ("IWM", "羅素 2000")]

# GICS (Wikipedia) -> Yahoo/finviz sector names, and Chinese labels
GICS_TO_SECTOR = {
    "Information Technology": "Technology", "Communication Services": "Communication Services",
    "Consumer Discretionary": "Consumer Cyclical", "Consumer Staples": "Consumer Defensive",
    "Materials": "Basic Materials", "Financials": "Financial Services", "Health Care": "Healthcare",
    "Real Estate": "Real Estate", "Industrials": "Industrials", "Energy": "Energy", "Utilities": "Utilities",
}
SECTOR_ZH = {
    "Technology": "科技", "Communication Services": "通訊", "Consumer Cyclical": "非必需消費",
    "Consumer Defensive": "必需消費", "Basic Materials": "原材料", "Financial Services": "金融",
    "Healthcare": "醫療", "Real Estate": "房地產", "Industrials": "工業", "Energy": "能源",
    "Utilities": "公用",
}

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
    """All 8 temperature indicators as daily series (percent units)."""
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


def _f(x, d=2):
    """float rounded for JSON, or None for missing."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, d)


def _growth(new, old):
    if new is None or old is None or old == 0:
        return None
    return (new / abs(old) - 1) * 100


def fundamentals_row(f):
    """Derived fundamental fields from one cached Yahoo record."""
    if not f:
        return {}
    revq, epsq, reva = f.get("revq") or [], f.get("epsq") or [], f.get("reva") or []
    # sequential (quarter over previous quarter) growth for the latest 3 quarters
    rev_seq = [_growth(revq[i], revq[i + 1]) for i in range(min(3, len(revq) - 1))]
    eps_seq = [_growth(epsq[i], epsq[i + 1]) for i in range(min(3, len(epsq) - 1))]
    up3 = lambda xs: len(xs) == 3 and all(x is not None and x > 0 for x in xs)
    # year-on-year growth of each of the latest 3 quarters; the weakest of the 3 is what a filter needs
    yoy3 = lambda xs: (min(g) if len(xs) >= 7 and None not in (g := [_growth(xs[i], xs[i + 4]) for i in range(3)])
                       else None)
    # Nasdaq's quarterly history goes back ~3 fiscal years; Yahoo's is the fallback
    nrev, neps, epsh = f.get("nrev") or [], f.get("neps") or [], f.get("epsh") or []
    g3r = yoy3(nrev) if len(nrev) >= 7 else yoy3(revq)
    g3e = next((g for g in (yoy3(neps), yoy3(epsh), yoy3(epsq)) if g is not None), None)
    rising = lambda xs: len(xs) >= 3 and None not in xs[:3] and xs[0] > xs[1] > xs[2]
    efc, rfc = f.get("efc") or [None, None], f.get("rfc") or [None, None]
    pct = lambda x: None if x is None else x * 100
    ipo_years = None
    if f.get("ipo"):
        ipo_years = (pd.Timestamp.now().timestamp() - f["ipo"] / 1000) / (365.25 * 86400)
    edate = None
    if f.get("edate"):
        edate = pd.Timestamp(f["edate"], unit="s").strftime("%Y-%m-%d")
    return dict(
        mcap=_f((f.get("mcap") or 0) / 1e9, 2) if f.get("mcap") else None,  # US$ billions
        beta=_f(f.get("beta")), inst=_f((f["inst"] * 100) if f.get("inst") is not None else None, 1),
        sq=_f((f["sq"] * 100) if f.get("sq") is not None else None, 1),
        eq=_f((f["eq"] * 100) if f.get("eq") is not None else None, 1),
        sa=_f(_growth(reva[0], reva[1]) if len(reva) >= 2 else None, 1),
        pe=_f(f.get("pe"), 1), fpe=_f(f.get("fpe"), 1),
        rev3=up3(rev_seq), eps3=up3(eps_seq),
        g3r=_f(g3r, 1), g3e=_f(g3e, 1),
        yr=rising(reva) and rising(f.get("epsa") or []) if f.get("epsa") else None,
        e0=_f(pct(efc[0]), 1), e1=_f(pct(efc[1]), 1), r0=_f(pct(rfc[0]), 1), r1=_f(pct(rfc[1]), 1),
        ipo=_f(ipo_years, 1), edate=edate, ind=f.get("ind"),
    )


def build(px, meta, fund=None, fng=None, earn=None):
    """px: dict of OHLCV DataFrames. meta: DataFrame indexed by ticker with n, gics, ndx, sp500...
    fund: {ticker: cached Yahoo record}."""
    fund = fund or {}
    # Keep only completed S&P sessions; Yahoo can add a partial row for today
    # (e.g. VIX trades before the cash open), which would leave every stock NaN.
    P = _clean(px)
    days = P["Close"].index
    close, high, low, vol = P["Close"], P["High"], P["Low"], P["Volume"]

    # ---- temperature (unchanged method) ----
    ind = indicator_frame(close, high, close["^VIX"]).dropna().tail(HIST_DAYS)

    # ---- per-stock technicals over the whole universe ----
    tick = [t for t in meta.index if t in close and close[t].dropna().shape[0] >= 253]
    c, h, l, v = close[tick], high[tick], low[tick], vol[tick]
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    hi252, lo252 = h.rolling(252).max(), l.rolling(252).min()
    last, prev = c.iloc[-1], c.iloc[-2]
    chg = (last / prev - 1) * 100
    ret = {k: (last / c.iloc[-1 - k] - 1) * 100 for k in (5, 15, 21, 63, 126, 252)}
    rs = 0.4 * ret[63] + 0.2 * ret[126] + 0.2 * ret[252] + 0.2 * ret[21]
    rs_rank = (rs.rank(pct=True) * 98 + 1).round()
    avgv50 = v.rolling(50).mean().iloc[-1]
    rvol = v.iloc[-1] / avgv50
    new_high = h.iloc[-1] >= hi252.iloc[-2]
    new_low = l.iloc[-1] <= lo252.iloc[-2]

    # MVP ("ants"): last 15 sessions — 10+ up days, volume +20% vs the 15 before, price +20%
    up = (c.diff() > 0).tail(15).sum()
    v15, v15p = v.tail(15).mean(), v.iloc[-30:-15].mean()
    mvp_m = up >= 10
    mvp_v = (v15 / v15p - 1) >= 0.20
    mvp_p = ret[15] >= 20
    mvp = mvp_m.astype(int) + mvp_v.astype(int) + mvp_p.astype(int)

    stocks = []
    for t in tick:
        lc, m50, m200 = last[t], ma50[t].iloc[-1], ma200[t].iloc[-1]
        if pd.isna(lc) or pd.isna(m200):
            continue
        m200b = ma200[t].iloc[-21]
        tt = [lc > m50, m50 > m200, lc > m200, m200 > m200b, lc >= hi252[t].iloc[-1] * 0.75]
        gics = meta.at[t, "gics"]
        sec = fund.get(t, {}).get("ysec") or GICS_TO_SECTOR.get(gics, gics or "")
        row = dict(
            s=t, n=meta.at[t, "n"], sec=sec, ndx=bool(meta.at[t, "ndx"]),
            c=_f(lc), chg=_f(chg[t]), r5=_f(ret[5][t], 1), r21=_f(ret[21][t], 1), r63=_f(ret[63][t], 1),
            r126=_f(ret[126][t], 1), r252=_f(ret[252][t], 1),
            d50=_f((lc / m50 - 1) * 100, 1), d200=_f((lc / m200 - 1) * 100, 1),
            x50=bool(m50 > m200), dh=_f((lc / hi252[t].iloc[-1] - 1) * 100, 1),
            av=int(avgv50[t]) if not pd.isna(avgv50[t]) else 0, vol=int(v[t].iloc[-1]) if not pd.isna(v[t].iloc[-1]) else 0,
            rvol=_f(rvol[t]), nh=bool(new_high[t]), nl=bool(new_low[t]),
            tt=int(sum(bool(x) for x in tt)), rsr=int(rs_rank[t]),
            mvp=int(mvp[t]), mvpm=int(up[t]), mvpv=_f((v15[t] / v15p[t] - 1) * 100, 0), mvpp=_f(ret[15][t], 1),
        )
        row.update(fundamentals_row(fund.get(t)))
        if earn and t in earn:  # Nasdaq's calendar is more current than Yahoo and says before/after the market
            row["edate"], row["etime"] = earn[t]
        stocks.append(row)

    # ---- breadth: all stocks and Nasdaq-100 only ----
    def breadth(rows):
        n = len(rows) or 1
        cnt = lambda k: sum(1 for r in rows if k(r))
        return dict(
            n=len(rows), adv=cnt(lambda r: (r["chg"] or 0) > 0), dec=cnt(lambda r: (r["chg"] or 0) < 0),
            nh=cnt(lambda r: r["nh"]), nl=cnt(lambda r: r["nl"]),
            a50=cnt(lambda r: r["d50"] is not None and r["d50"] > 0),
            a200=cnt(lambda r: r["d200"] is not None and r["d200"] > 0))

    above50 = (c > ma50).where(ma50.notna()).mean(axis=1) * 100
    above200 = (c > ma200).where(ma200.notna()).mean(axis=1) * 100
    nd = [t for t in tick if meta.at[t, "ndx"]]
    nd50 = (c[nd] > ma50[nd]).where(ma50[nd].notna()).mean(axis=1) * 100
    nd200 = (c[nd] > ma200[nd]).where(ma200[nd].notna()).mean(axis=1) * 100

    hist = []
    for d, row in ind.iterrows():
        val = {k: round(float(row[k]), 2) for k, *_ in RULES}
        pts = {k: points(val[k], y, r, s) for k, _n, y, r, s, _u in RULES}
        hist.append(dict(d=d.strftime("%m-%d"), score=sum(pts.values()), v=val, p=pts,
                         b50=_f(above50[d], 1), b200=_f(above200[d], 1),
                         n50=_f(nd50[d], 1), n200=_f(nd200[d], 1)))

    def pct(t, k):
        s = close[t].dropna()
        return _f((s.iloc[-1] / s.iloc[-1 - k] - 1) * 100)

    def ytd(t):
        s = close[t].dropna()
        base = s[s.index.year < s.index[-1].year]
        return _f((s.iloc[-1] / base.iloc[-1] - 1) * 100) if len(base) else None

    sectors = [dict(t=t, n=n, d1=pct(t, 1), d5=pct(t, 5), d10=pct(t, 10), h6=pct(t, 126), ytd=ytd(t))
               for t, n in SECTORS if t in close]
    idx = [dict(n=n, l=_f(close[t].dropna().iloc[-1]), p=_f(close[t].dropna().iloc[-2]))
           for t, n in INDICES if t in close]
    minis = []
    for t, n in MINIS:
        if t not in close:
            continue
        o = P["Open"][t].tail(60)
        minis.append(dict(n=n, d=[x.strftime("%m-%d") for x in o.index],
                          o=[_f(x) for x in o], h=[_f(x) for x in high[t].tail(60)],
                          l=[_f(x) for x in low[t].tail(60)], c=[_f(x) for x in close[t].tail(60)]))

    return dict(
        asof=days[-1].strftime("%Y-%m-%d"),
        hist=hist, tech=tech(close, high, meta), panic=_panic(indicator_frame(close, high, close["^VIX"]).dropna()),
        stocks=stocks, idx=idx, sectors=sectors, minis=minis,
        breadth=dict(all=breadth(stocks), ndx=breadth([s for s in stocks if s["ndx"]])),
        spx=dict(close=[round(float(x)) for x in close["^GSPC"].dropna().tail(252)]),
        rules=[dict(k=k, n=n, y=y, r=r, side=s, u=u) for k, n, y, r, s, u in RULES],
        sector_zh=SECTOR_ZH, nuniv=len(stocks), fng=fng,
        nfund=sum(1 for s in stocks if s.get("mcap") is not None))


def _panic(ind):
    from .tech import panic
    return panic(ind)


def tech(close, high, meta):
    """Nasdaq-100 temperature history in the same shape as the S&P one, or None without ^NDX/^VXN."""
    from .tech import TECH_RULES, tech_indicator_frame, tech_score
    if "^NDX" not in close or "^VXN" not in close:
        return None
    ind = tech_indicator_frame(close, high, [t for t in meta.index if meta.at[t, "ndx"]]).dropna().tail(HIST_DAYS)
    hist = []
    for d, row in ind.iterrows():
        score, pts = tech_score(row)
        hist.append(dict(d=d.strftime("%m-%d"), score=score, v={k: round(float(row[k]), 2) for k, *_ in TECH_RULES}, p=pts))
    return dict(hist=hist, rules=[dict(k=k, n=n, y=y, r=r, side=sd, u=u) for k, n, y, r, sd, u in TECH_RULES]) if hist else None


def wants_fundamentals(px, meta):
    """Tickers worth a fundamentals lookup: all Nasdaq-100 plus anything near KC's technical filter."""
    days = px["Close"]["^GSPC"].dropna().index
    c = px["Close"].loc[days]
    v = px["Volume"].loc[days]
    tick = [t for t in meta.index if t in c]
    c, v = c[tick], v[tick]
    last = c.iloc[-1]
    ma50, ma200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    ok = (last > 20) & (last > ma50 * 0.97) & (ma50 > ma200 * 0.97) & (v.rolling(50).mean().iloc[-1] > 50_000)
    picked = set(ok[ok].index) | {t for t in tick if meta.at[t, "ndx"]}
    return sorted(picked)


def _clean(px):
    days = px["Close"]["^GSPC"].dropna().index
    return {k: v.sort_index().loc[days].ffill(limit=3) for k, v in px.items()}


def stock_pages(px, stocks, days=260):
    """Per-stock chart data for the stock page: last `days` sessions of OHLCV plus 50/200-day lines."""
    P = _clean(px)
    close = P["Close"]
    ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
    idx = close.index[-days:]
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    out = {}
    for row in stocks:
        t = row["s"]
        col = lambda df, d=2: [_f(x, d) for x in df[t].loc[idx]]
        out[t] = dict(row, d=dates, o=col(P["Open"]), h=col(P["High"]), l=col(P["Low"]), cl=col(close),
                      v=[None if pd.isna(x) else int(x) for x in P["Volume"][t].loc[idx]],
                      m50=col(ma50), m200=col(ma200))
    return out
