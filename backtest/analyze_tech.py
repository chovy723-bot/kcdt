"""Tech temperature (Nasdaq-100 version) and the "panic peak" hint.

python backtest/analyze_tech.py  ->  prints tables, writes backtest/out/report_tech.txt via shell redirect
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from analyze import SPLIT, START, by_group, episodes, forward, load, score  # noqa: E402
from kcdt.compute import DOW30, RULES  # noqa: E402

NDX_MEMBERS = (ROOT / "data/ndx_members.txt").read_text().split()


def tech_indicators(close, high, basket, index="^NDX", vol="^VXN"):
    """The same 8 indicators, with the Nasdaq-100 index, VXN and a Nasdaq-100 basket."""
    ix, ix_h = close[index], high[index]
    ma50, ma200 = ix.rolling(50).mean(), ix.rolling(200).mean()
    b, bh = close[basket], high[basket]
    m50, m200, h252 = b.rolling(50).mean(), b.rolling(200).mean(), bh.rolling(252).max()
    df = pd.DataFrame(index=close.index)
    df["vix5"] = close[vol].rolling(5).mean()
    df["vs50"] = (ix / ma50 - 1) * 100
    df["pct50"] = (b > m50).where(m50.notna()).mean(axis=1) * 100
    df["dist52"] = (b / h252 - 1).where(h252.notna() & (bh.notna().rolling(252).sum() >= 252)).mean(axis=1) * 100
    df["pct200"] = (b > m200).where(m200.notna()).mean(axis=1) * 100
    df["ma50_200"] = (ma50 / ma200 - 1) * 100
    df["rv21"] = np.log(ix).diff().rolling(21).std() * np.sqrt(252) * 100
    df["spx52"] = (ix / ix_h.rolling(252).max() - 1) * 100
    return df.loc[START:].dropna()


def same_percentile_rules(ind_tech, ind_spx, rules):
    """Tech thresholds at the same historical percentile as the S&P thresholds sit at (2004–2015),
    so a tech yellow / red is as rare as an S&P one."""
    out = []
    for k, n, y, r, s, u in rules:
        a, t = ind_spx[k].loc[:SPLIT], ind_tech[k].loc[:SPLIT]
        q = lambda v: (a >= v).mean() if s == "high" else (a <= v).mean()
        val = lambda p: t.quantile(1 - p) if s == "high" else t.quantile(p)
        step = 5 if k in ("pct50", "pct200") else 1
        rnd = lambda v: round(round(v / step) * step, 1)
        out.append((k, n, rnd(val(q(y))), rnd(val(q(r))), s, u))
    return out


def panic_signals(ind, rules, cool=21):
    """Days when VIX-5d and 21d volatility are BOTH red; plus the first such day after `cool` quiet days."""
    rr = {k: (y, r) for k, _n, y, r, _s, _u in rules}
    both = (ind["vix5"] >= rr["vix5"][1]) & (ind["rv21"] >= rr["rv21"][1])
    first = both & ~both.shift(1, fill_value=False).rolling(cool, min_periods=1).max().astype(bool)
    return both, first


def panic_peak(ind, rules, drop=0.15, within=10):
    """Panic easing: VIX-5d has come down `drop` from its 10-day high while both were red within `within` days."""
    both, _ = panic_signals(ind, rules)
    recent = both.rolling(within, min_periods=1).max().astype(bool)
    off = ind["vix5"] <= ind["vix5"].rolling(10).max() * (1 - drop)
    sig = recent & off
    first = sig & ~sig.shift(1, fill_value=False).rolling(21, min_periods=1).max().astype(bool)
    return sig, first


def fwd_table(sig, fw, label):
    f = fw.reindex(sig.index)
    g = f[sig].dropna()
    base = f.dropna()
    rows = {label: g, "全部日子": base}
    t = pd.DataFrame({k: dict(days=len(v), r21=v.r21.mean(), r63=v.r63.mean(), up63=(v.r63 > 0).mean() * 100,
                              dd63=v.dd63.mean(), dd10=(v.dd63 <= -10).mean() * 100) for k, v in rows.items()}).T
    return t.round(1)


def main():
    close, high = load()
    ndx = [t for t in NDX_MEMBERS if t in close]
    print("Nasdaq-100 members with data:", len(ndx), "of", len(NDX_MEMBERS))
    print("members with history by year:", {y: int(close.loc[f"{y}-06-30":f"{y}-07-05", ndx].notna().any().sum())
                                            for y in (2004, 2008, 2012, 2016, 2020, 2024)})
    from analyze import indicators
    ind_spx = indicators(close, high, [t for t in DOW30 if t in close])
    fwq = forward(close["QQQ"]).loc[START:]

    for bname, basket in (("ndx_all", ndx), ("dow30", [t for t in DOW30 if t in close])):
        ind = tech_indicators(close, high, basket)
        print(f"\n===== tech temperature, basket {bname}: {ind.index[0].date()} .. {ind.index[-1].date()} =====")
        print(ind.quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).round(1).T.to_string())
        rules_t = same_percentile_rules(ind, ind_spx, RULES)
        for label, rules in (("S&P thresholds as-is", RULES), ("tech thresholds (same rarity)", rules_t)):
            sc, pts = score(ind, rules)
            print(f"\n--- {label} ---")
            for k, n, y, r, s, u in rules:
                print(f"  {k:<9} yellow {y:>6} red {r:>6}  yellow+ {(pts[k] >= 1).mean()*100:4.1f}%  red {(pts[k] == 2).mean()*100:4.1f}%")
            fd = fwq.reindex(sc.index).dropna()
            bucket = pd.cut(sc, [-1, 4, 9, 16], labels=["0-4", "5-9", "10-16"]).reindex(fd.index)
            print("  QQQ forward by zone:\n" + by_group(bucket, fd).to_string())
            for per, sl in (("all", slice(None, None)), ("test", slice(SPLIT, None))):
                s = sc.loc[sl]
                pos = pd.Series(np.where(s >= 10, 0, np.where(s >= 5, 0.5, 1.0)), index=s.index).shift(1).fillna(1)
                eq = (1 + close["QQQ"].pct_change().reindex(pos.index).fillna(0) * pos).cumprod()
                print(f"  exposure 100/50/0 {per}: cagr {(eq.iloc[-1] ** (252 / len(eq)) - 1) * 100:.1f} "
                      f"maxdd {(eq / eq.cummax() - 1).min() * 100:.1f}")
            e = episodes(close["QQQ"].loc[START:], sc, (5, 10))
            print("  QQQ falls of 15%+:\n" + e.to_string())
            # S&P score vs tech score on the same days
            sc_spx, _ = score(ind_spx, RULES)
            j = pd.concat([sc_spx.rename("spx"), sc.rename("tech")], axis=1).dropna()
            print(f"  corr with S&P score {j.corr().iloc[0, 1]:.2f}; tech-S&P >= 4 on {((j.tech - j.spx) >= 4).mean()*100:.1f}% of days")

    print("\n===== panic peak hint =====")
    for name, ind, rules in (("S&P (VIX, S&P vol)", ind_spx, RULES),):
        both, first = panic_signals(ind, rules)
        print(f"\n{name}: both red on {both.mean()*100:.1f}% of days, {int(first.sum())} episodes")
        print(fwd_table(both, fwq, "雙紅燈日子").to_string())
        print(fwd_table(first, fwq, "雙紅燈第一日").to_string())
        for d in (0.10, 0.15, 0.20):
            sig, f1 = panic_peak(ind, rules, drop=d)
            print(f"\n  panic easing (VIX-5d {int(d*100)}% off 10d high, both red within 10d): {int(f1.sum())} signals")
            print(fwd_table(f1, fwq, "見頂訊號第一日").to_string())
        sig, f1 = panic_peak(ind, rules, drop=0.15)
        lst = fwq.reindex(f1[f1].index).join(close["QQQ"].rename("px"))
        lst["score"] = score(ind, rules)[0].reindex(lst.index)
        print("\n  each signal (15% version):\n" + lst.round(1).to_string())
        b1 = fwq.reindex(first[first].index)
        b1["score"] = score(ind, rules)[0].reindex(b1.index)
        print("\n  each first double-red day:\n" + b1.round(1).to_string())


if __name__ == "__main__":
    main()
