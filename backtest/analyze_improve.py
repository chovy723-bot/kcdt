"""Can other indicators warn earlier (2022 slow bear) or bring you back sooner (2020 / 2025 V-shapes)?

Candidates use fixed, textbook parameters (no tuning), so the 2016+ check stays honest.
python backtest/analyze_improve.py > backtest/out/report_improve.txt
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from analyze import SPLIT, START, forward, indicators, load, score  # noqa: E402
from kcdt.compute import DOW30, RULES  # noqa: E402
from kcdt.tech import TECH_RULES, tech_indicator_frame  # noqa: E402

NDX = (ROOT / "data/ndx_members.txt").read_text().split()


def candidates(c):
    """Warning flags (True = risk-off) and recovery flags (True = panic receding)."""
    q = c["QQQ"]
    nd = c[[t for t in NDX if t in c]]
    f = pd.DataFrame(index=c.index)
    f["qqq_below_200"] = q < q.rolling(200).mean()
    f["qqq_50_falling"] = q.rolling(50).mean().diff(20) < 0
    credit = c["HYG"] / c["IEF"]
    f["credit_below_100"] = (credit < credit.rolling(100).mean()).where(credit.rolling(100).count() == 100)
    term = c["^VIX"] / c["^VIX3M"]
    f["vix_backwardation"] = (term >= 1).where(term.notna())
    lows = (nd <= nd.rolling(252).min()).where(nd.rolling(252).count() == 252)
    f["new_lows_10pct"] = lows.mean(axis=1) * 100 >= 10
    small = c["IWM"] / c["SPY"]
    f["smallcap_below_200"] = small < small.rolling(200).mean()
    semi = c["SMH"] / c["QQQ"]
    f["semis_below_100"] = semi < semi.rolling(100).mean()
    f["curve_inverted"] = (c["^TNX"] - c["^IRX"]) < 0
    # recovery
    m20 = q.rolling(20).mean()
    f["R_above_20_rising"] = (q > m20) & (m20.diff(5) > 0)
    f["R_term_normal"] = (term < 0.9).where(term.notna())
    above20 = (nd > nd.rolling(20).mean()).where(nd.rolling(20).count() == 20).mean(axis=1) * 100
    f["R_breadth_thrust"] = (above20 >= 75) & (above20.rolling(15).min() <= 25)
    f["R_thrust_recent"] = f["R_breadth_thrust"].rolling(21, min_periods=1).max().astype(bool)
    return f.loc[START:], above20


def falls(px, min_dd=15):
    p, out, peak, i = px.dropna(), [], 0, 0
    v = p.values
    while i < len(p):
        if v[i] >= v[peak]:
            peak = i
        elif v[i] / v[peak] - 1 <= -min_dd / 100:
            j = i
            while j + 1 < len(p) and v[j + 1] < v[peak]:
                j += 1
            tr = peak + int(np.argmin(v[peak:j + 1]))
            out.append((p.index[peak], p.index[tr], v[tr] / v[peak] - 1))
            peak = i = j + 1
            continue
        i += 1
    return out


def lead(flag, px, eps):
    """For each fall: share of the fall already done on the first flagged day (from 63 days before the peak)."""
    res = []
    for pk, tr, depth in eps:
        w = flag.loc[px.index[max(0, px.index.get_loc(pk) - 63)]:tr]
        w = w[w.astype(bool)]
        if w.empty:
            res.append(None)
            continue
        d = w.index[0]
        res.append(max(0, round((px[d] / px[pk] - 1) / depth * 100)) if d > pk else 0)
    return res


def run(c, expo, label, out):
    r = c["QQQ"].pct_change()
    for per, sl in (("all", slice(START, None)), ("train", slice(START, SPLIT)), ("test", slice(SPLIT, None))):
        e = expo.loc[sl].shift(1).fillna(1)
        x = r.reindex(e.index).fillna(0) * e
        eq = (1 + x).cumprod()
        out.setdefault(label, {})[per] = (
            f"cagr {(eq.iloc[-1] ** (252 / len(eq)) - 1) * 100:5.1f} maxdd {(eq / eq.cummax() - 1).min() * 100:6.1f} "
            f"in {e.mean() * 100:3.0f}%")
    return out


def main():
    c, h = load()
    f, above20 = candidates(c)
    q = c["QQQ"].loc[START:]
    eps = falls(q)
    fw = forward(c["QQQ"]).loc[START:]
    names = [k for k in f if not k.startswith("R_")]
    print("QQQ falls of 15%+:", [(a.date().isoformat(), round(d * 100)) for a, _b, d in eps])

    sc, _ = score(indicators(c, h, [t for t in DOW30 if t in c]), RULES)
    tind = tech_indicator_frame(c, h, NDX).loc[START:].dropna()
    tsc, _ = score(tind, TECH_RULES)
    base_flags = {"score>=5": sc >= 5, "score>=10": sc >= 10, "tech>=5": tsc >= 5, "tech>=10": tsc >= 10}

    print("\n=== warning flags: % of fall done when first flagged (0 = before the peak), and lift ===")
    rows = []
    for k, flag in list(base_flags.items()) + [(k, f[k]) for k in names]:
        flag = flag.reindex(q.index).fillna(False).astype(bool)
        fd = fw.reindex(flag.index).dropna()
        fl = flag.reindex(fd.index)
        rows.append(dict(flag=k, on=round(flag.mean() * 100, 1),
                         dd10_on=round((fd.dd63[fl] <= -10).mean() * 100, 1),
                         dd10_off=round((fd.dd63[~fl] <= -10).mean() * 100, 1),
                         r63_on=round(fd.r63[fl].mean(), 1),
                         **{a.strftime("%Y-%m"): v for (a, _b, _d), v in zip(eps, lead(flag, q, eps))}))
    print(pd.DataFrame(rows).set_index("flag").to_string())

    print("\n=== recovery flags: days after the QQQ low until the flag fires, and gain missed ===")
    for k in ("R_above_20_rising", "R_term_normal", "R_thrust_recent"):
        line = []
        for pk, tr, d in eps:
            w = f[k].loc[tr:]
            w = w[w.astype(bool)]
            if w.empty:
                line.append("-")
                continue
            line.append(f"{tr.strftime('%Y-%m')}: {q.index.get_loc(w.index[0]) - q.index.get_loc(tr)}d "
                        f"+{(q[w.index[0]] / q[tr] - 1) * 100:.0f}%")
        print(f"{k:20}", " | ".join(line))
    # how long the score takes to fall back below 5 after the low
    line = []
    for pk, tr, d in eps:
        w = sc.loc[tr:]
        w = w[w < 5]
        line.append(f"{tr.strftime('%Y-%m')}: {q.index.get_loc(w.index[0]) - q.index.get_loc(tr)}d "
                    f"+{(q[w.index[0]] / q[tr] - 1) * 100:.0f}%" if len(w) else "-")
    print(f"{'score back < 5':20}", " | ".join(line))

    print("\n=== exposure rules on QQQ (no costs, cash earns 0) ===")
    out = {}
    one = pd.Series(1.0, index=q.index)
    zone = lambda s: pd.Series(np.where(s >= 10, 0, np.where(s >= 5, 0.5, 1.0)), index=s.index)
    t200 = (~f["qqq_below_200"]).astype(float)
    rec = ((f["R_above_20_rising"] & f["R_term_normal"].fillna(True).astype(bool)) | f["R_thrust_recent"]).astype(bool)
    credit = f["credit_below_100"].fillna(False).astype(bool)
    back = f["vix_backwardation"].fillna(False).astype(bool)
    strategies = {
        "buy & hold": one,
        "A score zones 100/50/0": zone(sc),
        "B tech score zones": zone(tsc.reindex(q.index).ffill()),
        "C QQQ > 200-day line": t200,
        "D A + fast return (recovery = 100%)": zone(sc).where(~rec, 1.0),
        "E B + fast return": zone(tsc.reindex(q.index).ffill()).where(~rec, 1.0),
        "F B, warn 50% extra if QQQ<200d or credit weak": pd.concat([
            zone(tsc.reindex(q.index).ffill()),
            pd.Series(np.where(f["qqq_below_200"].astype(bool) | credit, 0.5, 1.0), index=q.index)], axis=1).min(axis=1),
        "G F + fast return": pd.concat([
            zone(tsc.reindex(q.index).ffill()),
            pd.Series(np.where(f["qqq_below_200"].astype(bool) | credit, 0.5, 1.0), index=q.index)], axis=1).min(axis=1).where(~rec, 1.0),
        "H 200d line + fast return": t200.where(~rec, 1.0),
        "I B, 0% in VIX backwardation": zone(tsc.reindex(q.index).ffill()).where(~back, 0.0),
        "J A + rebound floor 50% (QQQ above rising 20d line)": pd.concat(
            [zone(sc), f["R_above_20_rising"].astype(float) * 0.5], axis=1).max(axis=1),
        "K B + rebound floor 50%": pd.concat(
            [zone(tsc.reindex(q.index).ffill()), f["R_above_20_rising"].astype(float) * 0.5], axis=1).max(axis=1),
    }
    for k, e in strategies.items():
        run(c, e.reindex(q.index).fillna(1.0), k, out)
    print(pd.DataFrame(out).T.to_string())

    print("\n=== swapping one of the 8 indicators (zones 100/50/0, then + rebound floor) ===")
    from kcdt.compute import points
    ind = indicators(c, h, [t for t in DOW30 if t in c])
    cr = c["HYG"] / c["IEF"]
    ind["credit"] = ((cr / cr.rolling(100).mean() - 1) * 100).reindex(ind.index)
    nd = c[[t for t in NDX if t in c]]
    ind["lows"] = ((nd <= nd.rolling(252).min()).where(nd.rolling(252).count() == 252).mean(axis=1) * 100).reindex(ind.index)

    def score_of(rules, extra=()):
        s = pd.Series(0, index=ind.index)
        for k, y, r, sd in [(k, y, r, sd) for k, _n, y, r, sd, _u in rules] + list(extra):
            s = s + ind[k].map(lambda v, y=y, r=r, sd=sd: 0 if pd.isna(v) else points(v, y, r, sd))
        return s.reindex(q.index).ffill()

    drop = lambda *ks: [r for r in RULES if r[0] not in ks]
    swaps = {"current 8": score_of(RULES),
             "credit replaces rv21": score_of(drop("rv21"), [("credit", -1, -3, "low")]),
             "credit replaces vs50": score_of(drop("vs50"), [("credit", -1, -3, "low")]),
             "new lows replaces rv21": score_of(drop("rv21"), [("lows", 3, 10, "high")]),
             "credit + lows replace rv21, vs50": score_of(drop("rv21", "vs50"),
                                                          [("credit", -1, -3, "low"), ("lows", 3, 10, "high")])}
    out = {}
    floor = f["R_above_20_rising"].astype(float) * 0.5
    for k, s in swaps.items():
        run(c, zone(s), k, out)
        run(c, pd.concat([zone(s), floor], axis=1).max(axis=1), k + " + floor", out)
    print(pd.DataFrame(out).T.to_string())


if __name__ == "__main__":
    main()
