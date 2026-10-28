"""Backtest the KCDT temperature score against later S&P 500 / QQQ moves.

python backtest/analyze.py  ->  prints tables, writes backtest/out/*.csv and results.json
Needs backtest/data/{close,high}.csv.gz from fetch_history.py.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from kcdt.compute import DOW30, RULES, points  # noqa: E402

SECTOR9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
START = "2004-01-01"
SPLIT = "2016-01-01"  # thresholds are fitted before this date and checked after it
H = 63  # forward window: about 3 months


def load():
    c = pd.read_csv(ROOT / "data/close.csv.gz", index_col=0, parse_dates=True)
    h = pd.read_csv(ROOT / "data/high.csv.gz", index_col=0, parse_dates=True)
    days = c["^GSPC"].dropna().index
    return c.loc[days].ffill(limit=3), h.loc[days].ffill(limit=3)


def indicators(close, high, basket):
    """Same 8 indicators as kcdt.compute.indicator_frame, but a stock only counts once it has
    enough history (current Dow members such as V listed after 2004)."""
    spx, spx_h = close["^GSPC"], high["^GSPC"]
    ma50, ma200 = spx.rolling(50).mean(), spx.rolling(200).mean()
    b, bh = close[basket], high[basket]
    m50, m200, h252 = b.rolling(50).mean(), b.rolling(200).mean(), bh.rolling(252).max()
    df = pd.DataFrame(index=close.index)
    df["vix5"] = close["^VIX"].rolling(5).mean()
    df["vs50"] = (spx / ma50 - 1) * 100
    df["pct50"] = (b > m50).where(m50.notna()).mean(axis=1) * 100
    df["dist52"] = (b / h252 - 1).mean(axis=1) * 100
    df["pct200"] = (b > m200).where(m200.notna()).mean(axis=1) * 100
    df["ma50_200"] = (ma50 / ma200 - 1) * 100
    df["rv21"] = np.log(spx).diff().rolling(21).std() * np.sqrt(252) * 100
    df["spx52"] = (spx / spx_h.rolling(252).max() - 1) * 100
    return df.loc[START:].dropna()


def score(ind, rules):
    pts = pd.DataFrame({k: ind[k].map(lambda v, y=y, r=r, s=s: points(v, y, r, s))
                        for k, _n, y, r, s, _u in rules})
    return pts.sum(axis=1), pts


def forward(px):
    """Forward return and worst drawdown from today's close over the next H days (percent)."""
    fwd = (px.shift(-H) / px - 1) * 100
    fmin = px[::-1].rolling(H, min_periods=H).min()[::-1].shift(-1)
    fdd = (fmin / px - 1).clip(upper=0) * 100
    f21 = (px.shift(-21) / px - 1) * 100
    return pd.DataFrame({"r21": f21, "r63": fwd, "dd63": fdd})


def calibrate(ind, rules, train_end=SPLIT, y_q=0.25, r_q=0.10):
    """Yellow = the worst 25% of days, red = the worst 10%, measured on the training years."""
    tr = ind.loc[:train_end]
    out = []
    for k, n, _y, _r, s, u in rules:
        q = (lambda p: tr[k].quantile(1 - p)) if s == "high" else (lambda p: tr[k].quantile(p))
        step = 5 if k in ("pct50", "pct200") else 1 if k in ("vix5", "rv21", "dist52", "spx52") else 0.5
        rnd = lambda v: round(round(v / step) * step, 1)
        out.append((k, n, rnd(q(y_q)), rnd(q(r_q)), s, u))
    return out


def zone(score, cuts):
    lo, hi = cuts
    return pd.cut(score, [-1, lo - 1, hi - 1, 99], labels=["健康", "警戒", "危險"])


def by_group(groups, fw):
    t = fw.groupby(groups, observed=True).agg(
        days=("r63", "size"), r21=("r21", "mean"), r63=("r63", "mean"),
        up63=("r63", lambda x: (x > 0).mean() * 100), dd63=("dd63", "mean"),
        dd10=("dd63", lambda x: (x <= -10).mean() * 100))
    t["share"] = t["days"] / t["days"].sum() * 100
    return t.round(1)


def strategy(px, zones, exit_zone="危險", back_zone="健康"):
    """Educational only: hold the index, go to cash the day after the zone reaches exit_zone,
    come back the day after it falls to back_zone. No costs, no interest on cash."""
    inn, state = [], True
    for z in zones:
        if state and z == exit_zone:
            state = False
        elif not state and z == back_zone:
            state = True
        inn.append(state)
    pos = pd.Series(inn, index=zones.index).shift(1).fillna(True)
    r = px.pct_change().reindex(zones.index).fillna(0)
    def stats(rets):
        eq = (1 + rets).cumprod()
        yrs = len(rets) / 252
        return dict(cagr=round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
                    maxdd=round((eq / eq.cummax() - 1).min() * 100, 1),
                    vol=round(rets.std() * np.sqrt(252) * 100, 1))
    s = stats(r * pos)
    s["in_market"] = round(pos.mean() * 100, 1)
    s["switches"] = int((pos.astype(int).diff().abs() == 1).sum())
    return s, stats(r)


def episodes(px, sc, cuts, min_dd=15):
    """Each peak-to-trough fall of min_dd% or more: when the score first hit 警戒 / 危險 and how much
    of the fall had already happened by then."""
    rows, peak_i, i = [], 0, 0
    p = px.dropna()
    s = sc.reindex(p.index)
    vals = p.values
    while i < len(p):
        if vals[i] >= vals[peak_i]:
            peak_i = i
        elif vals[i] / vals[peak_i] - 1 <= -min_dd / 100:
            j = i
            while j + 1 < len(p) and vals[j + 1] < vals[peak_i]:  # until a new high
                j += 1
            seg = p.iloc[peak_i:j + 1]
            tr_i = peak_i + int(np.argmin(seg.values))
            depth = vals[tr_i] / vals[peak_i] - 1
            row = dict(peak=p.index[peak_i].date().isoformat(), trough=p.index[tr_i].date().isoformat(),
                       depth=round(depth * 100, 1))
            win = s.iloc[max(0, peak_i - 21):tr_i + 1]
            for name, lvl in (("warn", cuts[0]), ("danger", cuts[1])):
                hit = win[win >= lvl]
                if len(hit):
                    d = hit.index[0]
                    row[name] = d.date().isoformat()
                    row[name + "_done"] = round((p[d] / vals[peak_i] - 1) / depth * 100)
                    row[name + "_days"] = int(p.index.get_loc(d) - peak_i)
            rows.append(row)
            peak_i = i = j + 1
            continue
        i += 1
    return pd.DataFrame(rows)


def main():
    close, high = load()
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    res = {}
    fw = {k: forward(close[k]).loc[START:] for k in ("^GSPC", "QQQ")}

    for basket_name, basket in (("dow30", [t for t in DOW30 if t in close]), ("sector9", SECTOR9)):
        ind = indicators(close, high, basket)
        ind.round(2).to_csv(out / f"indicators_{basket_name}.csv")
        new_rules = calibrate(ind, RULES)
        print(f"\n===== basket: {basket_name}  {ind.index[0].date()} .. {ind.index[-1].date()} =====")
        print("percentiles (whole period):")
        print(ind.quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).round(1).T.to_string())
        res[basket_name] = {"rules_new": new_rules, "start": str(ind.index[0].date()),
                            "end": str(ind.index[-1].date())}
        for label, rules in (("current", RULES), ("calibrated", new_rules)):
            sc, pts = score(ind, rules)
            print(f"\n--- {label} rules ---")
            for k, n, y, r, s, u in rules:
                print(f"  {n:<22} yellow {y:>6}  red {r:>6}  "
                      f"time yellow+ {(pts[k] >= 1).mean()*100:4.1f}%  red {(pts[k] == 2).mean()*100:4.1f}%")
            print("score distribution:", sc.value_counts(normalize=True).sort_index().mul(100).round(1).to_dict())
            r = res[basket_name][label] = {"score_dist": sc.value_counts().sort_index().to_dict()}
            for tk, f in fw.items():
                f = f.reindex(sc.index)
                bucket = pd.cut(sc, [-1, 0, 2, 4, 6, 8, 10, 12, 16],
                                labels=["0", "1-2", "3-4", "5-6", "7-8", "9-10", "11-12", "13-16"])
                fd = f.dropna()
                t = by_group(bucket.reindex(fd.index), fd)
                print(f"\n  {tk} forward by score bucket (r=return %, dd63=avg worst drop in next 63d, dd10=% of days with a 10%+ drop ahead):")
                print(t.to_string())
                r[f"buckets_{tk}"] = t.reset_index().to_dict("records")
                # per indicator light
                lights = {}
                for k, n, *_ in rules:
                    g = by_group(pts[k].reindex(f.dropna().index).map({0: "綠", 1: "黃", 2: "紅"}), f.dropna())
                    lights[k] = g.reset_index().to_dict("records")
                r[f"lights_{tk}"] = lights
            # pts per day for the site preview
            sc.to_frame("score").join(pts).to_csv(out / f"score_{basket_name}_{label}.csv")

    # zone cut search on the Dow basket with calibrated rules, train years only
    ind = indicators(close, high, [t for t in DOW30 if t in close])
    rules_new = res["dow30"]["rules_new"]
    print("\n===== zone cuts (dow30 basket) =====")
    zres = []
    for label, rules in (("current", RULES), ("calibrated", rules_new)):
        sc, _ = score(ind, rules)
        for lo in range(2, 9):
            for hi in range(lo + 2, 14):
                z = zone(sc, (lo, hi))
                row = dict(rules=label, warn=lo, danger=hi)
                for period, sl in (("train", slice(None, SPLIT)), ("test", slice(SPLIT, None))):
                    zz = z.loc[sl]
                    for tk in ("QQQ", "^GSPC"):
                        s, bh = strategy(close[tk], zz)
                        row[f"{period}_{tk}_cagr"] = s["cagr"]
                        row[f"{period}_{tk}_bh_cagr"] = bh["cagr"]
                        row[f"{period}_{tk}_maxdd"] = s["maxdd"]
                        row[f"{period}_{tk}_bh_maxdd"] = bh["maxdd"]
                        row[f"{period}_{tk}_in"] = s["in_market"]
                        row[f"{period}_{tk}_sw"] = s["switches"]
                    f = fw["QQQ"].loc[sl].dropna()
                    zf = z.reindex(f.index)
                    row[f"{period}_danger_share"] = round((zf == "危險").mean() * 100, 1)
                    row[f"{period}_healthy_dd63"] = round(f["dd63"][zf == "健康"].mean(), 1)
                    row[f"{period}_danger_dd63"] = round(f["dd63"][zf == "危險"].mean(), 1)
                zres.append(row)
    zdf = pd.DataFrame(zres)
    zdf.to_csv(out / "zone_search.csv", index=False)
    cols = ["rules", "warn", "danger", "train_QQQ_cagr", "train_QQQ_bh_cagr", "train_QQQ_maxdd", "train_QQQ_bh_maxdd",
            "train_QQQ_in", "test_QQQ_cagr", "test_QQQ_bh_cagr", "test_QQQ_maxdd", "test_QQQ_bh_maxdd", "test_QQQ_in",
            "test_QQQ_sw", "train_danger_share", "test_danger_share"]
    print(zdf.sort_values("train_QQQ_maxdd", ascending=False)[cols].head(25).to_string())
    print(zdf[(zdf.warn == 5) & (zdf.danger == 10)][cols].to_string())

    for label, rules, cuts in (("current", RULES, (5, 10)), ("calibrated", rules_new, (5, 10))):
        sc, _ = score(ind, rules)
        for tk in ("QQQ", "^GSPC"):
            e = episodes(close[tk].loc[START:], sc, cuts)
            print(f"\n--- {tk} falls of 15%+ ({label} rules, zones {cuts}) ---")
            print(e.to_string())
            res.setdefault("episodes", {})[f"{label}_{tk}"] = e.to_dict("records")

    # threshold variants: exposure 100% / 50% / 0% in 健康 / 警戒 / 危險 (zones 5 / 10)
    def mod(**kw):
        return [(k, n, *kw.get(k, (y, r)), s, u) for k, n, y, r, s, u in RULES]
    variants = {"A current": RULES, "B ma50_200 1/-3": mod(ma50_200=(1, -3)),
                "C B + spx52 -5/-15": mod(ma50_200=(1, -3), spx52=(-5, -15)),
                "D calibrated": [(k, n, -7, -15, sd, u) if k == "spx52" else (k, n, y, r, sd, u)
                                 for k, n, y, r, sd, u in rules_new]}  # spx52 red rounded to -15
    print("\n===== threshold variants, QQQ, exposure by zone =====")
    fq = fw["QQQ"]
    for name, rules in variants.items():
        sc, _ = score(ind, rules)
        fd = fq.reindex(sc.index).dropna()
        s2 = sc.reindex(fd.index)
        line = f"{name:20}"
        for lab, sl in (("all", slice(None, None)), ("test", slice(SPLIT, None))):
            s = sc.loc[sl]
            pos = pd.Series(np.where(s >= 10, 0, np.where(s >= 5, 0.5, 1.0)), index=s.index).shift(1).fillna(1)
            eq = (1 + close["QQQ"].pct_change().reindex(pos.index).fillna(0) * pos).cumprod()
            line += f" | {lab} cagr {(eq.iloc[-1] ** (252 / len(eq)) - 1) * 100:5.1f} maxdd {(eq / eq.cummax() - 1).min() * 100:6.1f}"
        line += (f" | >=10 dd10 {(fd.dd63[s2 >= 10] <= -10).mean() * 100:4.1f}"
                 f" | 0-4 dd10 {(fd.dd63[s2 < 5] <= -10).mean() * 100:4.1f}")
        print(line)

    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=str, indent=1))


if __name__ == "__main__":
    main()
