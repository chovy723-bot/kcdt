"""Economic calendar labels: which period a release covers, and whether a % is m/m, y/y or q/q.

Nasdaq's calendar gives neither, so:
- m/m, y/y and q/q come from ForexFactory's weekly calendar, whose titles say it ("Core PCE Price Index m/m"),
  matched to Nasdaq's rows by release time and forecast. ForexFactory only publishes the current week; for
  later weeks a pair of same-name % rows falls back to "smaller = m/m" and is flagged as a guess.
- The reference period follows each release's fixed schedule (CPI in November covers October, GDP covers
  the last finished quarter, jobless claims cover the week to last Saturday).
"""
import re

import pandas as pd
import requests

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
PER = {"m/m": "按月", "y/y": "按年", "q/q": "按季（年率）"}

# (name pattern, rule): months back from the release month (or a function of the release date), "q" = last finished quarter,
# "w-sat"/"w-fri" = week ending the previous Saturday/Friday. Patterns are tried in order.
PERIOD_RULES = [
    (r"Jobless Claims|Unemployment Claims", "w-sat"),
    (r"Crude|Gasoline|Distillate|Heating Oil|Cushing|Refinery|EIA|Natural Gas Storage", "w-fri"),
    (r"^(Advance |Prelim |Final )?GDP|Employment Cost Index|Productivity|Unit Labor Costs|Current Account", "q"),
    (r"JOLT|House Price|HPI|Case-Shiller|S&P/CS|Factory Orders|Business Inventories|Construction Spending|"
     r"Consumer Credit|^Trade Balance", 2),
    (r"Wholesale Inventories", lambda d: 2 if d.day <= 20 else 1),  # final early in the month, advance at the end
    (r"CB Consumer Confidence|Michigan|Chicago PMI|Richmond|Philadelphia|Philly|Empire State|NY Empire|"
     r"Dallas Fed|Kansas City Fed|Flash|S&P Global", 0),
    (r"CPI|PPI|PCE|Personal Income|Personal Spending|Retail Sales|Nonfarm|Non-Farm|Unemployment Rate|"
     r"Average Hourly|Durable|Goods Orders|Industrial Production|Capacity Utilization|Housing Starts|"
     r"Building Permits|Home Sales|Goods Trade Balance|Inventories|ISM|Import Price|Export Price|"
     r"Chicago Fed National|ADP Nonfarm|Challenger|Budget", 1),
]


def ref_period(name, day):
    """Chinese label for the period a release covers, or None when the schedule isn't known."""
    d = pd.Timestamp(day)
    for pat, rule in PERIOD_RULES:
        if not re.search(pat, name or "", re.I):
            continue
        if rule == "q":
            q = (d.to_period("Q") - 1)
            return f"{q.year} 年第 {q.quarter} 季"
        if rule in ("w-sat", "w-fri"):
            back = (d.weekday() - (5 if rule == "w-sat" else 4)) % 7 or 7
            end = d - pd.Timedelta(days=back)
            return f"截至 {end.month}/{end.day} 嗰星期"
        m = d.to_period("M") - (rule(d) if callable(rule) else rule)
        return f"{m.year} 年 {m.month} 月"
    return None


def _num(x):
    try:
        return float(re.sub(r"[^0-9.\-]", "", x or ""))
    except ValueError:
        return None


def forexfactory():
    """This week's US rows from ForexFactory: [{t (UTC), title, impact, forecast, previous}], or [] on failure."""
    try:
        r = requests.get(FF_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        rows = r.json()
    except Exception as e:
        print("forexfactory failed:", e)
        return []
    out = []
    for x in rows:
        if x.get("country") != "USD":
            continue
        try:
            t = pd.Timestamp(x["date"]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")
        except Exception:
            continue
        out.append(dict(t=t, title=x.get("title") or "", impact=x.get("impact"), f=x.get("forecast"), p=x.get("previous")))
    print(f"forexfactory: {len(out)} US rows this week")
    return out


def label(rows, ff):
    """Add ref (period covered), per (按月/按年/按季), perx (per is a guess) and imp (3 high, 2 medium) in place."""
    for r in rows:
        r["ref"] = ref_period(r.get("e"), r["d"])
    # 1) ForexFactory: same release time, and the forecast (or previous) is the same number
    for r in rows:
        best = None
        for x in ff:
            if x["t"] != r.get("u"):
                continue
            same = lambda a, b: a is not None and b is not None and abs(a - b) < 1e-6
            if same(_num(x["f"]), _num(r.get("c"))) or same(_num(x["p"]), _num(r.get("p"))):
                best = x
                break
        if best:
            m = re.search(r"\b(m/m|y/y|q/q)\b", best["title"])
            if m and (r.get("c") or r.get("p") or "").endswith("%"):
                r["per"] = PER[m.group(1)]
            r["imp"] = {"High": 3, "Medium": 2}.get(best["impact"])
    # 2) A same-name pair on one day where one side is known: the other is the opposite
    groups = {}
    for r in rows:
        groups.setdefault((r["d"], r.get("e")), []).append(r)
    for g in groups.values():
        if len(g) != 2 or not all((x.get("c") or x.get("p") or "").endswith("%") for x in g):
            continue
        a, b = g
        if a.get("per") and not b.get("per") and a["per"] in ("按月", "按年"):
            b["per"] = "按年" if a["per"] == "按月" else "按月"
        elif b.get("per") and not a.get("per") and b["per"] in ("按月", "按年"):
            a["per"] = "按年" if b["per"] == "按月" else "按月"
        elif not a.get("per") and not b.get("per"):  # next week: no ForexFactory, guess by size
            va, vb = (abs(_num(x.get("p") or x.get("c")) or 0) for x in g)
            if va != vb:
                small, big = (a, b) if va < vb else (b, a)
                small["per"], big["per"], small["perx"], big["perx"] = "按月", "按年", True, True
        if a.get("imp") or b.get("imp"):
            a["imp"] = b["imp"] = max(a.get("imp") or 0, b.get("imp") or 0) or None
    return rows
