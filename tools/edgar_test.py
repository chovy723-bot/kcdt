"""Compare quarter dating from SEC EDGAR with the site's current method, on the cached fundamentals."""
import json, os, sys, time
import pandas as pd
import requests
sys.path.insert(0, "app")
from kcdt.compute import fundamentals_row

UA = {"User-Agent": "KCDT research kcdt-bot@users.noreply.github.com"}
f = json.load(open("app/cache/fundamentals.json"))
nq = json.load(open("app/cache/nasdaq_quarters.json")) if os.path.exists("app/cache/nasdaq_quarters.json") else {}
f = {t: {**r, **nq.get(t, {})} for t, r in f.items()}
rows = {t: fundamentals_row(r) for t, r in f.items()}
eq = sorted(t for t, r in rows.items() if r.get("eqv"))
ciks = {x["ticker"].upper(): x["cik_str"] for x in requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=60).json().values()}


def quarters(cik):
    """{quarter end: (diluted EPS, filed, derived)} from 10-Q/10-K XBRL facts."""
    for concept in ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"):
        r = requests.get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{concept}.json", headers=UA, timeout=30)
        time.sleep(0.12)
        if r.status_code != 200:
            continue
        facts = [x for x in r.json()["units"].get("USD/shares", []) if x.get("start") and x.get("form", "").startswith(("10-Q", "10-K"))]
        if not facts:
            continue
        q, ytd, fy = {}, {}, {}
        for x in sorted(facts, key=lambda x: x["filed"]):
            days = (pd.Timestamp(x["end"]) - pd.Timestamp(x["start"])).days
            if 80 <= days <= 100:
                q[x["end"]] = (x["val"], x["filed"], False)
            elif 260 <= days <= 285:
                ytd[x["end"]] = x["val"]
            elif 350 <= days <= 380:
                fy[x["end"]] = (x["val"], x["filed"])
        for end, (v, filed) in fy.items():  # Q4 is often only in the annual figure: year minus nine months
            if end not in q:
                nine = [e for e in ytd if 80 <= (pd.Timestamp(end) - pd.Timestamp(e)).days <= 100]
                if nine:
                    q[end] = (round(v - ytd[nine[0]], 2), filed, True)
        if q:
            return concept, q
    return None, {}


out = dict(n=len(eq), cur_dated=0, no_cik=0, no_facts=0, edgar=0, edgar_yoy=0, agree_month=0, disagree=[], newer=0, older=0,
           eps_match=0, eps_diff=[], derived_latest=0, gained=0, forms={})
for i, t in enumerate(eq):
    r, rec = rows[t], f[t]
    cur = r.get("eqd")
    out["cur_dated"] += bool(cur)
    cik = ciks.get(t.replace("-", "."), ciks.get(t))
    if not cik:
        out["no_cik"] += 1
        continue
    try:
        concept, q = quarters(cik)
    except Exception as e:
        print("fail", t, e)
        continue
    if not q:
        out["no_facts"] += 1
        continue
    out["edgar"] += 1
    ends = sorted(q, reverse=True)
    last = ends[0]
    ago = [e for e in ends if 350 <= (pd.Timestamp(last) - pd.Timestamp(e)).days <= 380]
    if not ago:
        continue
    out["edgar_yoy"] += 1
    out["derived_latest"] += q[last][2]
    em = pd.Timestamp(last).strftime("%Y-%m")
    if not cur:
        out["gained"] += 1
    else:
        if cur[0] == em:
            out["agree_month"] += 1
        else:
            (out.__setitem__("newer", out["newer"] + 1) if em > cur[0] else out.__setitem__("older", out["older"] + 1))
            out["disagree"].append([t, cur, [em, pd.Timestamp(ago[0]).strftime("%Y-%m")], r["eqv"], [q[last][0], q[ago[0]][0]]])
    new, old = r["eqv"]
    if abs(q[last][0] - new) <= 0.011 and abs(q[ago[0]][0] - old) <= 0.011:
        out["eps_match"] += 1
    else:
        out["eps_diff"].append([t, r["eqv"], [q[last][0], q[ago[0]][0]], em, cur, q[last][2]])
    if i % 100 == 0:
        print(i, t, {k: v for k, v in out.items() if not isinstance(v, list)}, flush=True)
out["disagree_n"], out["eps_diff_n"] = len(out["disagree"]), len(out["eps_diff"])
json.dump(out, open("shots/edgar.json", "w"), indent=1)
print({k: v for k, v in out.items() if not isinstance(v, list)})
