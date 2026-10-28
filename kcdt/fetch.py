"""Download data: index member lists from Wikipedia, prices and fundamentals from Yahoo Finance."""
import io
import json
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import yfinance as yf

from .compute import DOW30, INDICES, SECTORS

WIKI = "https://en.wikipedia.org/wiki/"
LISTS = {  # name -> Wikipedia page
    "ndx": "Nasdaq-100",
    "sp500": "List_of_S%26P_500_companies",
    "sp400": "List_of_S%26P_400_companies",
    "sp600": "List_of_S%26P_600_companies",
}
UA = {"User-Agent": "Mozilla/5.0 KCDT"}
NASDAQ_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
             "Chrome/124.0 Safari/537.36", "Accept": "application/json"}


def _pick(cols, *keys):
    for k in keys:
        for name, orig in cols.items():
            if name.startswith(k):
                return orig
    return None


def _members(page):
    seen = []
    for url in (WIKI + page, "https://en.wikipedia.org/api/rest_v1/page/html/" + page):
        r = requests.get(url, headers=UA, timeout=30)
        try:
            tables = pd.read_html(io.StringIO(r.text))
        except ValueError:
            tables = []
        seen.append(f"{url} -> HTTP {r.status_code}, {len(r.text)} bytes, table rows {[len(t) for t in tables]}")
        for t in tables:
            if isinstance(t.columns, pd.MultiIndex):
                t.columns = [" ".join(str(x) for x in c if "Unnamed" not in str(x)) for c in t.columns]
            cols = {str(c).strip().lower(): c for c in t.columns}
            sym = _pick(cols, "symbol", "ticker")
            name = _pick(cols, "security", "company", "name")
            sec = _pick(cols, "gics sector", "sector", "icb industry", "industry")
            if sym is not None and name is not None and len(t) >= 80:
                return pd.DataFrame({
                    "s": t[sym].astype(str).str.strip().str.replace(".", "-", regex=False),
                    "n": t[name].astype(str),
                    "gics": t[sec].astype(str) if sec is not None else None,
                })
            if len(t) >= 80:
                seen.append(f"  columns {list(cols)}")
    raise RuntimeError(f"no member table found on {page}: " + " | ".join(seen))


def _nasdaq100():
    """Nasdaq's own list (Wikipedia's Nasdaq-100 page no longer carries the member table)."""
    r = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=NASDAQ_UA, timeout=30)
    rows = r.json()["data"]["data"]["rows"]
    return pd.DataFrame({"s": [x["symbol"].strip().replace(".", "-") for x in rows],
                         "n": [x["companyName"] for x in rows], "gics": None})


def _list(key, page, cache):
    """Fetch one member list; fall back to the last good copy so one bad source doesn't stop the update."""
    path = cache / f"members-{key}.json"
    tries = [_nasdaq100, lambda: _members(page)] if key == "ndx" else [lambda: _members(page)]
    for f in tries:
        try:
            m = f()
            if len(m) >= 80:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(m.to_json(orient="records"))
                return m
        except Exception as e:
            print(f"{key}: source failed: {e}")
    if path.exists():
        print(f"{key}: using cached member list")
        return pd.read_json(path, orient="records")
    raise RuntimeError(f"no member list for {key}")


def universe(cache="cache"):
    """One row per ticker: name, GICS sector and which lists it belongs to."""
    frames = []
    for key, page in LISTS.items():
        m = _list(key, page, pathlib.Path(cache))
        m[key] = True
        frames.append(m)
        print(f"{key}: {len(m)} members")
    u = pd.concat(frames).groupby("s").agg({
        "n": "first", "gics": "first",
        **{k: lambda x: bool(x.fillna(False).any()) for k in LISTS}})
    for k in LISTS:
        u[k] = u[k].fillna(False).astype(bool)
    u["gics"] = u["gics"].fillna("")
    return u


def _download(tickers, period):
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=False)
    return {f: df[f] for f in ("Open", "High", "Low", "Close", "Volume")}


def prices(tickers, period="2y", chunk=50, pause=3):
    """Download in small batches with pauses; Yahoo rate-limits big bursts from cloud IPs."""
    parts = {f: [] for f in ("Open", "High", "Low", "Close", "Volume")}
    todo = list(tickers)
    missing = []
    for attempt in range(4):
        missing = []
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            try:
                got_df = _download(part, period)
            except Exception as e:
                print("batch failed:", e)
                missing += part
                time.sleep(pause * 10)
                continue
            c = got_df["Close"]
            got = [t for t in part if t in c and not c[t].dropna().empty]
            for f in parts:
                parts[f].append(got_df[f][got])
            missing += [t for t in part if t not in got]
            time.sleep(pause)
        print(f"attempt {attempt + 1}: {len(tickers) - len(missing)}/{len(tickers)} downloaded")
        if len(missing) <= max(3, len(tickers) // 200):  # a few delisted symbols: not worth minutes of retries
            break
        todo = missing
        time.sleep(60 * (attempt + 1))
    if len(missing) > len(tickers) * 0.05:
        raise RuntimeError(f"{len(missing)} tickers missing, e.g. {missing[:10]}")
    out = {}
    for f, frames in parts.items():
        df = pd.concat(frames, axis=1)
        out[f] = df.loc[:, ~df.columns.duplicated()]
    return out


# ---- fundamentals (cached; Yahoo allows only a few hundred of these calls per run) ----

INFO_KEYS = {
    "marketCap": "mcap", "beta": "beta", "heldPercentInstitutions": "inst",
    "revenueGrowth": "sq", "earningsQuarterlyGrowth": "eq", "trailingPE": "pe",
    "forwardPE": "fpe", "sector": "ysec", "industry": "ind",
    "firstTradeDateMilliseconds": "ipo", "earningsTimestamp": "edate",
}


def _row_values(df, names):
    for n in names:
        if n in df.index:
            return [None if pd.isna(v) else float(v) for v in df.loc[n].tolist()]
    return []


SCHEMA = 2  # bump when _one() starts collecting new fields, so old cache entries refresh first


def _estimate_growth(df):
    """Analyst growth estimates for this and next fiscal year, as fractions."""
    if df is None or df.empty or "growth" not in df:
        return None, None
    g = df["growth"]
    val = lambda k: None if k not in g.index or pd.isna(g[k]) else float(g[k])
    return val("0y"), val("+1y")


def _one(t):
    tk = yf.Ticker(t)
    info = tk.info or {}
    rec = {v: info.get(k) for k, v in INFO_KEYS.items()}
    rec["_v"] = SCHEMA
    q = tk.quarterly_income_stmt
    if q is not None and not q.empty:
        q = q[sorted(q.columns, reverse=True)]  # newest first
        rec["revq"] = _row_values(q, ["Total Revenue", "Operating Revenue"])[:8]
        rec["epsq"] = _row_values(q, ["Diluted EPS", "Basic EPS"])[:8]
    a = tk.income_stmt
    if a is not None and not a.empty:
        a = a[sorted(a.columns, reverse=True)]
        rec["reva"] = _row_values(a, ["Total Revenue", "Operating Revenue"])[:4]
        rec["epsa"] = _row_values(a, ["Diluted EPS", "Basic EPS"])[:4]
    # Reported EPS goes back further here than in the income statement (needed for 3 quarters year on year)
    try:
        e = tk.get_earnings_dates(limit=16)
        if e is not None and not e.empty and "Reported EPS" in e:
            e = e["Reported EPS"].dropna().sort_index(ascending=False)
            rec["epsh"] = [float(x) for x in e.tolist()[:8]]
    except Exception as ex:
        print("earnings dates failed:", t, ex)
    for name, key in (("earnings_estimate", "efc"), ("revenue_estimate", "rfc")):
        try:
            rec[key] = list(_estimate_growth(getattr(tk, name)))
        except Exception as ex:
            print(f"{name} failed:", t, ex)
    return rec


def fundamentals(tickers, cache_file, max_age_days=7, max_calls=250, pause=0.6):
    """Refresh stale entries for `tickers` (old schema first, then oldest), keep the rest from the cache."""
    path = pathlib.Path(cache_file)
    cache = json.loads(path.read_text()) if path.exists() else {}
    now = time.time()
    age = lambda t: (cache.get(t, {}).get("_v", 1) >= SCHEMA, cache.get(t, {}).get("_ts", 0))
    stale = sorted((t for t in tickers if age(t)[0] is False or now - age(t)[1] > max_age_days * 86400), key=age)
    done = fails = 0
    for t in stale[:max_calls]:
        try:
            rec = _one(t)
            rec["_ts"] = now
            cache[t] = rec
            done += 1
        except Exception as e:
            fails += 1
            print("fundamentals failed:", t, e)
            if fails > 25:
                print("too many failures, stopping fundamentals refresh for this run")
                break
            time.sleep(pause * 10)
        time.sleep(pause)
    print(f"fundamentals: {done} refreshed, {len(stale) - done} still stale, {len(cache)} cached")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))
    return cache


# ---- quarterly revenue / EPS history from Nasdaq (Yahoo only gives ~5 quarters, too few for 3 quarters YoY) ----

def _money(x):
    """'$44,062(m)' -> 44062e6, '$0.76 (04/27/2025)' -> 0.76, '' -> None."""
    m = re.match(r"\s*(-?)\$?(-?)([\d,]*\.?\d+)\s*(\((m|b|k)\))?", x or "")
    if not m:
        return None
    v = float(m.group(3).replace(",", "")) * {"m": 1e6, "b": 1e9, "k": 1e3}.get(m.group(5), 1)
    return -v if (m.group(1) or m.group(2)) else v


def _nasdaq_quarters(t):
    r = requests.get(f"https://api.nasdaq.com/api/company/{t.replace('-', '.')}/revenue?limit=1", headers=NASDAQ_UA, timeout=30)
    rows = ((r.json().get("data") or {}).get("revenueTable") or {}).get("rows") or []
    cols = ["value2", "value3", "value4"]  # fiscal years, newest first
    quarters, cur = [], None
    for row in rows:
        label = row.get("value1", "")
        if label == "Totals":
            break
        if label == "Revenue" and cur is not None:
            cur["rev"] = [_money(row.get(c)) for c in cols]
        elif label == "EPS" and cur is not None:
            cur["eps"] = [_money(row.get(c)) for c in cols]
        elif label not in ("Revenue", "EPS", "Dividends"):
            cur = {"rev": [None] * 3, "eps": [None] * 3}
            quarters.append(cur)
    # newest first: latest fiscal year's quarters in reverse order, then the year before, ...
    seq = [(q["rev"][i], q["eps"][i]) for i in range(3) for q in reversed(quarters)]
    while seq and seq[0][0] is None:  # quarters of the current year not reported yet
        seq.pop(0)
    out = []
    for rev, eps in seq:  # keep only the unbroken run
        if rev is None:
            break
        out.append((rev, eps))
    return {"nrev": [x[0] for x in out], "neps": [x[1] for x in out]}


def nasdaq_quarters(tickers, cache_file, max_age_days=7, max_calls=400, workers=4):
    path = pathlib.Path(cache_file)
    cache = json.loads(path.read_text()) if path.exists() else {}
    now = time.time()
    stale = sorted((t for t in tickers if now - cache.get(t, {}).get("_ts", 0) > max_age_days * 86400),
                   key=lambda t: cache.get(t, {}).get("_ts", 0))[:max_calls]

    def one(t):
        try:
            rec = _nasdaq_quarters(t)
            rec["_ts"] = now
            return t, rec
        except Exception as e:
            print("nasdaq quarters failed:", t, e)
            return t, None

    with ThreadPoolExecutor(workers) as ex:
        got = [(t, rec) for t, rec in ex.map(one, stale) if rec is not None]
    cache.update(got)
    print(f"nasdaq quarters: {len(got)}/{len(stale)} refreshed, {len(cache)} cached")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))
    return cache


# ---- earnings calendar with timing (before the open / after the close) from Nasdaq ----

EARN_TIME = {"time-pre-market": "bmo", "time-after-hours": "amc"}


def earnings_calendar(days=15):
    """{ticker: [date, "bmo" | "amc" | None]} for the next `days` calendar days."""
    out = {}
    start = pd.Timestamp.now(tz="America/New_York").normalize()
    for d in pd.date_range(start, periods=days):
        if d.weekday() >= 5:
            continue
        day = d.strftime("%Y-%m-%d")
        try:
            r = requests.get(f"https://api.nasdaq.com/api/calendar/earnings?date={day}", headers=NASDAQ_UA, timeout=30)
            rows = ((r.json().get("data") or {}).get("rows")) or []
        except Exception as e:
            print("earnings calendar failed:", day, e)
            continue
        for row in rows:
            sym = (row.get("symbol") or "").strip().replace(".", "-")
            if sym and sym not in out:
                out[sym] = [day, EARN_TIME.get(row.get("time"))]
        time.sleep(0.5)
    print(f"earnings calendar: {len(out)} companies in the next {days} days")
    return out


# ---- CNN Fear & Greed (no official API; keep the last good copy if it fails) ----

def fear_greed(cache_file):
    path = pathlib.Path(cache_file)
    try:
        r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                                  "Referer": "https://edition.cnn.com/", "Accept": "application/json"},
                         timeout=30)
        r.raise_for_status()
        j = r.json()
        now = j["fear_and_greed"]
        hist = j.get("fear_and_greed_historical", {}).get("data", [])
        out = dict(score=round(float(now["score"]), 1), rating=now.get("rating"),
                   prev=now.get("previous_close"), w1=now.get("previous_1_week"),
                   m1=now.get("previous_1_month"), y1=now.get("previous_1_year"),
                   ts=str(now.get("timestamp", ""))[:10],
                   hist=[[time.strftime("%Y-%m-%d", time.gmtime(x["x"] / 1000)), round(float(x["y"]), 1)]
                         for x in hist][-260:],
                   live=True)
        for k in ("prev", "w1", "m1", "y1"):
            out[k] = None if out[k] is None else round(float(out[k]), 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        print(f"fear & greed: {out['score']} {out['rating']}")
        return out
    except Exception as e:
        print("fear & greed failed:", e)
        if path.exists():
            out = json.loads(path.read_text())
            out["live"] = False
            return out
        return None


def fetch_prices(u):
    extra = [t for t, _ in INDICES] + [t for t, _ in SECTORS] + DOW30 + ["QQQ", "^NDX", "^VXN"]
    return prices(sorted(set(list(u.index) + extra)))
