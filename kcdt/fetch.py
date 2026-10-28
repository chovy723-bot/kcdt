"""Download data: index member lists from Wikipedia, prices and fundamentals from Yahoo Finance."""
import io
import json
import pathlib
import time

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
    r = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100",
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36", "Accept": "application/json"},
                     timeout=30)
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
        if not missing:
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


def _one(t):
    tk = yf.Ticker(t)
    info = tk.info or {}
    rec = {v: info.get(k) for k, v in INFO_KEYS.items()}
    q = tk.quarterly_income_stmt
    if q is not None and not q.empty:
        q = q[sorted(q.columns, reverse=True)]  # newest first
        rec["revq"] = _row_values(q, ["Total Revenue", "Operating Revenue"])[:6]
        rec["epsq"] = _row_values(q, ["Diluted EPS", "Basic EPS"])[:6]
    a = tk.income_stmt
    if a is not None and not a.empty:
        a = a[sorted(a.columns, reverse=True)]
        rec["reva"] = _row_values(a, ["Total Revenue", "Operating Revenue"])[:3]
    return rec


def fundamentals(tickers, cache_file, max_age_days=7, max_calls=350, pause=0.6):
    """Refresh stale entries for `tickers` (oldest first), keep the rest from the cache."""
    path = pathlib.Path(cache_file)
    cache = json.loads(path.read_text()) if path.exists() else {}
    now = time.time()
    stale = sorted((t for t in tickers if now - cache.get(t, {}).get("_ts", 0) > max_age_days * 86400),
                   key=lambda t: cache.get(t, {}).get("_ts", 0))
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


def fetch_prices(u):
    extra = [t for t, _ in INDICES] + [t for t, _ in SECTORS] + DOW30 + ["QQQ"]
    return prices(sorted(set(list(u.index) + extra)))
