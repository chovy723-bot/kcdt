"""Offline checks for the Nasdaq parsers (no network)."""
import sys, types, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))
from kcdt import fetch


class _Resp:
    def __init__(self, rows): self.rows = rows
    def json(self): return {"data": {"rows": self.rows}}


def test_economic_calendar_dates_rows_one_day_back():
    asked = []

    def fake_get(url, **kw):
        day = url.rsplit("=", 1)[1]
        asked.append(day)
        if day == "2026-10-29":  # Nasdaq returns the 2026-10-28 FOMC decision under the next day's date
            return _Resp([{"country": "United States", "gmt": "14:00", "eventName": "Fed Interest Rate Decision",
                           "actual": "4.00%", "consensus": "4.00%", "previous": "4.00%"},
                          {"country": "United States", "gmt": "14:00", "eventName": "FOMC Statement",
                           "actual": "&nbsp;", "consensus": " ", "previous": "&nbsp;"},
                          {"country": "Japan", "gmt": "01:00", "eventName": "CPI", "actual": "1%"}])
        return _Resp([])

    real_get, real_sleep = fetch.requests.get, fetch.time.sleep
    fetch.requests.get, fetch.time.sleep = fake_get, lambda s: None
    try:
        rows = fetch.economic_calendar(today="2026-10-29")
    finally:
        fetch.requests.get, fetch.time.sleep = real_get, real_sleep
    assert asked[0] == "2026-10-27"  # Monday 10-26 is asked for as 10-27
    fed = [r for r in rows if r["e"] == "Fed Interest Rate Decision"][0]
    assert fed["d"] == "2026-10-28" and fed["u"] == "2026-10-28T18:00Z", fed  # 14:00 EDT
    stmt = [r for r in rows if r["e"] == "FOMC Statement"][0]
    assert stmt["a"] is None and stmt["p"] is None and stmt["c"] is None
    assert all(r["e"] != "CPI" for r in rows)


if __name__ == "__main__":
    test_economic_calendar_dates_rows_one_day_back()
    print("ok")
