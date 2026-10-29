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
        if "faireconomy" in url:
            raise OSError("offline")
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




def test_econ_labels():
    from kcdt.econ import label
    rows = [dict(d="2026-10-29", u="2026-10-29T12:30Z", e="Core PCE Price Index", a=None, c="0.3%", p="0.2%"),
            dict(d="2026-10-29", u="2026-10-29T12:30Z", e="Core PCE Price Index", a=None, c="3.1%", p="3.0%"),
            dict(d="2026-10-29", u="2026-10-29T12:30Z", e="GDP", a=None, c="3.2%", p="2.2%"),
            dict(d="2026-11-12", u="2026-11-12T13:30Z", e="CPI", a=None, c="0.2%", p="0.3%"),
            dict(d="2026-11-12", u="2026-11-12T13:30Z", e="CPI", a=None, c="2.9%", p="3.0%")]
    ff = [dict(t="2026-10-29T12:30Z", title="Core PCE Price Index m/m", impact="High", f="0.3%", p="0.2%"),
          dict(t="2026-10-29T12:30Z", title="Advance GDP q/q", impact="High", f="3.2%", p="1.5%")]
    label(rows, ff)
    pce_m, pce_y, gdp, cpi_m, cpi_y = rows
    assert (pce_m["per"], pce_y["per"]) == ("按月", "按年") and not pce_m.get("perx") and pce_m["imp"] == pce_y["imp"] == 3
    assert pce_m["ref"] == "2026 年 9 月"
    assert gdp["per"] == "按季（年率）" and gdp["ref"] == "2026 年第 3 季"
    assert (cpi_m["per"], cpi_y["per"], cpi_m["perx"]) == ("按月", "按年", True) and cpi_m["ref"] == "2026 年 10 月"


def test_econ_labels_remembered_and_notes():
    from kcdt.econ import label
    memory = {}
    week1 = [dict(d="2026-10-14", u="2026-10-14T12:30Z", e="CPI", c="0.2%", p="0.3%"),
             dict(d="2026-10-14", u="2026-10-14T12:30Z", e="CPI", c="2.9%", p="3.0%")]
    label(week1, [dict(t="2026-10-14T12:30Z", title="CPI y/y", impact="High", f="2.9%", p="3.0%")], memory)
    assert [r["per"] for r in week1] == ["按月", "按年"] and memory["CPI"] == {"0": "按月", "1": "按年"}
    # a later week with no ForexFactory rows: the remembered order is used, not a guess
    week2 = [dict(d="2026-11-12", u="2026-11-12T13:30Z", e="CPI", c="0.4%", p="0.2%"),
             dict(d="2026-11-12", u="2026-11-12T13:30Z", e="CPI", c="0.1%", p="2.9%")]
    label(week2, [], memory)
    assert [r["per"] for r in week2] == ["按月", "按年"] and not any(r.get("perx") for r in week2)
    notes = [dict(d="2026-10-28", u="2026-10-28T18:00Z", e="Fed Interest Rate Decision", c="4.00%", p="4.00%"),
             dict(d="2026-10-27", u="2026-10-27T17:00Z", e="2-Year Note Auction", c=None, p="4.787%"),
             dict(d="2026-10-27", u="2026-10-27T12:55Z", e="Redbook", c=None, p="9.0%"),
             dict(d="2026-10-29", u="2026-10-29T20:30Z", e="Fed's Balance Sheet", c=None, p="6,754B")]
    label(notes, [], {})
    assert [r["note"] for r in notes] == ["議息會議結果", "國債拍賣 · 數字係得標息率",
                                         "截至 10/24 嗰星期 · 大型零售商同店銷售按年", "截至 10/28（星期三）"]


if __name__ == "__main__":
    test_economic_calendar_dates_rows_one_day_back()
    test_econ_labels()
    test_econ_labels_remembered_and_notes()
    print("ok")
