"""python -m kcdt  ->  writes site/index.html and site/data.json"""
import json
import pathlib
import sys

from .compute import build, wants_fundamentals
from .fetch import fetch_prices, fundamentals, universe

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "site"


def render(data):
    tpl = (pathlib.Path(__file__).parent / "template.html").read_text(encoding="utf-8")
    body = tpl.replace("/*DATA*/", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    head = ('<!doctype html><html lang="zh-HK"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
            "body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>")
    return head + body + "</body></html>"


def main():
    meta = universe()
    px = fetch_prices(meta)
    fund = fundamentals(wants_fundamentals(px, meta), ROOT / "cache" / "fundamentals.json")
    data = build(px, meta, fund)
    if len(data["stocks"]) < 1000:
        sys.exit(f"only {len(data['stocks'])} stocks computed, refusing to publish")
    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (OUT / "index.html").write_text(render(data), encoding="utf-8")
    print(f"built {data['asof']}: score {data['hist'][-1]['score']}, "
          f"{len(data['stocks'])} stocks, {data['nfund']} with fundamentals")


if __name__ == "__main__":
    main()
