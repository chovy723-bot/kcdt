# KCDT 美股趨勢溫度計

每個交易日美股收市後，GitHub Actions 會：

1. 由 Wikipedia 拎 S&P 500 成份股名單
2. 由 Yahoo Finance 下載兩年日線（已調整拆股及派息）
3. 計市場溫度（8 個指標）、市場寬度、板塊強弱、強勢股
4. 生成 `site/index.html`，發佈到 GitHub Pages

本地試行：

```
pip install -r requirements.txt
python -m kcdt          # 生成 site/index.html
python -m tests.test_build   # 唔使上網嘅測試
```

計法同門檻喺 `kcdt/compute.py`。只供數據及教育用途，唔構成投資建議。
