# Temperature score backtest

- `fetch_history.py` (run by the "Backtest data" workflow) downloads daily prices since 2003 into `data/`.
- `analyze.py` rebuilds the 8 indicators from 2004, scores them with the site's current thresholds and with
  percentile-calibrated ones (fitted on 2004–2015, checked on 2016 onwards), and compares the score with
  the next 1–3 months of S&P 500 and QQQ. Output goes to `out/` (`report.txt` is the printed summary).

Caveats: the "30 leading stocks" are today's Dow 30 applied to the past (survivorship bias), so the run
repeats everything with the 9 original sector ETFs as a bias-free basket. Results are for education only.
