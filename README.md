# Real-Time Estimation of Fair Implied Volatility from Bitcoin Option Trades

Aleksandar Hristev, Bocconi University

This repository holds the model code of the paper. It prices Deribit BTC options with Black-76 on the inverse (BTC-quoted) convention, cleans option trades and derives a causal forward, log-moneyness, greeks and tick noise for each trade. Its core is a per-expiry Kalman filter that updates a volatility smile and a per-option deviation after every trade, in a fast factor form and an exact dense form, with maximum-likelihood estimation, noise multipliers and a fused variant that also takes the exchange mark as an observation. The benchmarks (desk rules, rolling smile fits, SVI, last trade plus surface move, rebuilt mark), the impact and realised-spread measures with a day bootstrap, and the features and Huber regressions of the learned models are included as well.

## Modules

| Module | Contents |
| --- | --- |
| `fairvol.config` | settings loader (`config/settings.yaml`) |
| `fairvol.pricing.black76` | Black-76 prices, greeks and implied volatility, USD and inverse |
| `fairvol.pricing.forward` | pricing conventions, trade-implied and put-call parity forwards, windowed medians |
| `fairvol.data.instruments` | Deribit instrument names, tick schedule |
| `fairvol.data.clean` | cleaning rules and trade-class flags |
| `fairvol.data.prepare` | causal forward, log-moneyness, delta, vega, tick noise, mark iv per trade |
| `fairvol.buckets` | expiry, delta and quantile buckets |
| `fairvol.filter.kalman` | per-expiry smile Kalman filter, factor form |
| `fairvol.filter.dense` | the same filter with the exact covariance, per-event noise multipliers and clock |
| `fairvol.filter.events` | event arrays from trades and grid queries, filter constants |
| `fairvol.filter.params` | specifications, free parameters, transforms and bounds |
| `fairvol.filter.fit` | maximum likelihood (L-BFGS-B) and Hessian standard errors |
| `fairvol.filter.simulate` | simulation from the filter's own model |
| `fairvol.filter.run` | exact filter with noise multipliers, per-trade and per-query outputs |
| `fairvol.filter.fused` | fused filter with mark events |
| `fairvol.benchmarks.halfspread` | half-spreads by cell |
| `fairvol.benchmarks.hsmodel` | half-spread model in price terms (least absolute deviations) |
| `fairvol.benchmarks.desk_rules` | last trade and rolling smile rules |
| `fairvol.benchmarks.rolling` | rolling quadratic and smile-basis fits on a trailing window |
| `fairvol.benchmarks.svi` | SVI, last trade plus surface move, rebuilt mark |
| `fairvol.impact.features` | time since last trade, signed flow, index returns and realised volatility |
| `fairvol.impact.markout` | first observation in a look-forward window, day-bootstrap means |
| `fairvol.impact.measure` | observation search, cells, reference labels |
| `fairvol.evaluation.bootstrap` | day-clustered bootstrap of impact with re-estimated cells |
| `fairvol.learned.features` | trade features for the learned models |
| `fairvol.learned.design` | design matrices, Huber regression and loss |

`parameters/selected.json` holds the two selected filters (F1-Q-res1-spot1 and F0-HW-res1-spot1: basis, weighting, per-option deviation `residual`, spot response `spot`, and the estimated parameters), the constant noise multiplier of F1, and the trade and mark-noise multipliers of the fused filter.

## Installation

Python 3.12 or later.

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

Run from the repository root. The example simulates two days of trades from the filter's own model and runs the selected F1 filter over them.

```python
import json

import numpy as np

from fairvol.config import load_settings
from fairvol.filter.events import assemble, filter_consts, trade_frame_events
from fairvol.filter.kalman import OUT_FIELDS, run_filter
from fairvol.filter.params import ParamMap, Spec
from fairvol.filter.simulate import simulate

s = load_settings("config/settings.yaml")
sel = json.load(open("parameters/selected.json"))["filters"]["F1-Q-res1-spot1"]
spec = Spec(sel["basis"], sel["weighting"], sel["residual"], sel["spot"])
pm = ParamMap(spec, s)
nat = pm.vector(np.array([sel["parameters"][n] for n in pm.names]))
consts = filter_consts(s)
cfg = {"t0_ms": 1735689600000, "days": 2, "expiry_tau0_days": [6.5, 30.0], "n_strikes": 15, "x_span": 3.0,
       "rate_per_day": 400, "batch_prob": 0.13, "spot0": 90000.0, "spot_vol": 0.5, "theta0": [55.0, -2.0, 3.0],
       "hs_c_tick": 0.3, "hs_c_vega": 0.15}
tr = simulate(spec, nat, consts, cfg, s, seed=1)
ev = assemble([trade_frame_events(tr, s, use_legs=False)])
ll, n, _, out = run_filter(*ev.arrays(), nat, spec.basis_id, spec.weighting_id, spec.residual, spec.spot,
                           consts, int(s["filter"]["burn_in_updates"]), cfg["t0_ms"], True)
err = out[OUT_FIELDS.index("m_prior")] - tr["true_fair"].to_numpy()[ev.src]
print(f"{len(tr)} trades, log-likelihood per trade {ll / n:.3f}, mean abs error {np.nanmean(np.abs(err)):.3f} vol points")
```

## Data

No data are included. Deribit option trades come from the exchange's public history interface, method `public/get_last_trades_by_currency_and_time`. Order-book data are the free first-day-of-month files of Tardis.dev.

## Licence

MIT, see `LICENSE`.
