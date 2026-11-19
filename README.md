# GTA VI trophy rarity forecaster — make.fun gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zeetasw8-sketch/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zeetasw8-sketch/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast, for every one of Grand Theft Auto VI's 55 PlayStation 5 trophies, the percent of players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z — one week after launch — as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy. Scored by pinball loss on the logit scale.

## The model

An Exophase earn rate is a ratio, `r = E / N`: earners of one trophy over the players tracked for the game. At launch both move fast in opposite directions — `N` explodes as new players get tracked (pushing `r` down) while `E` grows as players progress (pushing `r` up). A persistence forecast misses both forces, so `main.py` fits a damped trend to `logit(r)` itself over the recent history (last 48 h, 12 h recency half-life):

- Per-trophy weighted least squares of `logit(rate)` on time; slopes are partially pooled toward the cross-trophy median (weight `E / (E + 8)`), so a trophy with only a couple of earners borrows the typical growth rate instead of trusting its own noisy slope.
- The trend is extrapolated with damping, `advance = tau * (1 - exp(-H / tau))` with `tau = 1.5` days, so it bends toward a plateau instead of compounding forever over the 4.5-day final horizon.
- The earner counts `E` and the tracked base `N` are fitted the same way and feed the uncertainty term.

Uncertainty is honest about launch week. Sigma (logit units) combines a random-walk floor (`0.70` per sqrt-day, above the house baseline's `0.50`), the slope standard errors times the damped horizon, and the fit residuals. When the history is flat and short — as in the practice cases, cut from before launch when nobody had played yet — the sigma floor is raised instead: a flat pre-launch history says nothing about how far rates will jump once play starts, and the intervals say so.

An earners-over-base decomposition (`E_f / N_f`) was tried as a second opinion for the median and dropped: on synthetic final-like backtests it scored clearly worse (mean pinball 1.78 vs 1.05 for the direct trend alone), because the noisy small-count earner slopes and the base fit compound instead of cancelling.

## Backtests

The kit's practice answers are not published, so no true practice score exists. Two honest substitutes, both scored with the kit's own pinball loss on the logit scale (lower is better):

| Test | House baseline (starter) | This entry |
| --- | --- | --- |
| Pseudo-case: p01 history (flat, to 03:30Z 19 Nov) → live Exophase snapshot 19 Nov ~18:49Z, 55 trophies | 2.337 | **1.891** |
| Synthetic final-like cases, 7 scenarios (tracked base logistic 638 → 150k; per-trophy earn shares rising logistically toward tier ceilings; cut at 2.7 days, target at 7.0 days) — mean over scenarios | ~2.05 (3 seeds) | **1.04** |
| Same synthetic cases, worst single scenario | 3.44 | **2.04** |

The synthetic grid also tested the tuning: damping `tau` 0.7 / 1.0 / 1.5, random-walk sigma 0.55 / 0.70 / 0.85, and blend weights 0 / 0.5 / 1.0 for the decomposition. The shipped setting (`tau = 1.5`, `sigma_rw = 0.70`, decomposition weight 0) had the best mean and the best worst case across the 7 scenarios, not just the best on one seed. 90% interval coverage on the synthetic cases: 0.65–0.96 per scenario (baseline: 0.29–0.65).

Validity checks on the practice cases: output schema, one row per trophy, five non-decreasing quantiles in [0, 100], under 65,536 bytes — pass; two runs write byte-identical output; runtime ~0.03 s per case (limit 8 s). Example row from the p00 practice run (trophy 1, platinum, last rate 0.63%): `{"id": 1, "q": [0.1843, 0.3808, 0.63, 1.0405, 2.1306]}` — the median stays at the last read (flat history, no trend to follow) and the wide interval is the honest part.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic. Reads every `<id>.in` in the input directory and writes `<id>.out` beside it in the output directory.

## Files

- `main.py` — the forecaster (the whole entry; see its docstring for the full model).
- `README.md` — this file.
- `LICENSE` — MIT.
- `.makefun/owner.json` — payout wallet.
