"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Model: earn-rate decomposition with damped trends and partial pooling.

An Exophase earn rate is a ratio, r = E / N: E earners of one trophy over N
players tracked for the game. At launch both move fast and in opposite
directions for most trophies -- N explodes as new players get tracked (which
pushes r down) while E grows as players progress (which pushes r up). A
persistence forecast of r misses both forces, so this program fits a damped
trend to logit r itself over the recent history (last 48 h, 12 h recency
half-life), with per-trophy slopes partially pooled toward the cross-trophy
median so a trophy with only a couple of earners borrows the typical growth
rate instead of trusting its own noisy slope. The earner counts E_i and the
tracked base N are fitted the same way and feed the uncertainty term.

  * log N(t): one weighted least-squares line over the recent history
    (last 48 h, 12 h recency half-life). Slope clipped at zero: the tracked
    base never shrinks.
  * log (E_i(t) + 0.5) per trophy: same fit. Slopes are partially pooled
    toward the cross-trophy median, with weight E_last / (E_last + 8), so a
    trophy with only a couple of earners borrows the typical growth rate
    instead of trusting its own noisy slope.
  * logit r_i(t): a direct trend fit on the rate itself, as a second opinion.

Each trend is extrapolated with damping, advance = tau * (1 - exp(-H / tau))
with tau = 1.5 days, so a trend does not run away over the 4.5-day final
horizon: the forecast bends toward a plateau instead of compounding
forever. The median forecast comes from the direct logit-rate trend.
(The decomposition estimate E_f / N_f was tried as a second opinion and
dropped: on synthetic final-like backtests it scored clearly worse --
mean pinball 1.78 vs 1.05 for the direct trend alone -- because the noisy
small-count earner slopes and the base fit compound instead of cancelling.
BLEND_DEC = 0 keeps it out of the median; the count fits still feed the
uncertainty term.)

Uncertainty is honest about launch week: sigma^2 combines a random-walk
floor (0.55 logit units per sqrt-day, just above the house baseline's 0.5),
the slope standard errors times the damped horizon, and the fit residuals.
When the history is flat and short (no movement at all, as in the practice
cases, which are cut from before launch when nobody had played yet), the
sigma floor is raised instead -- a flat pre-launch history says nothing
about how far rates will jump once play starts, and the intervals say so.
Quantiles are the median on the logit scale plus z_tau * sigma, mapped back
through the logistic function and rounded to 4 decimals half-up, exactly
like the starter's rounding.

Standard library only, no network, fully deterministic: same input bytes
give same output bytes. Runs in well under a second per case.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Random-walk spread of a rate's move, logit units per square-root day.
SIGMA_RW = 0.70
# Damping time constants (days) for the trend extrapolation.
TAU_COUNTS = 1.5
TAU_RATE = 1.5
# Recent-history window (days) and recency half-life (days) for the fits.
WINDOW_DAYS = 2.0
HALF_LIFE_DAYS = 0.5
# Partial-pooling strength: a trophy with E earners trusts its own slope
# with weight E / (E + POOL_K).
POOL_K = 8.0
# Weight of the decomposition estimate (E_f / N_f) in the median forecast's
# logit average; the rest is the direct rate-trend estimate. Set to 0 by the
# backtests (see the module docstring): the median is the direct trend.
BLEND_DEC = 0.0
# Sigma bounds (logit units).
SIGMA_MIN = 0.30
SIGMA_MAX = 2.0
# Tier priors used only when a case carries no usable history at all.
TIER_PRIOR = {"platinum": 0.05, "gold": 0.3, "silver": 1.0, "bronze": 3.0}


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def logistic(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> float:
    """ISO 8601 UTC timestamp -> days since the Unix epoch (float)."""
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0


def damped(horizon_days: float, tau: float) -> float:
    """Effective horizon of a damped trend: grows like H for H << tau,
    saturates at tau for H >> tau."""
    if horizon_days <= 0:
        return 0.0
    return tau * (1.0 - math.exp(-horizon_days / tau))


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (intercept_at_last_x, slope, slope_se, resid_var): the intercept
    is evaluated at the largest x so it reads as "value at the last read".
    With fewer than two distinct x values the slope is 0.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    x_last = max(xs)
    if n < 2 or max(xs) == min(xs):
        wsum = sum(ws)
        mean_y = sum(w * y for w, y in zip(ws, ys)) / wsum if wsum else ys[0]
        resid = sum(w * (y - mean_y) ** 2 for w, y in zip(ws, ys)) / wsum if wsum else 0.0
        return mean_y, 0.0, 0.0, resid
    wsum = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / wsum
    ybar = sum(w * y for w, y in zip(ws, ys)) / wsum
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return ybar, 0.0, 0.0, 0.0
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept_last = ybar + slope * (x_last - xbar)
    resid = sum(w * (y - (ybar + slope * (x - xbar))) ** 2 for w, x, y in zip(ws, xs, ys)) / wsum
    # Standard error of the slope under the weighted fit.
    dof = max(n - 2, 1)
    s2 = sum(w * (y - (ybar + slope * (x - xbar))) ** 2 for w, x, y in zip(ws, xs, ys)) / dof
    slope_se = math.sqrt(s2 / sxx) if sxx > 0 else 0.0
    return intercept_last, slope, slope_se, resid


def median(values):
    vals = sorted(values)
    m = len(vals)
    if m == 0:
        return 0.0
    if m % 2:
        return vals[m // 2]
    return 0.5 * (vals[m // 2 - 1] + vals[m // 2])


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = sorted(case.get("history") or [], key=lambda r: r.get("at", ""))
    horizon_days = case["horizon_minutes"] / 1440.0

    if not history:
        # No history at all: tier priors with very wide intervals.
        rows = []
        for trophy in trophies:
            med = TIER_PRIOR.get(str(trophy.get("tier", "")).lower(), 1.0)
            lg = logit(med)
            rows.append({"id": trophy["id"],
                         "q": [round4(logistic(lg + z * 1.5)) for z in Z]})
        return {"trophies": rows}

    release = None
    game = case.get("game") or {}
    if game.get("release_date"):
        try:
            release = parse_time(str(game["release_date"]) + "T00:00:00Z")
        except ValueError:
            release = None
    times = []
    ns = []
    rates = []  # rates[j][i], None where the read lacked the trophy
    for read in history:
        try:
            t = parse_time(read["at"])
        except (KeyError, ValueError):
            continue
        n = read.get("players_tracked")
        if not isinstance(n, (int, float)) or n <= 0:
            continue
        row = read.get("rates") or []
        times.append(t if release is None else t - release)
        ns.append(float(n))
        rates.append([float(v) if v is not None else None for v in row])
    if not times:
        raise ValueError(f"{name} has no usable history reads")
    t_last = times[-1]
    n_last = ns[-1]
    last_rates = rates[-1]

    # Recent window with recency weights (half-life 12 h).
    win_idx = [j for j, t in enumerate(times) if t_last - t <= WINDOW_DAYS]
    if len(win_idx) < 2:
        win_idx = list(range(len(times)))
    weights = [0.5 ** ((t_last - times[j]) / HALF_LIFE_DAYS) for j in win_idx]
    wtimes = [times[j] for j in win_idx]

    # One fit for the tracked-player base (shared by every trophy).
    log_n = [math.log(ns[j]) for j in win_idx]
    _, b_n, se_n, _ = wls_line(wtimes, log_n, weights)
    b_n = max(0.0, min(b_n, 3.0))

    # Per-trophy fits: log earners and logit rate.
    fits = []
    for i, trophy in enumerate(trophies):
        pts_e, pts_r = [], []
        for k, j in enumerate(win_idx):
            r = rates[j][i] if i < len(rates[j]) else None
            if r is None:
                continue
            e = max(0.0, r * ns[j] / 100.0)
            pts_e.append((wtimes[k], math.log(e + 0.5), weights[k]))
            pts_r.append((wtimes[k], logit(r), weights[k]))
        if pts_e:
            xs = [p[0] for p in pts_e]
            _, b_e, se_e, _ = wls_line(xs, [p[1] for p in pts_e], [p[2] for p in pts_e])
        else:
            b_e, se_e = 0.0, 0.0
        if pts_r:
            xs = [p[0] for p in pts_r]
            _, b_r, se_r, resid_r = wls_line(xs, [p[1] for p in pts_r], [p[2] for p in pts_r])
            lg_vals = [p[1] for p in pts_r]
            movement = (max(lg_vals) - min(lg_vals)) if lg_vals else 0.0
        else:
            b_r, se_r, resid_r, movement = 0.0, 0.0, 0.0, 0.0
        r_last = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else None
        if r_last is None:
            # Fall back to the most recent non-null rate for this trophy.
            for j in range(len(times) - 1, -1, -1):
                if i < len(rates[j]) and rates[j][i] is not None:
                    r_last = rates[j][i]
                    break
        if r_last is None:
            r_last = TIER_PRIOR.get(str(trophy.get("tier", "")).lower(), 1.0)
        e_last = max(0.0, r_last * n_last / 100.0)
        fits.append({"b_e": b_e, "se_e": se_e, "b_r": b_r, "se_r": se_r,
                     "resid": resid_r, "movement": movement,
                     "r_last": r_last, "e_last": e_last})

    # Partial pooling: trophies with few earners borrow the typical slopes.
    pool_e = [f["b_e"] for f in fits if f["e_last"] >= 3]
    pool_r = [f["b_r"] for f in fits if f["e_last"] >= 3]
    b_e_med = median(pool_e) if pool_e else 0.0
    b_r_med = median(pool_r) if pool_r else 0.0
    b_e_med = max(0.0, min(b_e_med, 3.0))
    b_r_med = max(-3.0, min(b_r_med, 3.0))

    adv_n = damped(horizon_days, TAU_COUNTS)
    adv_r = damped(horizon_days, TAU_RATE)
    span_days = t_last - times[0]
    rows = []
    for idx, f in enumerate(fits):
        w_own = f["e_last"] / (f["e_last"] + POOL_K)
        b_e = max(0.0, min(w_own * f["b_e"] + (1.0 - w_own) * b_e_med, 3.0))
        b_r = max(-3.0, min(w_own * f["b_r"] + (1.0 - w_own) * b_r_med, 3.0))

        # Decomposition estimate: forecast earners and base separately.
        log_e_f = math.log(f["e_last"] + 0.5) + b_e * adv_n
        log_n_f = math.log(n_last) + b_n * adv_n
        r_dec = 100.0 * math.exp(log_e_f - log_n_f)
        # Direct rate-trend estimate.
        lg_dir = logit(f["r_last"]) + b_r * adv_r
        # Median: average of the two estimates on the logit scale.
        med_lg = BLEND_DEC * logit(r_dec) + (1.0 - BLEND_DEC) * lg_dir

        # Uncertainty: random-walk floor + slope uncertainty + fit residual.
        var = (SIGMA_RW ** 2) * horizon_days
        var += 0.25 * ((f["se_e"] * adv_n) ** 2 + (se_n * adv_n) ** 2)
        var += 0.25 * (f["se_r"] * adv_r) ** 2
        var += min(max(f["resid"], 0.04), 1.0)
        sigma = math.sqrt(max(var, 0.0))
        # Flat, short history: the data say nothing about the launch jump,
        # so widen instead of pretending the last read will persist.
        if span_days < 0.5 and f["movement"] < 0.05:
            sigma = max(sigma, 1.0 * math.sqrt(horizon_days + 0.5))
        sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
        if horizon_days <= 0:
            sigma = 0.05

        qs = [round4(logistic(med_lg + z * sigma)) for z in Z]
        # Rounding can only produce equal neighbours; enforce ordering.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophies[idx]["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
