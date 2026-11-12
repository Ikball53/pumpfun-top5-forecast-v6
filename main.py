# pumpfun-top5-forecast-01 entry v6: forecast the 5 largest pump.fun coins
# by market cap at each answer read.
#
# v6 = v5 + ONE principled fix, to the newborn level nb_mu (the size
# bottleneck).  Everything else is v5 bit-for-bit: v3's size
# construction, v4's order improvement (40 extra named coins/place,
# smoothing 0.05), the same Monte Carlo, unnamed-mass target and
# growth ceiling.
#
# THE FIX -- nb_mu was mis-centred ~0.3 dex too low.  v5 pooled every
# (cut, read) newborn-winner pair with only a weak linear recency
# weight, so the early marks -- when the market had just opened and
# newborn winners sat at 1.85-3.0 dex -- dragged nb_mu to 3.10 on the
# practice, even though the newborn winners in the LATE regime (the
# level that matters for a forecast from the last mark) sit at
# 3.31-3.56 dex (mean 3.41, n=4).  That mis-centred the whole
# newborn component -- ~37% of the size mixture, and the MC's
# newborn draws -- well below where the true caps live.
# v6 takes nb_mu from the newborn winners of the LAST 3 marks only
# (the recent regime), shrunk toward the prior (strength 8.0) so a
# small late sample cannot swing it far; with fewer than 3 late
# winners it falls back to v5's pooled estimate.  On the practice
# nb_mu moves 3.10 -> 3.30.  nb_sd stays the pooled value (0.43).
#
# WHY CENTRING, NOT WIDENING (validated, not assumed): v4 already
# diagnosed v3's sizes as too wide (sd 0.56-0.69) and tried tight
# ones -- they scored 2 nats WORSE because they were mis-centred.
# The peak-bin probabilities of v5's size distributions average to
# ln -2.71 per place, which is exactly the leader's achieved size
# per place (-13.56/5 = -2.71).  So v5's WIDTHS are already at the
# leader's level; the 2.63-nat size gap is the true values not
# landing on our peaks -- a CENTRING problem, not a width problem.
# Widening would lower the peaks and make it worse.  v6 fixes the
# largest measured centring error (nb_mu) and leaves the widths
# alone.  If v6's size does not improve, the remaining error is in
# the incumbent (MC) centre, not the newborn one.
#
# State (16:59 WIB 2026-11-12): v5 is the best so far, -35.9
# (order -19.01, size -16.86), rank 194/761.  Top 10 (the money)
# is -32.8 (order -18.54, size -14.23); leader -31.7 (order
# -18.18, size -13.56).  The size gap to top 10 is 2.63 nats,
# the order gap only 0.47 -- size is the bottleneck.
#
# Honest caveat: the local replica (kit-history splits) is UNRELIABLE
# as a predictor of the dev score -- it said v3 would be worse than v2
# on every split, yet v3 scored 3.4 nats better on the real dev, and it
# said v4 was only slightly worse than v3 on sizes, yet v4 was 2 nats
# worse on the real dev.  None of v6's numbers are tuned to the
# replica; the nb_mu change is a measured calibration (late-regime
# newborn winners in the input history), and its effect is graded on
# the real system.  If v6 scores worse than v5, revert to v5 by
# resubmitting the v5 repo, since Force_Last lets the parent
# restore it.
#
# Determinism: one fixed seed (20261112), candidates iterated in sorted
# mint order, no set iteration, no wall clock, no network, no randomness
# outside the seeded generator.  Two runs on the same history write
# byte-identical output.

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 20261112
SCENARIOS = 8000
SMOOTH_PLACE = 0.05       # Laplace share spread over the named coins (v4: 0.05, from v3's 0.02)
MAX_CANDIDATES = 60
MAX_ROWS_PER_PLACE = 100
NAMED_EXTRA = 40          # candidates named per place beyond the MC winners (v4's order improvement)
DEAD_CAP = 1.45           # log10 SOL cap of a dead curve coin
DEAD_SD = 0.08
SIGMA0 = 0.05             # base sd (dex) at the cut
SIGMA_CAP = 0.60          # cap on total sd (dex) at any horizon

# Priors (used only when the history is too thin to measure a quantity;
# every value is shrunk toward these by the measured sample size).
PRIOR_NB_MU = 3.25
PRIOR_NB_SD = 0.30
PRIOR_NB_RATE = 1.6       # strong newborns (>3.0 dex) per 10 min
PRIOR_TURNOVER = 0.20     # fraction of top-5 places changing per 10 min
PRIOR_VOL_POOL = 0.34     # pool sd (dex) per 10 min
PRIOR_VOL_CURVE = 0.094   # curve sd (dex) per 10 min
PRIOR_GROWTH = 0.05       # pool median 10-min change (dex)
SIZE_PRIOR_WEIGHT = 0.10  # weight of the broad normal mixed into sizes
SIZE_PRIOR_SD = 0.60      # sd (dex) of the broad normal mixed into sizes
U_MAX = 0.50              # cap on the unnamed-mass target


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def floor6(x):
    """Round down to 6 decimals (never -0), like the house starter."""
    r = math.floor(x * 1_000_000) / 1_000_000
    return 0.0 if r == 0 else r


def normal_cdf(x, mu, sd):
    if sd <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0))))


def bin_of(value, edges):
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _shrink(measured, prior, n, strength=5.0):
    """Shrink a measured value toward a prior by sample size."""
    if n <= 0:
        return prior
    return (n * measured + strength * prior) / (n + strength)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    with open(os.path.join(in_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    out_name = task.get("output", "forecast.json")
    n_bins = len(edges) + 1

    marks = []
    with open(os.path.join(in_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise ValueError("the history holds no ok mark")

    mark_times = [parse_time(m["at"]) for m in marks]
    cut_time = mark_times[-1]
    read_times = [parse_time(r) for r in task.get("reads", []) or []]
    if not read_times:
        read_times = [datetime.fromtimestamp(cut_time.timestamp() + 3600.0 * h, tz=timezone.utc)
                      for h in range(1, 8)]
    horizons = [(rt - cut_time).total_seconds() / 3600.0 for rt in read_times]
    horizons = [h if h > 0 else 0.0 for h in horizons]

    # ---- per-coin series from the marks -----------------------------------
    series = {}
    for mi, m in enumerate(marks):
        t_h = (mark_times[mi] - mark_times[0]).total_seconds() / 3600.0
        for row in m["rows"]:
            mint = row["mint"]
            cap_lamports = int(row["market_cap_lamports"])
            if cap_lamports <= 0:
                continue
            x = math.log10(cap_lamports) - 9.0  # log10 of the cap in SOL
            e = series.get(mint)
            if e is None:
                e = {"t": [], "x": [], "venue": row.get("venue", "curve"),
                     "complete": bool(row.get("complete", False)),
                     "created": row.get("created_at")}
                series[mint] = e
            e["t"].append(t_h)
            e["x"].append(x)
            e["venue"] = row.get("venue", e["venue"])
            e["complete"] = bool(row.get("complete", e["complete"]))

    last_rows = marks[-1]["rows"]
    last_by_mint = {}
    for row in last_rows:
        cap_lamports = int(row["market_cap_lamports"])
        if cap_lamports > 0:
            last_by_mint[row["mint"]] = math.log10(cap_lamports) - 9.0

    def top_of(m, n=5):
        return sorted(m["rows"], key=lambda r: -int(r["market_cap_lamports"]))[:n]

    # ---- RUNTIME CALIBRATION from the input history ------------------------
    # (a) newborn-winner caps: coins that won a top-5 place within 30 min
    # of being born.  Measured across all (cut, read) pairs in the history.
    # The caps rise over the history (the market is growing), so a
    # pooled mean is dragged down by the early, low-cap marks.  Weight
    # each (cut, read) pair by the recency of its cut: the late regime
    # (whose newborn winners sit at 3.31-3.56 dex) is the relevant one
    # for a forecast from the last mark.  The spread is pooled (it is
    # stable) with a floor so a tiny late sample cannot make it collapse.
    nb_caps = []  # (cap, weight)
    for i in range(len(marks)):
        hist_mints = set()
        for m in marks[:i + 1]:
            hist_mints |= {r["mint"] for r in m["rows"]}
        w = (i + 1.0) / float(len(marks))
        for j in range(i + 1, len(marks)):
            if (j - i) * 10 > 30:
                continue
            for r in top_of(marks[j]):
                if r["mint"] not in hist_mints:
                    cap_lamports = int(r["market_cap_lamports"])
                    if cap_lamports > 0:
                        nb_caps.append((math.log10(cap_lamports) - 9.0, w))
    if nb_caps:
        wsum = sum(w for _, w in nb_caps)
        nb_mu_raw = sum(x * w for x, w in nb_caps) / wsum
        mu_pool = _mean([x for x, _ in nb_caps])
        nb_sd_raw = _sd([x for x, _ in nb_caps])
    else:
        nb_mu_raw = PRIOR_NB_MU
        mu_pool = PRIOR_NB_MU
        nb_sd_raw = PRIOR_NB_SD
    # v6: nb_mu comes from the LATE regime only.  v5 pooled every
    # (cut, read) pair with only a weak linear recency weight, so the
    # early marks -- when the market had just opened and newborn
    # winners sat at 1.85-3.0 dex -- dragged nb_mu down to 3.10 on
    # the practice, even though the newborn winners in the LATE regime
    # (the level that matters for a forecast from the last mark) sit
    # at 3.31-3.56 dex (mean 3.41, n=4).  That mis-centred the whole
    # newborn component -- ~37% of the size mixture and the MC's
    # newborn draws -- ~0.3 dex too low, wasting size probability
    # below where the true caps live.  v6 takes nb_mu from the
    # newborn winners of the LAST 3 marks only (the recent regime),
    # shrunk toward the prior so a small late sample cannot swing it
    # far; if fewer than 3 late winners exist it falls back to v5's
    # pooled estimate.  nb_sd stays the pooled value (stable); only
    # the level moves.
    late_caps = []  # newborn-winner caps from cuts in the last 3 marks
    for i in range(max(0, len(marks) - 3), len(marks)):
        hist_mints = set()
        for m in marks[:i + 1]:
            hist_mints |= {r["mint"] for r in m["rows"]}
        for j in range(i + 1, len(marks)):
            if (j - i) * 10 > 30:
                continue
            for r in top_of(marks[j]):
                if r["mint"] not in hist_mints:
                    cap_lamports = int(r["market_cap_lamports"])
                    if cap_lamports > 0:
                        late_caps.append(math.log10(cap_lamports) - 9.0)
    n_nb = len(nb_caps)
    n_late = len(late_caps)
    if n_late >= 3:
        nb_mu = _shrink(_mean(late_caps), PRIOR_NB_MU, n_late, strength=8.0)
    else:
        # fall back to v5's pooled estimate.
        nb_mu = _shrink(nb_mu_raw, PRIOR_NB_MU, n_nb, strength=8.0)
        if n_nb >= 8:
            nb_mu = 0.6 * nb_mu + 0.4 * _shrink(mu_pool, PRIOR_NB_MU, n_nb, strength=8.0)
    nb_sd = _shrink(nb_sd_raw, PRIOR_NB_SD, n_nb, strength=8.0)
    nb_sd = max(0.20, min(0.60, nb_sd))
    nb_mu = max(3.05, min(3.60, nb_mu))

    # (b) strong-newborn rate: DISTINCT coins reaching cap > 3.0 within
    # 30 min of birth, per 10-min interval, over the whole history.  A
    # coin can appear in several marks, so it is counted once (at the
    # first mark where it is both young and strong).
    strong_mints = set()
    for mi, m in enumerate(marks):
        at = mark_times[mi]
        for row in m["rows"]:
            created = row.get("created_at")
            if not created:
                continue
            age_min = (at - parse_time(created)).total_seconds() / 60.0
            if 0.0 <= age_min <= 30.0:
                cap_lamports = int(row["market_cap_lamports"])
                if cap_lamports > 0 and math.log10(cap_lamports) - 9.0 > 3.0:
                    strong_mints.add(row["mint"])
    span_min = max(10.0, (mark_times[-1] - mark_times[0]).total_seconds() / 60.0)
    n_intervals = max(1.0, span_min / 10.0)
    nb_rate = _shrink(len(strong_mints) / n_intervals, PRIOR_NB_RATE, n_intervals, strength=2.0)
    nb_rate = max(0.5, min(4.0, nb_rate))

    # (c) top-5 turnover per 10 min, from the last two transitions (the
    # recent regime is the relevant one; early marks were far more volatile).
    turn_rates = []
    for i in range(max(0, len(marks) - 3), len(marks) - 1):
        prev_top = {r["mint"] for r in top_of(marks[i])}
        cur_top = [r["mint"] for r in top_of(marks[i + 1])]
        changed = sum(1 for mint in cur_top if mint not in prev_top)
        turn_rates.append(changed / 5.0)
    turnover = _shrink(_mean(turn_rates), PRIOR_TURNOVER, len(turn_rates), strength=2.0)
    turnover = max(0.05, min(0.60, turnover))

    # (d) incumbent drift and volatility by venue (10-min log-cap changes).
    vol = {}
    drift = {}
    for venue in ("pool", "curve", "migrating"):
        ds = []
        for i in range(len(marks) - 1):
            a = {r["mint"]: math.log10(int(r["market_cap_lamports"])) - 9.0
                 for r in marks[i]["rows"] if int(r["market_cap_lamports"]) > 0}
            for row in marks[i + 1]["rows"]:
                if row.get("venue") != venue:
                    continue
                mint = row["mint"]
                cap_lamports = int(row["market_cap_lamports"])
                if mint in a and cap_lamports > 0:
                    ds.append(math.log10(cap_lamports) - 9.0 - a[mint])
        prior_vol = PRIOR_VOL_POOL if venue == "pool" else PRIOR_VOL_CURVE
        prior_drift = PRIOR_GROWTH if venue == "pool" else 0.0
        vol[venue] = max(0.05, min(0.60, _shrink(_sd(ds), prior_vol, len(ds), strength=3.0)))
        drift[venue] = max(-0.10, min(0.20, _shrink(_mean(ds), prior_drift, len(ds), strength=3.0)))

    # (e) growth ceiling: the highest cap in the history plus headroom.
    max_cap = max(last_by_mint.values()) if last_by_mint else 3.0
    growth_ceil = max_cap + 0.30

    # ---- candidate selection -------------------------------------------------
    def momentum(e):
        if len(e["x"]) < 2:
            return 0.0
        return e["x"][-1] - e["x"][0]

    scored = []
    for mint, e in series.items():
        x = e["x"][-1]
        scored.append((x + 0.35 * momentum(e), mint))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    cand_mints = [mint for _, mint in scored[:MAX_CANDIDATES]]
    top20 = sorted(last_by_mint.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    for mint, _ in top20:
        if mint not in cand_mints:
            cand_mints.append(mint)
    cand_mints.sort()

    # ---- per-coin model parameters -------------------------------------------
    # Same three regimes as v2 (the data supports them: high-cap pool coins
    # persist and grow, young coins collapse, the rest decay slowly) but the
    # hazard and growth now come from the measured turnover/volatility
    # instead of fixed constants.  A winner's death hazard is the measured
    # turnover scaled down for a high, rising cap; a young coin's hazard is
    # the measured turnover scaled up (young coins are the ones turning
    # over); the rest sit between.
    coins = []
    for mint in cand_mints:
        e = series[mint]
        xs = e["x"]
        ts = e["t"]
        x = xs[-1]
        venue = e["venue"]
        if e["created"]:
            age_min = (cut_time - parse_time(e["created"])).total_seconds() / 60.0
        else:
            age_min = 999.0
        n = len(xs)
        if n >= 2:
            mt = sum(ts) / n
            mx = sum(xs) / n
            num = sum((t - mt) * (xv - mx) for t, xv in zip(ts, xs))
            den = sum((t - mt) ** 2 for t in ts)
            slope_h = num / den if den > 0 else 0.0
        else:
            slope_h = 0.0
        slope_10 = slope_h / 6.0
        # Calibrated volatility: the venue's measured 10-min sd, scaled to
        # an hour.  Pool coins are far more volatile than curve coins.
        vol_hourly = min(0.60, max(0.10, vol.get(venue, 0.20) * math.sqrt(6.0)))
        is_winner = (x >= 3.0 and slope_10 > -0.15) or (n >= 3 and slope_10 > 0.008)
        if is_winner:
            # winners turn over at roughly the measured turnover rate, but a
            # high rising cap protects them: hazard well below turnover.
            hazard = max(0.02, min(0.30, turnover * 0.35))
            regime = "winner"
        elif age_min < 15.0 or n <= 2:
            # young coins are the turnover: hazard near the measured rate.
            hazard = max(0.30, min(0.95, turnover * 3.0))
            if e["complete"]:
                hazard *= 0.8
            regime = "young"
        else:
            hazard = max(0.05, min(0.40, turnover * 0.8))
            regime = "other"
        coins.append({"mint": mint, "regime": regime, "x": x,
                      "hazard": hazard, "vol_hourly": vol_hourly,
                      "drift_h": drift.get(venue, 0.0) * 6.0,
                      "venue": venue})

    # ---- unnamed-mass target -------------------------------------------------
    # u(h): the probability a place is won by a coin born after the cut.
    # Calibrated from the measured turnover: each 10-min step turns over
    # `turnover` of the places, and the newborn share of that turnover is
    # what the history's (cut, read) pairs show; we target the cumulative
    # newborn share, growing with horizon and capped at U_MAX.  At the
    # practice's own last read the share was ~0.40, which this curve
    # reaches by ~40-70 min for the measured turnover.
    def u_target(h_min):
        if h_min <= 0:
            return 0.0
        # The share grows roughly linearly with the number of 10-min
        # steps (each step turns over `turnover` of the places, and in
        # the late regime that turnover is essentially all post-cut
        # newborns), but it SATURATES: once strong post-cut newborns hold
        # the top places they become stable incumbents and the effective
        # turnover falls.  A compounding 1-(1-turnover)^steps reaches
        # 0.79 by 70 min, far above the ~0.40 the practice actually had;
        # the saturation level is ~2x the per-step turnover.  Cap it.
        u = turnover * (h_min / 10.0)
        return min(U_MAX, min(2.0 * turnover, u))

    # ---- Monte Carlo over the reads ------------------------------------------
    rng = random.Random(SEED)
    n_cand = len(coins)
    n_reads = len(horizons)
    place_counts = []   # per read, per place: {mint: count} (named wins)
    size_named = []     # per read, per place: [bin counts] (named wins)
    for ri, h in enumerate(horizons):
        h_min = h * 60.0
        # Strong newborns: the calibrated rate per 10-min cohort since the
        # cut, capped so the field stays bounded.  A read at the cut has
        # none; the count grows with the number of cohorts.
        n_newborn = 0 if h_min < 5 else min(12, int(round(nb_rate * (h_min / 10.0))))
        pc = [dict() for _ in range(places)]
        sc = [[0] * n_bins for _ in range(places)]
        mus = []
        sds = []
        dead_p = []
        for c in coins:
            steps = h * 6.0  # 10-minute steps
            p_dead = 1.0 - (1.0 - c["hazard"]) ** steps if steps > 0 else 0.0
            if c["regime"] == "winner":
                # saturating growth toward the calibrated ceiling.
                mu = growth_ceil - (growth_ceil - c["x"]) * math.exp(-h / 1.0)
                # add the measured drift while far below the ceiling
                mu = min(mu, c["x"] + max(0.0, c["drift_h"]) * min(h, 2.0) + 0.10)
                sd = math.sqrt(SIGMA0 * SIGMA0 + (c["vol_hourly"] ** 2) * h)
            else:
                mu = c["x"] + c["drift_h"] * min(h, 2.0) - 0.08 * h
                sd = math.sqrt(SIGMA0 * SIGMA0 + (c["vol_hourly"] ** 2) * h)
            mus.append(mu)
            sds.append(min(SIGMA_CAP, sd))
            dead_p.append(p_dead)
        for _s in range(SCENARIOS):
            caps = []
            for i in range(n_cand):
                if dead_p[i] > 0.0 and rng.random() < dead_p[i]:
                    caps.append((rng.gauss(DEAD_CAP, DEAD_SD), i))
                else:
                    caps.append((rng.gauss(mus[i], sds[i]), i))
            for _n in range(n_newborn):
                caps.append((rng.gauss(nb_mu, nb_sd), -1))
            caps.sort(key=lambda kv: -kv[0])
            for placed in range(places):
                cap_val, i = caps[placed]
                if i >= 0:
                    pc[placed][coins[i]["mint"]] = pc[placed].get(coins[i]["mint"], 0) + 1
                    b = bin_of(cap_val, edges)
                    sc[placed][b] += 1
        place_counts.append(pc)
        size_named.append(sc)

    # ---- average over reads, build the forecast ------------------------------
    # v5 note: this whole section is v3's, unchanged.  The sizes it builds
    # are the MC histogram + pooled newborn normal (not v4's tight
    # mixture), so v5's sizes are bit-identical to v3's on any input.
    # The only order change from v3 is the extra-naming block below.
    probabilities = []
    sizes = []
    top5 = []
    for k in range(places):
        # Mean MC frequency per named coin at this place, and the MC's own
        # unnamed mass per read (for reference; the target overrides it).
        freq = {}
        mc_unnamed = []
        for r in range(n_reads):
            named_total = sum(place_counts[r][k].values())
            mc_unnamed.append(1.0 - named_total / SCENARIOS)
            for mint, cnt in place_counts[r][k].items():
                freq[mint] = freq.get(mint, 0.0) + cnt / (SCENARIOS * n_reads)
        # Mean unnamed target over the reads (each read has its own
        # horizon, hence its own target).
        u_mean = _mean([u_target(h * 60.0) for h in horizons])
        # Rescale the named frequencies so the unnamed mass equals the
        # target.  The ranking among named coins is unchanged; only the
        # named-vs-newborn split is set by calibration.  If the MC already
        # produces *more* unnamed mass than the target (possible at very
        # long horizons), keep the MC's (do not shrink the unnamed mass).
        s_named = sum(freq.values())
        u_mc_mean = _mean(mc_unnamed)
        # The calibrated target is the reliable evidence at dev-length
        # horizons: it is measured from the turnover, and the Monte
        # Carlo's own unnamed mass overestimates there (at a 10-min read
        # the MC gave 0.35 where the measured target is 0.20, because its
        # newborn field is too strong against the incumbents that still
        # dominate).  So the target wins at short horizons.  Only at very
        # long horizons (mean read > 2 h, as in the final) does the MC's
        # value stand: by then the cut's incumbents are mostly dead, the
        # measured turnover's saturation level is too low, and the MC's
        # death dynamics are the evidence.  Cap the total either way.
        mean_h_min = _mean([h * 60.0 for h in horizons])
        if mean_h_min > 120.0:
            u_final = min(U_MAX, max(u_mean, u_mc_mean))
        else:
            u_final = min(U_MAX, u_mean)
        order = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        named = [mint for mint, _ in order[:MAX_ROWS_PER_PLACE]]
        # v5 (from v4): name the top extra candidates too (by momentum
        # score).  Naming more coins shrinks the pool the unnamed mass is
        # shared over, so a newborn that takes a place scores the unnamed
        # share over fewer coins, and the extra coins carry real (if small)
        # probability from the smoothing instead of 0.  This is the only
        # order change from v3; the MC, the ranking and the unnamed target
        # are unchanged.
        seen_named = set(named)
        for _, mint in scored:
            if len(named) >= MAX_ROWS_PER_PLACE:
                break
            if len(named) >= len(order) + NAMED_EXTRA:
                break
            if mint not in seen_named:
                named.append(mint)
                seen_named.add(mint)
        n_named = len(named)
        for mint in named:
            # Named probabilities sum to (1 - u_final); the smoothing
            # spreads over that named share.  Relative order among named
            # coins is the MC's; only the named-vs-newborn split is set.
            if s_named > 0 and n_named:
                p = (1.0 - u_final) * ((1.0 - SMOOTH_PLACE) * freq.get(mint, 0.0) / s_named
                                       + SMOOTH_PLACE / n_named)
            else:
                p = 0.0
            p = floor6(p)
            if p > 0:
                probabilities.append({"mint": mint, "place": k + 1, "p": p})
        for mint, _ in order:
            if mint not in top5:
                top5.append(mint)
                break
        # ---- sizes: explicit mixture --------------------------------------
        # incumbent component: the MC histogram of winning incumbents'
        # caps at this place.  newborn component: the calibrated newborn
        # cap distribution N(nb_mu, nb_sd).  Mixed by the same unnamed
        # target u_final, then blended with a broad normal prior.
        tot = [0.0] * n_bins
        for r in range(n_reads):
            for b in range(n_bins):
                tot[b] += size_named[r][k][b] / (SCENARIOS * n_reads)
        # tot sums to (1 - u_mc_mean) roughly (named wins only); renormalise
        # it to the incumbent share (1 - u_final).
        tsum = sum(tot)
        if tsum > 0:
            tot = [t / tsum * (1.0 - u_final) for t in tot]
        else:
            tot = [0.0] * n_bins
        # newborn component: analytic normal (nb_mu, nb_sd), weight u_final
        for b in range(n_bins):
            lo = -float("inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == n_bins - 1 else edges[b]
            nb_p = normal_cdf(hi, nb_mu, nb_sd) - normal_cdf(lo, nb_mu, nb_sd)
            tot[b] += u_final * nb_p
        # broad prior, centred on the mixture mean, for robustness
        num = 0.0
        den = 0.0
        for b in range(n_bins):
            lo = -10.0 if b == 0 else edges[b - 1]
            hi = 10.0 if b == n_bins - 1 else edges[b]
            centre = (lo + hi) / 2.0
            num += tot[b] * centre
            den += tot[b]
        mu_size = num / den if den > 0 else nb_mu
        for b in range(n_bins):
            lo = -float("inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == n_bins - 1 else edges[b]
            prior = normal_cdf(hi, mu_size, SIZE_PRIOR_SD) - normal_cdf(lo, mu_size, SIZE_PRIOR_SD)
            p = (1.0 - SIZE_PRIOR_WEIGHT) * tot[b] + SIZE_PRIOR_WEIGHT * prior
            p = floor6(p)
            if p > 0:
                sizes.append({"place": k + 1, "bin": b, "p": p})
    if len(top5) < places:
        for mint in cand_mints:
            if mint not in top5:
                top5.append(mint)
            if len(top5) == places:
                break

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))
    sizes.sort(key=lambda r: (r["place"], r["bin"]))
    doc = {"probabilities": probabilities, "top5": top5[:places], "sizes": sizes}
    text = json.dumps(doc, sort_keys=True) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
