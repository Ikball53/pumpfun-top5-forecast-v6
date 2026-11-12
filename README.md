# pumpfun-top5-forecast-01 — v6

Forecast the 5 largest pump.fun coins by market cap at each answer read.

## What v6 is: v5 + one fix to the newborn level nb_mu

**v6 = v5 bit-for-bit except for nb_mu.** v5 scored −35.9 on the dev
(order −19.01, size −16.86), rank 194/761; top 10 is −32.8, leader
−31.7. The size gap to top 10 is 2.63 nats, the order gap only
0.47 — size is the bottleneck.

The fix: v5 pooled every (cut, read) newborn-winner pair with only a
weak linear recency weight, so the early marks — when the market had
just opened and newborn winners sat at 1.85–3.0 dex — dragged nb_mu
down to 3.10 on the practice, even though the newborn winners in the
LATE regime (the level that matters for a forecast from the last mark)
sit at 3.31–3.56 dex (mean 3.41, n=4). That mis-centred the whole
newborn component (~37% of the size mixture, and the MC's newborn
draws) ~0.3 dex too low. v6 takes nb_mu from the newborn winners of
the LAST 3 marks only, shrunk toward the prior (strength 8.0); with
fewer than 3 late winners it falls back to v5's pooled estimate. On
the practice nb_mu moves 3.10 → 3.30 and the size distributions shift
up +0.11–0.12 dex at every place with their widths (and peak
probabilities) preserved. nb_sd stays the pooled 0.43.

Why centring, not widening: v4 already diagnosed v3's sizes as too
wide and tried tight ones — they scored 2 nats WORSE because they were
mis-centred. The peak-bin probabilities of v5's size distributions
average to ln −2.71 per place, exactly the leader's achieved size per
place (−13.56/5). So v5's widths are already at the leader's level;
the gap is the true values not landing on our peaks — a centring
problem. Widening would lower the peaks and make it worse.

## What v5 is: a hybrid of v3's sizes and v4's order

**Score breakdown first.** On the dev leaderboard:

| entry | order | size | total |
|---|---|---|---|
| v2 | −21.49 | −18.20 | −39.7 |
| v3 (new) | −19.46 | −16.86 | −36.3 |
| v4 | −18.96 | −18.88 | −37.8 |
| leader (crave) | −18.18 | −13.56 | −31.7 |

v3 is the best so far (−36.3) and v4 is worse (−37.8), but the two
versions fail in opposite halves: **v3's sizes are good and v4's order
is better; v4's sizes are bad and v3's order is weaker.** v5 takes each
version's proven half:

**SIZES: v3's construction, unchanged.** v4 replaced v3's sizes with a
tight per-read normal mixture (sd 0.27–0.38 per place) that scored
−18.88 — 2.02 nats WORSE than v3's −16.86. The tight sizes were a
mistake: v3's wider mixture (the Monte Carlo histogram of winning
incumbents' caps + one pooled newborn normal N(nb_mu, nb_sd), blended
with a broad prior at 10% weight, sd 0.60) matched the dev's actual
spread, while v4's tight normals were too narrow and mis-centred, so
the true caps fell in the tails. v5 keeps v3's size code, constants,
unnamed-mass target, growth ceiling and Monte Carlo EXACTLY as they
are. Verified: on the kit input v5's `sizes` array is **bit-identical
to v3's** (392 rows, same values), and the top5 is identical too. So
v5's size score on the dev should be v3's −16.86.

**ORDER: v4's improvements.** v4 scored −18.96 on order vs v3's −19.46
(+0.50). The improvement is that v4 NAMES MORE COINS per place: the
top 40 momentum candidates beyond the coins that ever win a Monte
Carlo scenario (v3 named only the ~14 MC winners), and it raises the
Laplace smoothing from 0.02 to 0.05 so the extra coins carry real (if
small) probability. Naming more coins shrinks the pool the unnamed
mass is shared over, so a place won by a post-cut newborn scores the
unnamed share over ~46 coins instead of ~86. v5 names 54 coins per
place (the ~14 MC winners + 40 extra), same as v4. The Monte Carlo,
the ranking among named coins and the unnamed-mass target are v3's;
only the named set and the smoothing change. The named probability
sum per place is unchanged from v3 (0.6294 on the kit input), so the
unnamed mass is v3's ~0.37.

**NOT carried over from v4:** the tight sizes, the per-place newborn
caps, the raised unnamed-mass target (U_MAX 0.65, 3× turnover), the
measured top-cap growth rate and the larger growth ceiling. All of
those feed v4's size construction or its Monte Carlo, and v5's sizes
must stay v3's.

**Target (a principled sum, not a promise):** v3's size −16.86 + v4's
order −18.96 = about −35.8, better than v3's −36.3 and v4's −37.8.

## Why this combination and not v4's

v4's order improvement and its size regression came from different
changes, so they can be separated. The order gain is the named-set
expansion (+ smoothing); the size loss is the tight normal mixture.
Nothing in the named-set expansion touches the size code: the sizes
are built from the Monte Carlo histogram and the pooled newborn
normal, never from the named list. So v5 gets v4's order mechanism
without paying v4's size price.

## Local replica score — read the caveat

**The local replica does NOT predict the dev score.** It was already
proven wrong twice: it said v3 would be worse than v2 on every split,
yet v3 scored 3.4 nats better on the real dev; it said v4 was only
slightly worse than v3 on sizes, yet v4 was 2 nats worse on the real
dev. None of v5's numbers are tuned to the replica. The evidence for
v5 is the real-dev facts: v3's sizes scored −16.86 and v4's order
scored −18.96 on the same dev, and v5 reproduces the first exactly
(bit-identical sizes) while reproducing the second's mechanism
(54 named coins per place, smoothing 0.05). The dev score of v5
cannot be predicted locally. If v5 scores worse than v3 on the dev,
it can be reverted to v3 (the best so far, −36.3) by resubmitting the
v3 repo, since Force_Last lets the parent restore it.

## Running it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR` / `$MAKEFUN_OUTPUT_DIR` when set, else
`/input` / `/output`. Writes `forecast.json` (38,664 B on the practice
input, well under the 1 MB limit): exactly 3 keys — `probabilities`
(270 rows: 54 named coins per place), `top5` (5 mints), `sizes` (bins
0–82). Deterministic: one fixed seed, candidates in sorted mint order,
no set iteration, no wall clock, no network. Two runs on the same
history are byte-identical (verified: same sha256).

Runtime ~4 s on the practice input (limit 300 s, 1 CPU, 1024 MB).

## Files

- `main.py` — the entry (Python 3, stdlib only).
- `.makefun/owner.json` — payout wallet (user's own Solana wallet).
- `LICENSE` — MIT.
