# Phase 1, Step D — Spatial-redistribution test

**Verdict: PARTIAL / MECHANISM DECOMPOSED.** Periodic spatial redistribution —
not the "isolation" semantics — is the dominant driver of the *survival* and
*energy* benefit in the harshest cell. A sham arm that only relocates agents
(no isolation flag, no loneliness signal, no return-to-centre) reproduces
**~2/3 of the extinction-risk reduction** and **~3/4 of the energy gain**. But
sham does **not** reproduce the *fitness* benefit at all — that effect is
isolation-specific. So isolation is largely a vehicle for spatial
redistribution on the outcomes that carry the paradox (survival, energy), with
a residual on fitness that redistribution alone cannot explain.

---

## 1. Question and design

Step C established that food is globally saturated — the standing stock is pinned
at the `max_food = 200` cap in every arm — so this model's scarcity is **local
and search-limited, not global**. In the harshest factorial cell
(`metabolism_discount = 0.0`, `predator_protection = False`) isolation confers
no metabolic or predator subsidy; the only thing it mechanically does is
**periodically teleport a fraction of the swarm to random positions**. Step D
tests whether that relocation is the active ingredient.

Three worlds are run **mirrored on the same seed** (30 seeds, 5 generations ×
1000 steps, N≈50/generation, 30% fraction, relocation every 50 steps for 50
steps):

| Arm | Relocation | `is_isolated` flag / loneliness | Return-to-centre on release |
|-----|-----------|-------------------------------|-----------------------------|
| **control** | none | — | — |
| **isolation** | yes | **yes** | **yes** (teleported back near centre) |
| **sham** | yes (same cadence, same scatter) | no | **no** (agents keep acting from wherever they wandered) |

Sham therefore strips isolation down to **pure periodic relocation**. If sham
reproduces the effect, spatial redistribution is the mechanism and isolation is
merely its vehicle. All persisted per-step quantities were already computed in
`World.step` or cheaply derived from existing state (dispersion = mean pairwise
distance, already computed; local food = mean food in each forager's sensor
range, O(n) over existing observations; eat rate = derived from
`food_eaten_this_step`) — **no new per-step simulation computation was added.**

---

## 2. The spatial mechanism is real and identical across both treated arms

Global standing food stock is indistinguishable between arms — everything is at
the cap — so nothing below is a global food-liberation effect:

| Arm | Global food (mean standing stock) |
|-----|-----------------------------------|
| control | 199.50 |
| isolation | 199.34 |
| sham | 199.25 |

Yet **inside the relocation windows**, both treated arms are measurably less
clumped and sitting on richer local patches than control (paired across seeds,
same-seed contrast, both arms alive):

| Contrast (in-window) | Isolation vs control | Sham vs control |
|----------------------|----------------------|-----------------|
| **Spatial dispersion** (mean pairwise dist.) | +1.33, p = 0.039 (17/30) | +1.74, p = 0.022 (19/30) |
| **Local food** in sensor range | +0.148, p = 0.0036 (22/30) | +0.111, p = 0.0086 (19/30) |
| **Eat rate** / agent in-window | 0.0202 vs 0.0187 (ctrl) | 0.0212 vs 0.0187 (ctrl) |

So the hypothesised chain holds and **does not require the isolation label**:
control clumps → depletes its local patch while the rest of the saturated grid
stays rich → relocation (either arm) scatters foragers onto untouched food →
higher local food, higher eat rate. The trajectory figure (panel 1–2) shows
control's dispersion collapsing late in each generation as survivors converge,
while both treated arms stay dispersed; control's local food sits visibly below
both treated arms throughout.

---

## 3. Which outcome carries the signal — extinction/survival and energy, not fitness

The user's caution was correct: the three outcome families dissociate, so they
must be reported separately.

### 3a. Extinction / survival — the paradox lives here, and sham reproduces most of it

| Arm | Extinctions | Rate | Abs. risk reduction vs control |
|-----|-------------|------|-------------------------------|
| control | 10/30 | 33.3% | — |
| **isolation** | **1/30** | **3.3%** | **−0.300** |
| **sham** | 4/30 | 13.3% | −0.200 |

McNemar exact tests on the discordant (same-seed) pairs:
- **Isolation vs control:** 10 rescues, 1 harm → p = **0.012** (significant).
- **Sham vs control:** 9 rescues, 3 harms → p = 0.146 (same direction, not
  significant at this n).

**Sham reproduces 67% of isolation's extinction-risk reduction (−0.200 of
−0.300).** The paradox that motivated all of Phase 1 — isolation improving
survival in the *harshest* cell — is therefore substantially a
spatial-redistribution effect. Note the population panel: sham (green) actually
sustains the **highest** average population late in each generation, yet still
goes extinct more often than isolation — extinction is a tail event driven by
the worst seeds, and isolation's return-to-centre appears to protect those tails
that sham leaves scattered.

### 3b. Energy — the cleanest mechanistic signature; sham reproduces ~3/4

| Arm | Energy diff vs control | Cohen's d | Paired p | Seeds treat>ctrl |
|-----|------------------------|-----------|----------|------------------|
| **isolation** | **+6.29** | **3.18** | ≈ 0 | 30/30 |
| **sham** | +4.62 | 2.25 | ≈ 0 | 28/30 |

Both arms lift mean energy massively and near-unanimously across seeds — exactly
what "foragers keep landing on richer local patches" predicts. **Sham reproduces
73% of the energy gain (+4.62 of +6.29).** This is the strongest evidence that
the survival benefit is a redistribution effect: energy is up because relocation
puts agents on food, with or without the isolation label.

### 3c. Fitness — isolation-specific; sham reproduces essentially none

| Arm | Fitness impact | Cohen's d | Paired p | Seeds treat>ctrl |
|-----|----------------|-----------|----------|------------------|
| **isolation** | **+0.047** | **+0.78** | **0.0031** | 20/30 |
| **sham** | −0.009 | −0.12 | 0.65 | 15/30 |

Here the arms **diverge**. Isolation produces a moderate, significant fitness
gain; sham produces a null (slightly negative) effect — sham reproduces
**0%** of the fitness benefit. Whatever lifts the evolving fitness score is
present in isolation and absent from pure relocation.

---

## 4. Decomposition — how much of isolation is "just" spatial redistribution

| Outcome | Isolation effect | Sham effect | **Fraction sham reproduces** |
|---------|------------------|-------------|------------------------------|
| Extinction-risk reduction | −0.300 | −0.200 | **67%** |
| Energy gain | +6.29 | +4.62 | **73%** |
| Fitness gain | +0.047 | −0.009 | **~0%** |

**Plainly:** on the outcomes that carry the paradox — staying alive and staying
fed — isolation is **mostly a delivery mechanism for periodic spatial
redistribution**, with sham reproducing roughly two-thirds to three-quarters of
the effect. Isolation is *not* magic here; scattering the swarm off its depleted
clump is most of the story.

---

## 5. The residual — what redistribution alone does not explain

A real residual remains, and it is not noise:

1. **Fitness (0% reproduced).** The evolving fitness score improves only under
   isolation. Sham and isolation differ in exactly two things, so the residual
   must come from one or both: (a) the `is_isolated` flag / loneliness inner
   state that only isolation carries, or (b) the **return-to-centre teleport on
   release** — isolation re-concentrates its relocated agents near the middle of
   the grid, whereas sham leaves them wherever they wandered. Hypothesis (b) is
   the more parsimonious and directly testable: re-concentration near a
   population core may matter for whatever social/reproductive interactions feed
   the fitness metric, independent of local food.

2. **Extinction tail (residual 33%).** Isolation prevents 9/10 control
   extinctions; sham prevents 6/10. The extra tail protection tracks the same
   return-to-centre difference — sham's better *average* population but *worse*
   extinction count suggests that leaving agents permanently scattered helps the
   median seed but hurts the worst seeds, where a re-concentrated core is what
   survives.

Both residual candidates point at the **return-to-centre** operation rather than
the loneliness flag. A clean follow-up (Phase 1e, if pursued) would add a fourth
arm — "sham + return-to-centre" (relocate, no isolation flag, but teleport back
on release) — to separate re-concentration from the loneliness signal. If that
arm closes the fitness/extinction-tail gap, the loneliness semantics can be
discarded entirely and the whole phenomenon reduces to "scatter, then
re-gather."

---

## 6. Bottom line

- The **spatial-redistribution hypothesis is confirmed** as the mechanism for
  the survival and energy benefit: dispersion up, local food up, eat rate up in
  **both** treated arms despite identical global food; sham reproduces ~67% of
  the extinction-risk reduction and ~73% of the energy gain.
- **The signal is carried by extinction/survival and energy, not fitness.** The
  headline paradox (isolation rescuing the harshest cell) is largely a
  redistribution effect and isolation is largely its vehicle.
- **The verdict is not binary.** A genuine isolation-specific residual survives
  on the fitness metric (and on the extinction *tail*), most plausibly
  attributable to the return-to-centre re-concentration rather than the
  loneliness flag — a specific, testable next step.

---

## 7. Limitations

- **Single cell, single fraction.** Harshest cell only (md = 0.0,
  predator_protection = False), 30% fraction, one relocation cadence (50/50).
  The decomposition percentages are specific to this operating point.
- **n = 30 seeds.** Adequate for the paired fitness/energy contrasts (both
  reach p < 0.005) but underpowered for the extinction McNemar on sham
  (p = 0.146 despite a same-direction 9-vs-3 split). The 67% extinction figure
  is a point estimate on a noisy tail statistic — treat it as consistent with,
  not proof of, "most of the effect."
- **Two things distinguish sham from isolation** (loneliness flag AND
  return-to-centre), so the residual cannot yet be attributed to one of them;
  §5 proposes the fourth arm that would.
- **Relocation destinations use an independent RNG stream** so that choosing
  where agents land does not perturb the worlds' own dynamics; the three arms
  are otherwise identical at t = 0 (shared seed, reset ID counter).
- Global-food equality (§2) rules out the Step-C food-liberation account, as
  intended.

---

*Data: `data/results/phase1/spatial_test/checkpoint.jsonl` (30 seeds),
`spatial_analysis.json`. Figures:
`figures/spatial_trajectories.{png,pdf}` (dispersion / local food / population /
energy), `figures/spatial_effect_sizes.{png,pdf}`. Reproduce:
`scripts/run_spatial_test.py` then `scripts/analyze_spatial_test.py`.*
