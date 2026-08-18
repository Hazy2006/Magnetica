# Magnetica: Tier 0/1 correctness and honesty fixes

Source: joint audit document (Tier 0-4), 2026-08-17.

## Context

Magnetica simulates a particle under two interchangeable physics models
(attraction: central-force stand-in, Verlet; Lorentz: v×B magnetic force,
semi-implicit Euler) but the field arrows shown on screen match neither —
they always render the dipole field. The analyzer also always reports
central-force PE/status, meaningless in Lorentz mode. The Lorentz
integrator has a measured, uncorrected energy leak. This spec fixes the
display/physics mismatch, the analyzer's mode-blindness, the integrator
leak, and two verified Tier 1 bugs, in the priority order from the audit
doc's "suggested debugging order."

Decisions already made with the user:
- Display arrows must match whichever model actually drives the particle
  (audit doc option 0.1a), not stay illustrative-only.
- Default mode stays attraction; both models get documented clearly
  rather than switching the default (0.2).
- Convexity metric is kept.
- Tests added are scoped to validating the changes in this spec, not a
  general coverage push (no `test_field.py` recreation, no conftest
  figure-cleanup fixture — both are separable follow-ups, not blockers).

## A. Mode-aware field display

`compute_field_arrows` (render.py) always calls `compute_total_field`
(field.py), the in-plane dipole field, regardless of mode. This is
replaced with a per-mode field:

- **Attraction mode**: new `compute_attraction_field(magnets, x, y, g)`
  in `field.py`, vectorizing the same formula
  `Particle._attraction_acceleration` already uses per-particle
  (`g·strength·offset / (r²+radius²)^1.5`, summed over magnets) across
  the grid. Quiver arrows show this field's direction.
- **Lorentz mode**: `bz` (`compute_bz`) is a scalar, out-of-plane field
  with no in-plane direction. The quiver is hidden in this mode; only
  the `bz`-magnitude heatmap is shown, with a colorbar/label clarifying
  it's `B_z` (out of plane).

`render.py`'s `_tick`/`_build_figure` branch on `self.mode` to pick the
field function and toggle quiver visibility.

## B. 1.4 fix: self-cross detection across lap boundaries

`OrbitAnalyzer._on_crossing` (analysis.py:171-174) currently resets
`_lap_points` to a single seed point every time a lap completes. If a
self-intersection's two segments straddle that reset boundary,
`analyze_lap` never evaluates both segments together and reports zero
crossings despite a visible self-intersection (e.g. a figure-eight).

Fix: stop resetting `_lap_points`. It becomes a plain rolling
`deque(maxlen=LAP_POINT_CAP)` that is only ever appended to (never
manually cleared); `analyze_lap` runs on this rolling window at every
crossing, same call site, same O(n²) cost bound. `is_convex`/
`crossing_count` now reflect self-intersections in the trailing
`LAP_POINT_CAP` points rather than strictly "since the last lap
boundary" — behavior is unchanged for laps already longer than
`LAP_POINT_CAP` (points already dropped silently today) and only
changes (fixes) laps shorter than the cap.

1.2 (escape semantics) required no fix — verified already box-bounds
(`_out_of_bounds` checks `xlim`/`ylim` independently), matching the
visible domain. 2.1 (trail `pop(0)`) required no fix — already
`deque(maxlen=TRAIL_LENGTH)`.

## C. Tests

1. Lorentz speed-invariance test for `particle.step()` — tight
   tolerance (e.g. `pytest.approx(initial_speed, rel=1e-8)`) held
   across many steps. Proves the Boris/rotation rewrite (section G)
   actually removes the energy leak; the current test only checks
   speed doesn't blow past 10x, which the leak already satisfies.
2. Test for `compute_attraction_field` (section A) — verifies direction
   points toward the magnet and magnitude matches
   `_attraction_acceleration`.

## D. 0.3 fix: mode-aware analyzer

`OrbitAnalyzer` gets a `mode` parameter, defaulting to attraction mode
so existing test call sites (`OrbitAnalyzer(g=..., dt_per_sample=...)`)
keep working unchanged. `InteractiveScene` passes its existing
`self.mode` explicitly (it already rebuilds the analyzer on mode
toggle via `_reset_analyzer` — no new plumbing needed there). In
Lorentz mode:

- Skip `_potential_energy` and the `e = ke + pe` / status sign-check;
  `pe`, `e`, `status` become `None`.
- `energy_history` still records `ke`/`t` (should be flat post-Boris —
  a visible confirmation the fix works); `pe`/`e`/`rel_err` stay empty.
- Geometric metrics (`max_radius`, `period`, `closure_pct`, `is_convex`,
  `crossing_count`, Poincaré/lap data) are mode-agnostic and unaffected.

`render.py`'s `_analyzer_table_text()` and the energy subplot hide the
PE/Total E/Status rows/lines in Lorentz mode, showing only KE (labeled
to explain why it's flat) plus the geometric metrics.

## E. 1.3: Lorentz tuning experiment

Before touching escape logic: experiment (uncommitted script) with
initial conditions for a bounded/orbiting Lorentz trajectory — slower
launch speed, starting deeper in the field, stronger magnet moments.
Doing this after section G (Boris) matters: if part of today's
near-immediate escape is the energy leak inflating speed, fixing the
leak first may make tuning easier or unnecessary. Report findings
(a working bounded example, or confirmation that escape is the
expected physical outcome and enlarging the domain is the honest fix)
rather than presupposing an answer.

## F. 2.2: dirty-flag field caching

`self._field_dirty = True` set on: magnet drag (`on_motion`), scroll/
strength change (`on_scroll`), add/remove (`_try_add_magnet`, right-
click delete in `on_press`), mode toggle. `_tick` recomputes the field
grid (whichever function section A selects) and `strength_norm`'s
percentile together only when the flag is set, then clears it.

## G. 1.1 fix: Boris pusher (exact rotation for out-of-plane B)

`particle.step()` currently does semi-implicit Euler on `v×B`, which
leaks energy (measured ~1.0015x per step) because the force depends on
velocity but is evaluated from last step's velocity before an ordinary
linear update. Since `bz` is purely out-of-plane, the true motion is an
*exact* rotation of velocity at rate `ω = charge·bz/mass`. Fix: rotate
`(vx, vy)` by `θ = -ω·dt` via a 2×2 rotation matrix, then advance
position with the rotated velocity. This preserves `|v|` to float
precision regardless of step size. This is the out-of-plane-B
specialization of the general Boris pusher — no new class or
abstraction, one method's update logic changes.

## H. Documentation/UX

- Fix the stale overlap error message (`field.py:15`, "please change
  coordinates in the code") — wrong since magnets are draggable. New
  text points at dragging instead.
- `step()`'s docstring documents why the rotation approach suits a
  velocity-dependent force, mirroring `step_attracted`'s existing
  rationale docstring for Verlet.
- README updates reflecting 0.1/0.2: both models exist, attraction is
  a stand-in force (not real magnetism), arrows match whichever model
  is active.
- Convexity metric (`_cross`/`segments_intersect`/`analyze_lap` →
  `is_convex`) is kept as-is.
- Divide-by-zero guard comment (`field.py:30`) stays untouched.

## Out of scope for this pass

Recreating `tests/test_field.py` and adding a `conftest.py` figure-
cleanup fixture were both considered and explicitly deferred — neither
is required to validate the fixes above.
