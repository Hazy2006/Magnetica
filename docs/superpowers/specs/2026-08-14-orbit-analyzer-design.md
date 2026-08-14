# Orbit Analyzer, Poincaré Section, and Energy Monitor — Design

Date: 2026-08-14
Status: Approved for implementation planning

## Context

`InteractiveScene` (`magnetica/render.py`) currently draws the magnetic
field and animates a particle under a central attractive force
(`Particle.step_attracted`) inside a single `[-5, 5] x [-5, 5]` axes.
Since the particle now bounces elastically off the edges of that box
(`Particle.clamp_to_bounds`), a close pass by a magnet can still send
it onto a numerically-unbound trajectory — it just no longer escapes
the visible frame. There is currently no way to see *that this
happened*, or to inspect the shape/periodicity of the orbit at all.

This adds three live-updating pieces of analysis, all driven from the
same physics already being stepped each tick:

1. An **orbit analyzer** table — six live stats about the current
   orbit's shape and dynamical state.
2. A **Poincaré section** plot — a small scatter plot that
   accumulates one point per orbit "lap."
3. An **energy monitor** plot — a small rolling time-series of
   kinetic, potential, total energy, and relative energy error.

## Goals

- Live, per-tick (or per-lap, where noted) updates of all three
  pieces while `InteractiveScene` runs.
- Keep physics/analysis logic separate from rendering, so it is
  unit-testable without matplotlib or a running event loop.
- New panels must not visually collide with the existing legend,
  status text, or the bottom instructions.

## Non-goals

- Rigorous action-angle / formal Poincaré-map theory. The system
  (multiple, possibly asymmetric, draggable magnets) is not
  integrable in general; the "section" defined below is a practical,
  visually useful simplification, not a claim of exact conserved
  quantities.
- Persisting analysis data across sessions, or exporting it.
- Any new user interaction (the two new panels are read-only displays;
  drag/scroll/add/remove continue to target the main plot only).

## Architecture

New module `magnetica/analysis.py`:

```python
class OrbitAnalyzer:
    def __init__(self, g):
        ...

    def update(self, t, position, velocity, magnets):
        """Advance the analyzer by one physics substep. Called with the
        same t/position/velocity/magnets the particle was just stepped
        to. Internally detects section crossings, updates all running
        metrics, and appends to the Poincaré/energy history buffers."""

    # Read-only snapshot for the current tick, consumed by render.py:
    @property
    def status(self) -> str: ...          # "Bounded" | "Unbound"
    @property
    def period(self) -> float | None: ... # seconds, None until 2 crossings
    @property
    def closure_pct(self) -> float | None: ...
    @property
    def is_convex(self) -> bool | None: ...
    @property
    def crossing_count(self) -> int: ...  # self-intersections in last completed loop
    @property
    def max_radius(self) -> float: ...
    @property
    def poincare_points(self) -> list[tuple[float, float]]: ...  # (r, v_r) per lap
    @property
    def energy_history(self) -> ...  # arrays of t, KE, PE, E, relative error
```

`render.py` owns one `OrbitAnalyzer` instance per `InteractiveScene`,
constructed alongside the particle, and calls `.update(...)` from
inside the existing `PARTICLE_STEPS_PER_FRAME` substep loop in
`_tick`, right after `step_attracted` + `clamp_to_bounds`. All physics
constants (`g`, magnet list) it needs are the same ones already used
for `step_attracted`, so behavior always matches what's actually being
simulated, including live magnet edits (drag/add/remove/rescale).

Energy is computed with the same softened-potential form implied by
`step_attracted`'s force law:

```
U(r) = -g * sum(strength_i / sqrt(r_i^2 + radius_i^2))
KE   = 0.5 * mass * |velocity|^2
E    = KE + U
```

## Metric definitions

**Section crossing.** Reference point = current centroid of
`self.magnets` (recomputed live, so it tracks drags/add/remove). Let
`(rel_x, rel_y) = position - centroid`. A crossing is detected when
`rel_y` changes sign between consecutive substeps while `rel_x > 0`
(i.e., the particle crosses the ray from the centroid through +x, in
either rotational direction). To avoid double-counting jitter near
that ray, a crossing occurring within `MIN_CROSSING_INTERVAL` (0.5s of
simulation time) of the previous one is ignored.

Each crossing stores: simulation time `t`, radius from centroid `r`,
and radial velocity `v_r = velocity · (rel / |rel|)` at the moment of
crossing (linear-interpolated between the two straddling substeps for
a slightly better estimate, or the post-step sample if interpolation
isn't worth the complexity — implementation detail, not a
behavior-visible choice).

**Period.** `t` of the most recent crossing minus `t` of the one
before it. `None` (displayed as `—`) until at least 2 crossings exist.

**Closure %.** Using radius at the two most recent crossings `r_curr`,
`r_prev`:

```
closure = 100 * (1 - |r_curr - r_prev| / max(r_curr, r_prev, eps))
```

`None` (`—`) until at least 2 crossings exist. 100% = the orbit
returns to the same radius every lap (closed ellipse-like orbit); a
lower number means the orbit is precessing/drifting lap to lap.

**Convexity & crossings (self-intersections).** Both computed over the
same window: the position samples collected between the previous two
crossings (i.e., exactly one completed lap). Cleared and restarted
every time a new crossing is detected.

- *Convexity*: walk the lap's polyline, take the cross product of each
  pair of consecutive segment vectors; convex iff all cross products
  share the same sign (allowing near-zero/collinear within a small
  tolerance). `None` (`—`) until one full lap's worth of points
  exists.
- *Crossings*: count of self-intersections within that same lap's
  polyline via pairwise segment-intersection tests — **not** run over
  the entire session's trajectory. Correction from an earlier revision
  of this spec: a real orbit with `PARTICLE_DT = 0.01` and a period of
  several seconds produces 500-1500+ raw substep points per lap, not
  "tens to a couple hundred" — the O(n²) segment-intersection test on
  that many raw points is expensive enough to stall the live UI for
  seconds per lap. `_lap_points` must be explicitly capped (e.g. a
  fixed-size ring buffer, `LAP_POINT_CAP`) independent of how many
  substeps actually occur between crossings.

Both values update once per completed lap, not every tick — they show
the shape of the *last finished lap* until the next one completes.

**Max radius.** Running max of `|position - centroid|` since the
analyzer was created (i.e., since the scene started or was last
reset). Never resets on its own; O(1) per tick.

**Status (Bounded / Unbound).** Sign of `E` computed above, evaluated
every tick (not lap-gated) — `Bounded` if `E < 0`, else `Unbound`.
This is a live, immediate readout, unlike the lap-gated metrics.

## Poincaré section plot

Small axes in the top-left grid cell. Scatter of `(r, v_r)` — one
point appended per section crossing (same crossings that drive period
and closure). Points accumulate up to a cap (last 500) to bound
memory/draw cost; older points drop off. Axes autoscale to the data
with a small margin. Title "Poincaré section", axis labels "r"
and "radial velocity".

A tight cluster/single point reads as a periodic orbit; a closed curve
as quasi-periodic; a scattered cloud as chaotic — standard reading for
this kind of section, useful even without formal guarantees.

## Energy monitor plot

Small axes in the bottom-left grid cell. Rolling window of the last
~30 seconds of simulation time (buffer sized from `dt_per_sample`,
the interval between individual `OrbitAnalyzer.update()` calls —
which is `PARTICLE_DT` alone, since the analyzer is fed once per
physics substep, not once per animation frame. An earlier revision of
this spec said `PARTICLE_DT * PARTICLE_STEPS_PER_FRAME`, which would
undersize the window by `PARTICLE_STEPS_PER_FRAME`x; the implementation
correctly uses `dt_per_sample=PARTICLE_DT`). Three lines — KE, PE, Total E — on
the primary y-axis; a fourth line — relative energy error
`|E(t) - E_0| / |E_0|` (`E_0` = energy at analyzer construction) — on
a **secondary (twin) y-axis in %**, since it lives on a much smaller
scale than KE/PE/E and would be invisible sharing their axis. One
legend covering both axes' lines. Title "Energy monitor", x-label
"time (s)".

## Layout changes to `InteractiveScene._build_figure`

Replace the single `plt.subplots()` with a `GridSpec(2, 2)`:

```
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(
    2, 2, width_ratios=[1, 1.7], height_ratios=[1, 1],
    left=0.06, right=0.97, top=0.93, bottom=0.18,
    wspace=0.30, hspace=0.40,
)
ax_poincare = fig.add_subplot(gs[0, 0])
ax_energy   = fig.add_subplot(gs[1, 0])
ax_main     = fig.add_subplot(gs[:, 1])
```

`ax_main` replaces today's `self.ax` and keeps all existing behavior
(mesh, quiver, scatter, trail, particle dot, legend, status text,
event handling — `on_press`/`on_motion`/`on_scroll` already gate on
`event.inaxes != self.ax`, so clicks on the two new read-only panels
are already correctly ignored once `self.ax` points at `ax_main`).

The bottom margin (`bottom=0.18`) leaves room below the grid for the
existing `fig.text(...)` instructions line so it doesn't collide with
`ax_energy`. The analyzer table is a new monospace text block anchored
at the bottom-left of `ax_main` (axes fraction `(0.02, 0.02)`,
`va='bottom'`), diagonal from the existing status text at
`(0.02, 0.98)` and clear of the legend at `upper right`.

Layout correctness (no overlap) will be checked by rendering an actual
frame to a PNG and visually inspecting it, not just by trusting the
numbers above.

## Testing plan

`tests/test_analysis.py` (headless, no matplotlib):
- Exact synthetic circular motion (fed directly as position/velocity
  samples, not run through `step_attracted`) → period matches the
  known angular velocity, closure ≈ 100%, convex = True, crossings =
  0.
- A synthetic self-intersecting star-shaped polyline → crossings > 0,
  convex = False.
- A slow, near particle vs. a fast, far particle around a single
  magnet → Bounded vs. Unbound status matches sign of hand-computed
  energy.
- Max radius tracks a monotonically-growing synthetic path correctly.

Integration (Agg backend, same style as the earlier wall-bounce
verification):
- Drive `InteractiveScene._tick` for many iterations; assert no
  exceptions, and that Poincaré/energy buffers grow as expected.
- Render one frame to PNG after a run and visually inspect it (read
  the image) to confirm the table/legend/instructions/new panels don't
  overlap.

## Open implementation details (not behavior-affecting, left to
implementation time)

- Exact interpolation (or lack thereof) of `v_r`/`r` at the crossing
  instant vs. using the post-crossing substep's values directly.
- Exact color/line-style choices for the 4 energy-monitor series and
  the Poincaré scatter, following the existing dark-background,
  high-contrast style already used in `render.py`.
