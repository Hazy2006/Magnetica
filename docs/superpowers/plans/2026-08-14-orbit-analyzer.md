# Orbit Analyzer, Poincaré Section, and Energy Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live orbit-analyzer stats table, a Poincaré-section scatter plot, and an energy-monitor line plot to the existing interactive scene, all driven from the particle physics already being stepped each tick.

**Architecture:** A new, matplotlib-free `OrbitAnalyzer` class (`magnetica/analysis.py`) tracks section crossings, period, closure, convexity, self-intersections, max radius, bounded/unbound status, Poincaré points, and a rolling energy history — fed one physics substep at a time. `InteractiveScene` (`magnetica/render.py`) owns one `OrbitAnalyzer` instance, restructures its single-axes figure into a `GridSpec` (main plot on the right, Poincaré + energy panels stacked on the left), and reads the analyzer's public properties each tick to update a text table and two plots.

**Tech Stack:** Python, NumPy, Matplotlib (`GridSpec`, `Agg` backend for headless tests), pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-orbit-analyzer-design.md`

## Global Constraints

- Analysis logic (`magnetica/analysis.py`) has no matplotlib dependency and is unit-testable standalone.
- The Poincaré and energy panels are read-only; existing drag/add/remove/scroll interactions continue to target only the main plot axes.
- No persistence or export of analysis data — in-memory only, reset when the scene is recreated.
- Convexity and self-intersection count (`crossing_count`) are computed **only over the most recently completed lap** (points since the previous section crossing), never over the full session history.
- Section-crossing reference = the ray from the magnets' current centroid through +x, recomputed live each tick (so it tracks dragged/added/removed magnets).
- Crossing debounce: ignore a new crossing within 0.5s of simulation time of the previous one (`MIN_CROSSING_INTERVAL = 0.5`).
- Poincaré point buffer capped at the most recent 500 points.
- Energy history is a rolling window of the last ~30 seconds of simulation time (`ENERGY_WINDOW_SECONDS = 30.0`).
- `g` passed into the analyzer must always be `ATTRACTION_STRENGTH`, the same constant already used for `step_attracted`, so displayed energy always matches what's actually being simulated.

---

### Task 1: `OrbitAnalyzer` core — energy, status, max radius

**Files:**
- Create: `magnetica/analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Produces: `OrbitAnalyzer(g, energy_window_seconds=30.0, dt_per_sample=None)`; `.update(t, position, velocity, magnets)`; read-only properties `.status -> str`, `.max_radius -> float`, `.period -> float | None`, `.closure_pct -> float | None`, `.is_convex -> bool | None`, `.crossing_count -> int`, `.poincare_points -> list[tuple[float, float]]`, `.energy_history -> dict[str, list[float]]` (keys `t`, `ke`, `pe`, `e`, `rel_err`). This task implements `status`, `max_radius`, and stubs the rest to their "no data yet" defaults (`None`/`0`/`[]`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analysis.py`:

```python
import pytest

from magnetica.analysis import OrbitAnalyzer
from magnetica.field import Magnet


def test_status_is_bounded_for_slow_nearby_particle():
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0)
    analyzer.update(0.0, position=(2.0, 0.0), velocity=(0.0, 0.0), magnets=magnets)
    assert analyzer.status == "Bounded"


def test_status_is_unbound_for_fast_far_particle():
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0)
    analyzer.update(0.0, position=(2.0, 0.0), velocity=(10.0, 0.0), magnets=magnets)
    assert analyzer.status == "Unbound"


def test_max_radius_tracks_running_maximum_from_centroid():
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0)
    analyzer.update(0.0, position=(1.0, 0.0), velocity=(0.0, 0.0), magnets=magnets)
    analyzer.update(0.1, position=(3.0, 0.0), velocity=(0.0, 0.0), magnets=magnets)
    analyzer.update(0.2, position=(2.0, 0.0), velocity=(0.0, 0.0), magnets=magnets)
    assert analyzer.max_radius == pytest.approx(3.0)


def test_properties_default_before_any_crossing():
    analyzer = OrbitAnalyzer(g=1.0)
    assert analyzer.period is None
    assert analyzer.closure_pct is None
    assert analyzer.is_convex is None
    assert analyzer.crossing_count == 0
    assert analyzer.poincare_points == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'magnetica.analysis'`

- [ ] **Step 3: Implement `magnetica/analysis.py`**

```python
import numpy as np
from collections import deque

MIN_CROSSING_INTERVAL = 0.5   # seconds of simulation time
POINCARE_CAP = 500
ENERGY_WINDOW_SECONDS = 30.0


def _kinetic_energy(velocity, mass=1.0):
    velocity = np.asarray(velocity, dtype=float)
    return 0.5 * mass * float(np.dot(velocity, velocity))


def _potential_energy(position, magnets, g):
    position = np.asarray(position, dtype=float)
    pe = 0.0
    for magnet in magnets:
        r = float(np.linalg.norm(magnet.position - position))
        strength = float(np.linalg.norm(magnet.moment))
        pe += -g * strength / np.sqrt(r ** 2 + magnet.radius ** 2)
    return pe


def _magnets_centroid(magnets):
    if not magnets:
        return np.zeros(2)
    return np.mean([m.position for m in magnets], axis=0)


class OrbitAnalyzer:
    """Tracks live orbit-shape and energy metrics from a stream of
    (t, position, velocity, magnets) physics substeps. Has no matplotlib
    dependency so it can be tested and reasoned about on its own."""

    def __init__(self, g, energy_window_seconds=ENERGY_WINDOW_SECONDS, dt_per_sample=None):
        self.g = g
        self._max_radius = 0.0
        self._status = "Bounded"

        self._prev_rel = None
        self._last_crossing_t = None
        self._crossing_times = deque(maxlen=2)
        self._crossing_radii = deque(maxlen=2)
        self._period = None
        self._closure_pct = None

        self._poincare_points = deque(maxlen=POINCARE_CAP)

        self._lap_points = []
        self._is_convex = None
        self._crossing_count = 0

        maxlen = None
        if dt_per_sample:
            maxlen = max(int(energy_window_seconds / dt_per_sample), 1)
        self._t_hist = deque(maxlen=maxlen)
        self._ke_hist = deque(maxlen=maxlen)
        self._pe_hist = deque(maxlen=maxlen)
        self._e_hist = deque(maxlen=maxlen)
        self._rel_err_hist = deque(maxlen=maxlen)
        self._e0 = None

    def update(self, t, position, velocity, magnets):
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)

        ke = _kinetic_energy(velocity)
        pe = _potential_energy(position, magnets, self.g)
        e = ke + pe
        if self._e0 is None:
            self._e0 = e
        rel_err = abs(e - self._e0) / abs(self._e0) if self._e0 != 0 else 0.0

        self._t_hist.append(t)
        self._ke_hist.append(ke)
        self._pe_hist.append(pe)
        self._e_hist.append(e)
        self._rel_err_hist.append(rel_err)

        self._status = "Bounded" if e < 0 else "Unbound"

        centroid = _magnets_centroid(magnets)
        rel = position - centroid
        r = float(np.linalg.norm(rel))
        self._max_radius = max(self._max_radius, r)

    @property
    def status(self):
        return self._status

    @property
    def max_radius(self):
        return self._max_radius

    @property
    def period(self):
        return self._period

    @property
    def closure_pct(self):
        return self._closure_pct

    @property
    def is_convex(self):
        return self._is_convex

    @property
    def crossing_count(self):
        return self._crossing_count

    @property
    def poincare_points(self):
        return list(self._poincare_points)

    @property
    def energy_history(self):
        return {
            "t": list(self._t_hist),
            "ke": list(self._ke_hist),
            "pe": list(self._pe_hist),
            "e": list(self._e_hist),
            "rel_err": list(self._rel_err_hist),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add magnetica/analysis.py tests/test_analysis.py
git commit -m "feat: add OrbitAnalyzer core (energy, status, max radius)"
```

---

### Task 2: Section crossings, period, closure, Poincaré points

**Files:**
- Modify: `magnetica/analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: `OrbitAnalyzer.update` and private state from Task 1 (`self._prev_rel`, `self._crossing_times`, `self._crossing_radii`, `self._poincare_points`).
- Produces: working `.period`, `.closure_pct`, `.poincare_points` (each a `(r, v_r)` tuple); a new private `_on_crossing(t, r, rel, velocity)` that Task 3 will extend.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis.py`:

```python
import numpy as np


def test_period_and_closure_for_exact_circular_motion():
    R = 2.0
    T = 4.0
    w = 2 * np.pi / T
    dt = 0.01
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=dt)

    # Start at angle = pi (well away from the crossing ray at angle 0)
    # and run for just over 2 full periods.
    steps = int(2.2 * T / dt)
    for i in range(steps):
        t = i * dt
        theta = np.pi + w * t
        position = (R * np.cos(theta), R * np.sin(theta))
        velocity = (-R * w * np.sin(theta), R * w * np.cos(theta))
        analyzer.update(t, position, velocity, magnets=[])

    assert analyzer.period is not None
    assert abs(analyzer.period - T) < 0.05
    assert analyzer.closure_pct is not None
    assert analyzer.closure_pct > 99.0
    assert len(analyzer.poincare_points) == 2


def test_no_crossing_within_debounce_interval():
    analyzer = OrbitAnalyzer(g=1.0)
    # Two samples that cross the ray twice within 0.1s of each other
    # (well under the 0.5s debounce) should only register once.
    analyzer.update(0.0, position=(1.0, -0.01), velocity=(0.0, 0.0), magnets=[])
    analyzer.update(0.05, position=(1.0, 0.01), velocity=(0.0, 0.0), magnets=[])
    analyzer.update(0.1, position=(1.0, -0.01), velocity=(0.0, 0.0), magnets=[])
    assert len(analyzer.poincare_points) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis.py -v -k crossing or circular`
Expected: FAIL (`period` stays `None`, `poincare_points` stays empty — no crossing detection implemented yet)

- [ ] **Step 3: Implement crossing detection**

In `magnetica/analysis.py`, replace the `update` method's centroid/max-radius block (from `centroid = _magnets_centroid(magnets)` to the end of `update`) with:

```python
        centroid = _magnets_centroid(magnets)
        rel = position - centroid
        r = float(np.linalg.norm(rel))
        self._max_radius = max(self._max_radius, r)

        if self._prev_rel is not None:
            prev_y = self._prev_rel[1]
            sign_changed = (prev_y <= 0 < rel[1]) or (prev_y >= 0 > rel[1])
            if sign_changed and rel[0] > 0:
                debounced = (
                    self._last_crossing_t is not None
                    and t - self._last_crossing_t < MIN_CROSSING_INTERVAL
                )
                if not debounced:
                    self._on_crossing(t, r, rel, velocity)
        self._prev_rel = rel
        self._lap_points.append(position.copy())

    def _on_crossing(self, t, r, rel, velocity):
        self._last_crossing_t = t
        self._crossing_times.append(t)
        self._crossing_radii.append(r)

        r_hat = rel / r if r > 0 else np.zeros(2)
        v_r = float(np.dot(velocity, r_hat))
        self._poincare_points.append((r, v_r))

        if len(self._crossing_times) == 2:
            self._period = self._crossing_times[1] - self._crossing_times[0]
            r_prev, r_curr = self._crossing_radii
            denom = max(r_prev, r_curr, 1e-9)
            self._closure_pct = 100.0 * (1.0 - abs(r_curr - r_prev) / denom)

        self._lap_points = [self._lap_points[-1]] if self._lap_points else []
```

Note: `self._lap_points.append(position.copy())` runs on every `update` call, after the crossing check, as shown above. When a crossing fires, `_on_crossing` first trims `_lap_points` down to just its last point (the end of the completed lap), then this line appends the new point (the start of the next lap) — so laps always begin seeded with the point where the previous one ended. This collects the points Task 3 analyzes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add magnetica/analysis.py tests/test_analysis.py
git commit -m "feat: add section-crossing detection, period, and closure to OrbitAnalyzer"
```

---

### Task 3: Lap convexity and self-intersection count

**Files:**
- Modify: `magnetica/analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Produces: module-level `segments_intersect(p1, p2, p3, p4) -> bool` and `analyze_lap(points) -> tuple[bool | None, int]` (is_convex, self-intersection count). `OrbitAnalyzer._on_crossing` calls `analyze_lap` on `self._lap_points` and stores the result in `self._is_convex` / `self._crossing_count`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
from magnetica.analysis import segments_intersect, analyze_lap


def test_segments_intersect_detects_crossing_x():
    assert segments_intersect((0, 0), (1, 1), (0, 1), (1, 0)) is True


def test_segments_intersect_ignores_parallel_segments():
    assert segments_intersect((0, 0), (1, 0), (0, 1), (1, 1)) is False


def test_analyze_lap_convex_circle_has_no_self_intersections():
    n = 40
    points = [(np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i in range(n)]
    is_convex, crossings = analyze_lap(points)
    assert is_convex is True
    assert crossings == 0


def test_analyze_lap_star_is_not_convex_and_self_intersects():
    # A pentagram traced through every second vertex of a regular pentagon.
    # Note: this turns the same rotational direction at every vertex (just
    # a wider angle than a simple pentagon), so a naive "same-sign turns"
    # check alone would wrongly call it convex -- it must also fail because
    # it self-intersects. That's exactly what this test guards.
    n = 5
    outer = [
        (np.cos(2 * np.pi * i / n - np.pi / 2), np.sin(2 * np.pi * i / n - np.pi / 2))
        for i in range(n)
    ]
    star_order = [0, 2, 4, 1, 3]
    points = [outer[i] for i in star_order]
    is_convex, crossings = analyze_lap(points)
    assert crossings > 0
    assert is_convex is False


def test_analyze_lap_returns_none_for_too_few_points():
    is_convex, crossings = analyze_lap([(0, 0), (1, 1)])
    assert is_convex is None
    assert crossings == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis.py -v -k segments_intersect or analyze_lap`
Expected: FAIL with `ImportError: cannot import name 'segments_intersect'`

- [ ] **Step 3: Implement `segments_intersect` and `analyze_lap`**

Add to `magnetica/analysis.py`, above the `OrbitAnalyzer` class:

```python
def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1, p2, p3, p4):
    """True if segment p1-p2 properly crosses segment p3-p4."""
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def analyze_lap(points, convex_tolerance=1e-9):
    """Analyze one completed orbit lap's worth of points: whether the
    polyline is convex and how many times it crosses itself. Returns
    (None, 0) if there are too few points to judge (fewer than 3).

    A star/rosette shape can turn the same rotational direction at every
    vertex (just a wider angle than a simple convex shape), so "all turns
    the same sign" alone is not sufficient to call something convex --
    it must also not cross itself."""
    if len(points) < 3:
        return None, 0

    signs = set()
    for i in range(len(points) - 2):
        c = _cross(points[i], points[i + 1], points[i + 2])
        if abs(c) > convex_tolerance:
            signs.add(c > 0)
    consistent_turning = len(signs) <= 1

    crossings = 0
    segment_count = len(points) - 1
    for i in range(segment_count):
        for j in range(i + 2, segment_count):
            if segments_intersect(points[i], points[i + 1], points[j], points[j + 1]):
                crossings += 1

    is_convex = consistent_turning and crossings == 0
    return is_convex, crossings
```

Then update `OrbitAnalyzer._on_crossing` (written in Task 2) to call it — replace the last line
(`self._lap_points = [self._lap_points[-1]] if self._lap_points else []`) with:

```python
        self._is_convex, self._crossing_count = analyze_lap(self._lap_points)
        self._lap_points = [self._lap_points[-1]] if self._lap_points else []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add magnetica/analysis.py tests/test_analysis.py
git commit -m "feat: add lap convexity and self-intersection analysis to OrbitAnalyzer"
```

---

### Task 4: Rolling energy history window

**Files:**
- Modify: `tests/test_analysis.py` (implementation already exists from Task 1 — this task is test-only, verifying the `dt_per_sample`/window behavior)
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: `OrbitAnalyzer(g, dt_per_sample=...)` and `.energy_history` from Task 1.

- [ ] **Step 1: Write the failing tests**

`tests/test_analysis.py` already has `import pytest` (Task 1) and
`from magnetica.field import Magnet` (Task 4's own test below reuses it).
Append to `tests/test_analysis.py`:

```python
def test_energy_history_rolls_off_old_samples_outside_window():
    dt = 0.1
    window = 1.0
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=dt, energy_window_seconds=window)
    for i in range(30):
        t = i * dt
        analyzer.update(t, position=(1.0, 0.0), velocity=(0.0, 0.0), magnets=[])
    hist = analyzer.energy_history
    assert len(hist["t"]) == 10
    assert hist["t"][0] == pytest.approx(2.0)
    assert hist["t"][-1] == pytest.approx(2.9)


def test_relative_energy_error_matches_formula():
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=0.1)

    analyzer.update(0.0, position=(2.0, 0.0), velocity=(0.0, 1.0), magnets=magnets)
    e0 = analyzer.energy_history["e"][-1]

    analyzer.update(0.1, position=(2.0, 0.0), velocity=(0.0, 2.0), magnets=magnets)
    e1 = analyzer.energy_history["e"][-1]
    expected_rel_err = abs(e1 - e0) / abs(e0)

    assert analyzer.energy_history["rel_err"][-1] == pytest.approx(expected_rel_err)
```

- [ ] **Step 2: Run tests to verify current behavior**

Run: `pytest tests/test_analysis.py -v -k energy_history or relative_energy`
Expected: PASS — this task validates behavior already implemented in Task 1 (the `deque(maxlen=...)` sizing and `rel_err` formula). If either test fails, the bug is in the `maxlen` calculation or the `rel_err` line in `update` from Task 1 — fix `magnetica/analysis.py` until both pass.

- [ ] **Step 3: N/A**

No new implementation expected; this task exists to lock the rolling-window behavior in with explicit tests before `render.py` starts depending on it.

- [ ] **Step 4: Run full analysis test suite**

Run: `pytest tests/test_analysis.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_analysis.py
git commit -m "test: lock down OrbitAnalyzer rolling energy-history window behavior"
```

---

### Task 5: Wire `OrbitAnalyzer` into `InteractiveScene` (layout skeleton + data flow)

**Files:**
- Modify: `magnetica/render.py`
- Test: `tests/test_render_integration.py` (new)

**Interfaces:**
- Consumes: `OrbitAnalyzer` from Task 1-4 (`magnetica.analysis.OrbitAnalyzer`).
- Produces: `InteractiveScene.analyzer` (an `OrbitAnalyzer` instance, updated every substep); `InteractiveScene.ax_main`, `.ax_poincare`, `.ax_energy` (new axes, `ax_poincare`/`ax_energy` empty of artists until Task 6); `InteractiveScene.sim_time` (float, simulation seconds elapsed).

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_integration.py`:

```python
import matplotlib
matplotlib.use("Agg")

import numpy as np

from magnetica.field import Magnet
from magnetica.render import InteractiveScene


def test_scene_has_three_axes_and_an_analyzer():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    assert scene.ax_main is not None
    assert scene.ax_poincare is not None
    assert scene.ax_energy is not None
    assert scene.analyzer is not None


def test_scene_ticks_without_error_and_analyzer_updates():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")

    for i in range(200):
        scene._tick(i)

    assert scene.analyzer.status in ("Bounded", "Unbound")
    assert scene.analyzer.max_radius > 0
    hist = scene.analyzer.energy_history
    assert len(hist["t"]) > 0


def test_scene_particle_still_respects_wall_bounds_with_analyzer_running():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")
    scene.particle.position = np.array([0.5, 2.0])
    scene.particle.velocity = np.array([0.0, -1.0])

    for i in range(400):
        scene._tick(i)
        assert -5 <= scene.particle.position[0] <= 5
        assert -5 <= scene.particle.position[1] <= 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_render_integration.py -v`
Expected: FAIL with `AttributeError: 'InteractiveScene' object has no attribute 'ax_main'`

- [ ] **Step 3: Add the import and analyzer wiring**

In `magnetica/render.py`, add the import alongside the existing ones (after line 17, `from magnetica.particle import Particle`):

```python
from magnetica.analysis import OrbitAnalyzer
```

Replace the `__init__` body from `start_position, start_velocity = default_particle_start(magnets)` (line 104) through `self.trail_y = []` (line 109) with:

```python
        start_position, start_velocity = default_particle_start(magnets)
        self.particle = Particle(position=start_position,
                                  velocity=start_velocity,
                                  charge=PARTICLE_CHARGE)
        self.trail_x = []
        self.trail_y = []
        self.sim_time = 0.0
        self.analyzer = OrbitAnalyzer(g=ATTRACTION_STRENGTH, dt_per_sample=PARTICLE_DT)
```

- [ ] **Step 4: Restructure `_build_figure` into a GridSpec layout**

Replace the entire `_build_figure` method (lines 117-149) with:

```python
    def _build_figure(self, title):
        bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y)

        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(
            2, 2, width_ratios=[1, 1.7], height_ratios=[1, 1],
            left=0.06, right=0.97, top=0.93, bottom=0.18,
            wspace=0.30, hspace=0.40,
        )
        self.ax_poincare = self.fig.add_subplot(gs[0, 0])
        self.ax_energy = self.fig.add_subplot(gs[1, 0])
        self.ax_main = self.fig.add_subplot(gs[:, 1])

        self.mesh = self.ax_main.pcolormesh(self.x, self.y, magnitude, cmap='inferno',
                                             shading='gouraud', norm=strength_norm(magnitude), zorder=0)
        self.quiver = self.ax_main.quiver(self.x, self.y, bx_norm, by_norm,
                                           color='white', alpha=0.8, zorder=2)
        self.scatter = self.ax_main.scatter(*magnet_positions(self.magnets).T,
                                             c='red', s=[magnet_marker_size(m) for m in self.magnets],
                                             edgecolors='black', zorder=5, picker=True)

        self.trail_line, = self.ax_main.plot([], [], color='deepskyblue', alpha=0.6, zorder=4)
        self.particle_dot, = self.ax_main.plot([], [], 'go', markersize=8, zorder=6)

        self.fig.colorbar(self.mesh, ax=self.ax_main, label='field strength')
        self.ax_main.set_xlim(*self.xlim)
        self.ax_main.set_ylim(*self.ylim)
        self.ax_main.set_xlabel('x')
        self.ax_main.set_ylabel('y')
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title(title)

        self.ax_main.legend([self.scatter, self.particle_dot, self.trail_line],
                             ['magnet', 'particle', 'trail'],
                             loc='upper right', fontsize=8, framealpha=0.7)

        self.status_text = self.ax_main.text(
            0.02, 0.98, '', transform=self.ax_main.transAxes, va='top', ha='left',
            fontsize=9, color='yellow', bbox=dict(facecolor='black', alpha=0.5, pad=3))

        self.fig.text(0.5, 0.02, INSTRUCTIONS, ha='center', va='bottom', fontsize=9)
```

(Task 6 appends Poincaré/energy artist setup to the end of this method — leave `ax_poincare`/`ax_energy` bare here.)

- [ ] **Step 5: Rename remaining `self.ax` references to `self.ax_main`**

In `on_press` (line 194), `on_motion` (line 221), and `on_scroll` (line 231), change every `event.inaxes != self.ax` to `event.inaxes != self.ax_main`.

- [ ] **Step 6: Wire the analyzer into the tick loop**

Replace the substep loop in `_tick` (lines 266-268):

```python
        for _ in range(PARTICLE_STEPS_PER_FRAME):
            self.particle.step_attracted(self.magnets, PARTICLE_DT, g=ATTRACTION_STRENGTH)
            self.particle.clamp_to_bounds(self.xlim, self.ylim)
```

with:

```python
        for _ in range(PARTICLE_STEPS_PER_FRAME):
            self.particle.step_attracted(self.magnets, PARTICLE_DT, g=ATTRACTION_STRENGTH)
            self.particle.clamp_to_bounds(self.xlim, self.ylim)
            self.sim_time += PARTICLE_DT
            self.analyzer.update(self.sim_time, self.particle.position, self.particle.velocity, self.magnets)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/ -v`
Expected: PASS (all tests, including the pre-existing `test_particle.py` suite — this confirms the wall-bounce fix from the previous session still works unchanged)

- [ ] **Step 8: Commit**

```bash
git add magnetica/render.py tests/test_render_integration.py
git commit -m "feat: wire OrbitAnalyzer into InteractiveScene and restructure layout into a GridSpec"
```

---

### Task 6: Live table, Poincaré scatter, and energy-monitor plot

**Files:**
- Modify: `magnetica/render.py`
- Test: `tests/test_render_integration.py`

**Interfaces:**
- Consumes: `InteractiveScene.analyzer` properties from Tasks 1-4; `self.ax_poincare`/`self.ax_energy` from Task 5.
- Produces: `InteractiveScene.analyzer_text`, `.poincare_scatter`, `.ke_line`, `.pe_line`, `.e_line`, `.rel_err_line`, `.ax_energy_twin` — all updated every tick.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render_integration.py`:

```python
def test_analyzer_table_and_side_plots_update_after_ticks():
    magnets = [Magnet(position=(0, 0), moment=(0, 3), radius=0.1)]
    scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=10, title="test")

    for i in range(600):
        scene._tick(i)

    assert scene.analyzer_text.get_text() != ""
    assert "Status" in scene.analyzer_text.get_text()
    assert len(scene.ke_line.get_xdata()) > 0
    assert len(scene.rel_err_line.get_xdata()) == len(scene.ke_line.get_xdata())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_render_integration.py -v -k analyzer_table_and_side_plots`
Expected: FAIL with `AttributeError: 'InteractiveScene' object has no attribute 'analyzer_text'`

- [ ] **Step 3: Add the new artists at the end of `_build_figure`**

Append to the end of `_build_figure` (after the `self.fig.text(0.5, 0.02, INSTRUCTIONS, ...)` line added in Task 5):

```python
        self.analyzer_text = self.ax_main.text(
            0.02, 0.02, '', transform=self.ax_main.transAxes, va='bottom', ha='left',
            fontsize=8, family='monospace', color='white',
            bbox=dict(facecolor='black', alpha=0.6, pad=4))

        self.poincare_scatter = self.ax_poincare.scatter([], [], s=8, c='deepskyblue')
        self.ax_poincare.set_title('Poincaré section', fontsize=9)
        self.ax_poincare.set_xlabel('r', fontsize=8)
        self.ax_poincare.set_ylabel('radial velocity', fontsize=8)
        self.ax_poincare.tick_params(labelsize=7)

        self.ke_line, = self.ax_energy.plot([], [], color='orange', label='KE')
        self.pe_line, = self.ax_energy.plot([], [], color='deepskyblue', label='PE')
        self.e_line, = self.ax_energy.plot([], [], color='white', label='Total E')
        self.ax_energy.set_title('Energy monitor', fontsize=9)
        self.ax_energy.set_xlabel('time (s)', fontsize=8)
        self.ax_energy.set_ylabel('energy', fontsize=8)
        self.ax_energy.tick_params(labelsize=7)

        self.ax_energy_twin = self.ax_energy.twinx()
        self.rel_err_line, = self.ax_energy_twin.plot(
            [], [], color='red', linestyle='--', label='Rel. error (%)')
        self.ax_energy_twin.set_ylabel('relative error (%)', fontsize=8)
        self.ax_energy_twin.tick_params(labelsize=7)

        energy_lines = [self.ke_line, self.pe_line, self.e_line, self.rel_err_line]
        self.ax_energy.legend(energy_lines, [l.get_label() for l in energy_lines],
                               fontsize=6, loc='upper left')
```

- [ ] **Step 4: Add live updates at the end of `_tick`**

Add a new method just above `_tick` (after `_status_message`, before `def _tick`):

```python
    def _analyzer_table_text(self):
        a = self.analyzer
        period = f"{a.period:.2f} s" if a.period is not None else "—"
        closure = f"{a.closure_pct:.1f}%" if a.closure_pct is not None else "—"
        convex = "—" if a.is_convex is None else ("Yes" if a.is_convex else "No")
        return (
            f"Orbit Analyzer\n"
            f"Status:     {a.status}\n"
            f"Period:     {period}\n"
            f"Closure:    {closure}\n"
            f"Convexity:  {convex}\n"
            f"Crossings:  {a.crossing_count}\n"
            f"Max radius: {a.max_radius:.2f}"
        )
```

Then, at the end of `_tick`, replace the `return` statement:

```python
        return (self.mesh, self.quiver, self.scatter,
                self.trail_line, self.particle_dot, self.status_text)
```

with:

```python
        self.analyzer_text.set_text(self._analyzer_table_text())

        points = self.analyzer.poincare_points
        if points:
            points_array = np.array(points)
            self.poincare_scatter.set_offsets(points_array)
            r_vals, vr_vals = points_array[:, 0], points_array[:, 1]
            r_pad = max((r_vals.max() - r_vals.min()) * 0.1, 0.1)
            vr_pad = max((vr_vals.max() - vr_vals.min()) * 0.1, 0.1)
            self.ax_poincare.set_xlim(r_vals.min() - r_pad, r_vals.max() + r_pad)
            self.ax_poincare.set_ylim(vr_vals.min() - vr_pad, vr_vals.max() + vr_pad)

        hist = self.analyzer.energy_history
        if hist["t"]:
            self.ke_line.set_data(hist["t"], hist["ke"])
            self.pe_line.set_data(hist["t"], hist["pe"])
            self.e_line.set_data(hist["t"], hist["e"])
            rel_err_pct = [v * 100 for v in hist["rel_err"]]
            self.rel_err_line.set_data(hist["t"], rel_err_pct)

            t_min, t_max = hist["t"][0], hist["t"][-1]
            self.ax_energy.set_xlim(t_min, t_max if t_max > t_min else t_min + 1)

            all_energy = hist["ke"] + hist["pe"] + hist["e"]
            e_pad = max((max(all_energy) - min(all_energy)) * 0.1, 0.1)
            self.ax_energy.set_ylim(min(all_energy) - e_pad, max(all_energy) + e_pad)

            self.ax_energy_twin.set_ylim(0, max(rel_err_pct) * 1.2 + 1e-6)

        return (self.mesh, self.quiver, self.scatter,
                self.trail_line, self.particle_dot, self.status_text,
                self.analyzer_text, self.poincare_scatter,
                self.ke_line, self.pe_line, self.e_line, self.rel_err_line)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 6: Render a snapshot and visually verify layout**

Run this as a one-off script (does not need to be committed) to produce a PNG, then view it with an image-reading tool to confirm the analyzer table doesn't overlap the legend or status text, and the instructions line at the bottom doesn't overlap the energy-monitor panel:

```python
import matplotlib
matplotlib.use("Agg")

from magnetica.field import Magnet
from magnetica.render import InteractiveScene

magnets = [
    Magnet(position=(-1, 1), moment=(0, 1)),
    Magnet(position=(1, -1), moment=(3, 3)),
    Magnet(position=(1, 1), moment=(-2, 0)),
    Magnet(position=(-1, -1), moment=(1, -1)),
]
scene = InteractiveScene(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=15, title="verify")
for i in range(600):
    scene._tick(i)
scene.fig.savefig("orbit_analyzer_snapshot.png", dpi=120)
```

Save it to a scratch location (not the repo), run it, then view the resulting PNG. If any text overlaps, adjust the offending element's position/fontsize (`analyzer_text` position, `ax_energy`/`ax_poincare` fontsize, or the GridSpec `wspace`/`hspace`/margins from Task 5) and re-render until clean.

- [ ] **Step 7: Commit**

```bash
git add magnetica/render.py tests/test_render_integration.py
git commit -m "feat: add live orbit-analyzer table, Poincare scatter, and energy monitor plot"
```

---

## Self-Review Notes

- **Spec coverage:** status/period/closure/convexity/crossings/max-radius table (Tasks 1-3, 6), Poincaré section (Tasks 2, 6), energy monitor with KE/PE/E/relative-error on a twin axis (Tasks 1, 4, 6), GridSpec layout with main plot on the right (Task 5), table placed bottom-left of the main plot clear of existing labels (Task 6), all metrics computed from live `self.magnets`/`ATTRACTION_STRENGTH` so drag/add/remove/rescale stay reflected (Task 5) — every spec section has a task.
- **Type consistency checked:** `OrbitAnalyzer.update(t, position, velocity, magnets)` signature is identical across Tasks 1-6; `energy_history` dict keys (`t`, `ke`, `pe`, `e`, `rel_err`) match between the Task 1 implementation and every later consumer in Task 6; `poincare_points` is consistently a `list[tuple[float, float]]`.
- **No placeholders:** every step has literal code and concrete assertions; no "TODO"/"handle appropriately" left in any task.
