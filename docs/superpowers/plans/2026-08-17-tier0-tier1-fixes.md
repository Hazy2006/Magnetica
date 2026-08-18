# Tier 0/1 Correctness and Honesty Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the field display honestly match whichever physics model drives the particle, fix the Lorentz integrator's energy leak with an exact-rotation (Boris-equivalent) pusher, fix a lap-boundary blind spot in self-intersection detection, make the orbit analyzer mode-aware, add field-recompute caching, and document the remaining physical/UX gaps.

**Architecture:** No new modules. Targeted changes across the existing four files: `magnetica/field.py` (new attraction-field function, error message fix), `magnetica/particle.py` (rotation-based Lorentz step), `magnetica/analysis.py` (rolling self-cross window, mode parameter), `magnetica/render.py` (mode-aware field display, mode-aware analyzer wiring, dirty-flag caching). Each task is independently testable and lands its own commit.

**Tech Stack:** Python, NumPy, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-17-tier0-tier1-fixes-design.md`

## Global Constraints

- Comments: bare minimum only. Add a comment only when it documents a non-obvious *why* (a hidden constraint, a subtle invariant, a workaround) — never restate what the code does. Existing comments the spec calls out to preserve (the divide-by-zero guard in `field.py:30`, the `_reset_analyzer`/`_toggle_mode` docstrings, the Lorentz-escape comment in `render.py`) must NOT be touched or reworded.
- No new abstractions/classes beyond what each task specifies. YAGNI.
- Default mode stays `MODE_ATTRACTION` — do not change `InteractiveScene.__init__`'s default.
- Existing test call sites that don't pass `mode` (e.g. `OrbitAnalyzer(g=1.0)`) must keep working unchanged after Task 5.
- Test command: `pytest` from the repo root (`C:\Users\Puiu\OneDrive\Desktop\Magnetica`). No pytest.ini/config exists; plain discovery works. (Environment note added during execution: bare `pytest` is not on PATH in this environment — use `python -m pytest` instead.)
- Do not recreate `tests/test_field.py` or add a `conftest.py` — explicitly out of scope per the spec.

---

### Task 1: Boris-equivalent rotation pusher for Lorentz mode

**Files:**
- Modify: `magnetica/particle.py:10-20` (the `step` method)
- Test: `tests/test_particle.py` (add a new test; do not remove the existing bounded-growth test)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Particle.step(self, bz, dt)` keeps its exact existing signature and side effects (mutates `self.velocity`, `self.position` in place, returns `None`). Callers in `render.py` and elsewhere are unaffected — only the internal update rule changes.

**Context:** `Particle.step` currently does semi-implicit Euler on the Lorentz force (`fx = charge*(vy*bz)`, `fy = charge*(-vx*bz)`), which leaks energy every step because the velocity-dependent force is evaluated from the pre-step velocity before an ordinary linear update. Since `bz` is a scalar out-of-plane field, the true motion is an *exact* rotation of the velocity vector at angular rate `omega = charge*bz/mass`. Replacing the Euler update with a direct rotation by `theta = -omega*dt` preserves `|v|` to float precision regardless of `dt` — this is the out-of-plane-B specialization of the general Boris pusher.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_particle.py`:

```python
def test_particle_speed_is_conserved_by_rotation_step():
    """The Lorentz force is always perpendicular to velocity and does no
    work, so |v| should be exactly conserved (to float precision) by a
    correct integrator -- not just bounded, as the loose 10x check above
    verifies. This is the test that would have caught the pre-Boris
    energy leak (~1.0015x per step, compounding to 2x -> 348x)."""
    magnets = [Magnet(position=(0, 0), moment=(0, 2), radius=0.1)]
    p = Particle(position=(1, 0), velocity=(0, 2), charge=1.0)
    dt = 0.001
    initial_speed = np.linalg.norm(p.velocity)

    for _ in range(3200):
        px, py = p.position
        bz = compute_bz(magnets, np.array([px]), np.array([py]))[0]
        p.step(bz, dt)
        assert np.linalg.norm(p.velocity) == pytest.approx(initial_speed, rel=1e-9)
```

This needs `import pytest` at the top of `tests/test_particle.py` (not currently imported — only `numpy as np` is).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_particle.py::test_particle_speed_is_conserved_by_rotation_step -v`
Expected: FAIL — `assert 2.00... == approx(2.0 +- 2.0e-09)` fails because current Euler step drifts speed upward over 3200 steps.

- [ ] **Step 3: Write minimal implementation**

Replace `magnetica/particle.py:10-20`:

```python
def step(self, bz, dt):
    """Rotates velocity by the exact gyration angle for one step (bz is
    purely out-of-plane, so the true motion is a rotation) -- the
    out-of-plane-B specialization of the Boris pusher. Preserves |v|
    exactly, unlike the semi-implicit Euler update this replaced."""
    theta = -(self.charge * bz / self.mass) * dt
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    vx, vy = self.velocity
    self.velocity = np.array([vx * cos_t - vy * sin_t, vx * sin_t + vy * cos_t])
    self.position += self.velocity * dt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_particle.py -v`
Expected: PASS on all tests in the file, including the new one and the pre-existing `test_particle_speed_stays_bounded_in_orbit` (which the new behavior trivially satisfies since speed no longer grows at all).

- [ ] **Step 5: Commit**

```bash
git add magnetica/particle.py tests/test_particle.py
git commit -m "fix: replace Lorentz step's leaking Euler update with an exact rotation"
```

---

### Task 2: Attraction-mode field function

**Files:**
- Modify: `magnetica/field.py` (add a new function after `compute_bz`)
- Test: create `tests/test_field.py` (this file does not currently exist — it was previously deleted as empty; this task adds only the one test needed for the new function, not a general test-coverage pass)

**Interfaces:**
- Consumes: `Magnet` class from `field.py` (already exists: `.position`, `.moment`, `.radius`).
- Produces: `compute_attraction_field(magnets, x, y, g)` in `field.py`, returning `(fx, fy)` NumPy arrays shaped like `x`/`y`, for Task 3 to consume in `render.py`.

**Context:** `Particle._attraction_acceleration` (particle.py:22-30) computes, for a single particle position, `g * strength * offset / (r_sq + radius**2)**1.5` summed over magnets, where `offset = magnet.position - particle.position`. This task vectorizes that same formula over a grid of `(x, y)` points so it can drive the field-arrow display in attraction mode (Section A of the spec) instead of the dipole field.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field.py`:

```python
import numpy as np

from magnetica.field import Magnet, compute_attraction_field


def test_attraction_field_points_toward_single_magnet_and_matches_acceleration_formula():
    magnet = Magnet(position=(0, 0), moment=(0, 2), radius=0.1)
    g = 0.5
    x = np.array([[2.0]])
    y = np.array([[0.0]])

    fx, fy = compute_attraction_field([magnet], x, y, g)

    offset = magnet.position - np.array([2.0, 0.0])
    strength = np.linalg.norm(magnet.moment)
    r_sq = offset[0] ** 2 + offset[1] ** 2
    softened = (r_sq + magnet.radius ** 2) ** 1.5
    expected = g * strength * offset / softened

    assert fx[0, 0] == pytest.approx(expected[0])
    assert fy[0, 0] == pytest.approx(expected[1])
    assert fx[0, 0] < 0  # magnet is to the left, so the pull points in -x
```

This needs `import pytest` at the top.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_field.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_attraction_field'`.

- [ ] **Step 3: Write minimal implementation**

Add to `magnetica/field.py`, after `compute_bz`:

```python
def compute_attraction_field(magnets, x, y, g):
    """Vectorized form of Particle._attraction_acceleration's central-pull
    formula, over a grid -- drives the attraction-mode field display so it
    matches the force actually steering the particle, instead of the
    unrelated dipole field."""
    fx_total = np.zeros_like(x, dtype=float)
    fy_total = np.zeros_like(y, dtype=float)
    for magnet in magnets:
        offset_x = magnet.position[0] - x
        offset_y = magnet.position[1] - y
        strength = np.sqrt(magnet.moment[0] ** 2 + magnet.moment[1] ** 2)
        r_sq = offset_x ** 2 + offset_y ** 2
        softened = (r_sq + magnet.radius ** 2) ** 1.5
        fx_total += g * strength * offset_x / softened
        fy_total += g * strength * offset_y / softened
    return fx_total, fy_total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_field.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add magnetica/field.py tests/test_field.py
git commit -m "feat: add vectorized attraction-mode field function"
```

---

### Task 3: Mode-aware field display in render.py

**Files:**
- Modify: `magnetica/render.py` (`compute_field_arrows`, `_build_figure`, `_tick`, imports)

**Interfaces:**
- Consumes: `compute_attraction_field` from Task 2 (`magnetica.field`), `compute_bz` (already imported), `ATTRACTION_STRENGTH` (already defined in `render.py:23`).
- Produces: `InteractiveScene` now shows the attraction-pull field (with quiver) in attraction mode and the `bz`-magnitude heatmap only (quiver hidden) in Lorentz mode. No new public methods; `_tick`'s return tuple and existing method signatures are unchanged.

**Context:** `compute_field_arrows` (render.py:64-71) always calls `compute_total_field` (dipole), used identically in both `_build_figure` (render.py:123) and `_tick` (render.py:388), regardless of `self.mode`. This task branches both call sites on `self.mode`. There is no existing test file assertion on quiver visibility or the exact field values shown, so this task does not add a new test (test coverage for the underlying field function is Task 2's job) — it must not break `test_scene_starts_in_attraction_mode`, `test_lorentz_mode_ticks_without_error_and_speed_stays_bounded`, or any other `test_render_integration.py` test, all of which will be re-run as verification.

- [ ] **Step 1: Update the import line**

In `magnetica/render.py:9`, change:

```python
from magnetica.field import Magnet, compute_total_field, compute_bz, check_no_overlap, OverlappingMagnetsError
```
to:
```python
from magnetica.field import Magnet, compute_attraction_field, compute_bz, check_no_overlap, OverlappingMagnetsError
```

(`compute_total_field` is no longer used anywhere in `render.py` after this task — remove it from the import list.)

- [ ] **Step 2: Replace `compute_field_arrows` with a mode-aware version**

Replace `magnetica/render.py:64-71`:

```python
def compute_field_arrows(magnets, x, y, mode):
    if mode == MODE_LORENTZ:
        magnitude = compute_bz(magnets, x, y)
        return np.zeros_like(x), np.zeros_like(y), magnitude

    fx, fy = compute_attraction_field(magnets, x, y, ATTRACTION_STRENGTH)
    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    mag_safe = np.where(magnitude == 0, 1e-9, magnitude)  # avoid divide-by-zero
    return fx / mag_safe, fy / mag_safe, magnitude
```

(`MODE_LORENTZ`/`ATTRACTION_STRENGTH` are module-level constants already defined above this function, so no new parameters needed beyond `mode`.)

- [ ] **Step 3: Update `_build_figure` call site and hide the quiver in Lorentz mode**

In `magnetica/render.py:123`, change:
```python
bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y)
```
to:
```python
bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y, self.mode)
```

After the existing quiver creation (`render.py:137-138`, `self.quiver = self.ax_main.quiver(...)`), add:
```python
self.quiver.set_visible(self.mode != MODE_LORENTZ)
```

Also update the colorbar label so it's accurate in both modes. Change `render.py:146`:
```python
self.fig.colorbar(self.mesh, ax=self.ax_main, label='field strength (arb. units)')
```
to:
```python
self.colorbar = self.fig.colorbar(self.mesh, ax=self.ax_main, label=self._field_label())
```

Add a new small method right above `_build_figure` (after the `__init__` method, before `_build_figure` at render.py:122):
```python
def _field_label(self):
    return 'B_z, out of plane (arb. units)' if self.mode == MODE_LORENTZ else 'attraction field strength (arb. units)'
```

- [ ] **Step 4: Update `_tick` call site and keep quiver visibility/colorbar label in sync with mode toggles**

In `magnetica/render.py:388`, change:
```python
bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y)
```
to:
```python
bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y, self.mode)
self.quiver.set_visible(self.mode != MODE_LORENTZ)
self.colorbar.set_label(self._field_label())
```
placed immediately after the existing three-line block (`self.mesh.set_array(...)`, `self.mesh.set_norm(...)`, `self.quiver.set_UVC(...)` at render.py:389-391) — i.e. the full replacement for that region is:
```python
bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y, self.mode)
self.mesh.set_array(magnitude.ravel())
self.mesh.set_norm(strength_norm(magnitude))
self.quiver.set_UVC(bx_norm, by_norm)
self.quiver.set_visible(self.mode != MODE_LORENTZ)
self.colorbar.set_label(self._field_label())
```

- [ ] **Step 5: Run the full test suite to verify nothing broke**

Run: `pytest tests/test_render_integration.py -v`
Expected: PASS on all 16 existing tests (none reference quiver visibility or field values directly, so this is a regression check, not new coverage).

- [ ] **Step 6: Commit**

```bash
git add magnetica/render.py
git commit -m "feat: show the field that actually drives the particle in each mode"
```

---

### Task 4: Fix self-cross detection across lap boundaries

**Files:**
- Modify: `magnetica/analysis.py:151-174` (`_on_crossing`)
- Test: `tests/test_analysis.py` (add a new test)

**Interfaces:**
- Consumes: nothing new.
- Produces: `OrbitAnalyzer._lap_points` is now a plain rolling `deque(maxlen=LAP_POINT_CAP)` that is never manually reset. `is_convex`/`crossing_count` properties keep their existing types (`bool|None`, `int`) and meaning to external callers — only their accuracy near lap boundaries changes.

**Context:** `_on_crossing` (analysis.py:171-174) currently discards `_lap_points` down to a single seed point every time a lap completes (2 crossings), so `analyze_lap` (called at analysis.py:169, using the *pre-reset* deque) can miss a self-intersection whose two segments straddle that reset boundary — e.g. a figure-eight orbit reporting `crossing_count == 0` despite a visible self-crossing. Removing the reset makes `_lap_points` a genuine rolling window, so any self-intersection within the trailing `LAP_POINT_CAP` (400) points gets caught regardless of where lap boundaries fall.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_analysis.py`:

```python
def test_self_crossing_is_detected_even_when_it_straddles_a_lap_boundary():
    # A figure-eight: two circular lobes joined at the origin, traced
    # lobe-by-lobe so the self-crossing (at the origin) sits right where
    # the old code used to reset _lap_points after the first lobe's
    # ray-crossing completed a "lap". magnets=[] centers the Poincare
    # ray-crossing check at the origin, same as other tests in this file.
    R = 1.0
    dt = 0.01
    analyzer = OrbitAnalyzer(g=1.0, dt_per_sample=dt)

    t = 0.0
    # Right lobe: circle centered at (R, 0), traced clockwise starting
    # and ending at the origin, passing through (2R, 0) -- this alone
    # produces two ray crossings (a full "lap" under the old reset logic).
    n = 200
    for i in range(n + 1):
        theta = 2 * np.pi * i / n
        position = (R - R * np.cos(theta), R * np.sin(theta))
        velocity = (0.0, 0.0)
        analyzer.update(t, position, velocity, magnets=[])
        t += dt

    # Left lobe: circle centered at (-R, 0), traced starting and ending
    # at the origin again -- the path now revisits the origin, which is
    # the self-crossing point of the full figure-eight.
    for i in range(n + 1):
        theta = 2 * np.pi * i / n
        position = (-R + R * np.cos(theta), R * np.sin(theta))
        velocity = (0.0, 0.0)
        analyzer.update(t, position, velocity, magnets=[])
        t += dt

    assert analyzer.crossing_count > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis.py::test_self_crossing_is_detected_even_when_it_straddles_a_lap_boundary -v`
Expected: FAIL — `assert 0 > 0` (the pre-fix reset drops the right lobe's points before the left lobe's pass through the origin can be compared against them).

- [ ] **Step 3: Write minimal implementation**

Replace `magnetica/analysis.py:171-174` (the tail of `_on_crossing`, right after the `if len(self._crossing_times) == 2:` block):

```python
        seed = self._lap_points[-1] if self._lap_points else None
        self._lap_points = deque(maxlen=LAP_POINT_CAP)
        if seed is not None:
            self._lap_points.append(seed)
```

with: (delete these four lines entirely — nothing replaces them; `_lap_points` keeps accumulating as the plain `deque(maxlen=LAP_POINT_CAP)` created in `__init__` at analysis.py:91, so points roll off the oldest end automatically instead of being explicitly reset).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis.py -v`
Expected: PASS on all tests including the new one. Pay attention to `test_lap_points_never_exceed_cap_with_no_crossings` (still must pass — it doesn't depend on the reset, only on `maxlen`) and `test_crossing_events_increments_every_crossing_unlike_crossing_count` (a circle never self-intersects, so removing the reset shouldn't change its `crossing_count == 0` result — this test's `magnets=[]`/circular-motion scenario has no lap boundary self-crossing to expose).

- [ ] **Step 5: Commit**

```bash
git add magnetica/analysis.py tests/test_analysis.py
git commit -m "fix: stop resetting the self-cross detection window at lap boundaries"
```

---

### Task 5: Mode-aware analyzer

**Files:**
- Modify: `magnetica/analysis.py` (`OrbitAnalyzer.__init__`, `update`)
- Modify: `magnetica/render.py` (`_reset_analyzer`, `_analyzer_table_text`, `_tick`'s energy-plot block, `_build_figure`'s energy-plot setup)
- Test: `tests/test_analysis.py` (add new tests)

**Interfaces:**
- Consumes: `MODE_ATTRACTION`/`MODE_LORENTZ` string constants — `analysis.py` does not import from `render.py` (would be a backwards dependency, since `render.py` already imports from `analysis.py`); instead `OrbitAnalyzer` accepts a plain `mode` string parameter and compares it against a new pair of constants defined in `analysis.py` itself, `MODE_ATTRACTION = "attraction"` and `MODE_LORENTZ = "lorentz"` (same string values `render.py` already uses for its own same-named constants, so passing `render.py`'s `self.mode` straight through works without translation).
- Produces: `OrbitAnalyzer(g, energy_window_seconds=..., dt_per_sample=None, mode=MODE_ATTRACTION)` — new optional 4th parameter, default preserves all existing call sites' behavior exactly. `OrbitAnalyzer.status`, `.energy_history["pe"]`, `.energy_history["e"]`, `.energy_history["rel_err"]` become `None` (for `status`) or `[]`-populated-with-no-PE-entries (for the history lists) when `mode == MODE_LORENTZ` — see Step 3 for the exact history-shape decision.

**Context:** `_potential_energy` (analysis.py:16-23) and the `status` sign-check (analysis.py:129) are unconditionally computed from the central-force potential, meaningless when the particle is actually following Lorentz dynamics. `render.py` already has a comment acknowledging this (render.py:110-112) without fixing it. `InteractiveScene` already knows `self.mode` and already rebuilds the analyzer via `_reset_analyzer` on every mode toggle (render.py:344-351), so the render-side plumbing is just passing one more argument through.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_analysis.py`:

```python
from magnetica.analysis import OrbitAnalyzer, LAP_POINT_CAP, MODE_ATTRACTION, MODE_LORENTZ, segments_intersect, analyze_lap
```

(replaces the existing import line at the top of the file, which is missing `MODE_ATTRACTION, MODE_LORENTZ`).

```python
def test_lorentz_mode_analyzer_reports_no_status_or_potential_energy():
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0, mode=MODE_LORENTZ)
    analyzer.update(0.0, position=(2.0, 0.0), velocity=(0.0, 1.0), magnets=magnets)

    assert analyzer.status is None
    hist = analyzer.energy_history
    assert hist["ke"] == [pytest.approx(0.5)]
    assert hist["pe"] == []
    assert hist["e"] == []
    assert hist["rel_err"] == []


def test_attraction_mode_analyzer_default_still_reports_status_and_pe():
    # Default mode is unchanged -- existing call sites like
    # OrbitAnalyzer(g=1.0) elsewhere in this file must keep working.
    magnets = [Magnet(position=(0, 0), moment=(0, 1), radius=0.1)]
    analyzer = OrbitAnalyzer(g=1.0)
    analyzer.update(0.0, position=(2.0, 0.0), velocity=(0.0, 0.0), magnets=magnets)

    assert analyzer.status == "Bounded"
    hist = analyzer.energy_history
    assert len(hist["pe"]) == 1
    assert len(hist["e"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis.py::test_lorentz_mode_analyzer_reports_no_status_or_potential_energy tests/test_analysis.py::test_attraction_mode_analyzer_default_still_reports_status_and_pe -v`
Expected: first test FAILs with `TypeError: __init__() got an unexpected keyword argument 'mode'` (or `ImportError` for `MODE_LORENTZ` if collection fails first); second test currently passes already (documenting no-regression) but will be collected alongside so run both together.

- [ ] **Step 3: Write minimal implementation**

In `magnetica/analysis.py`, add near the top (after the existing module constants at line 8):
```python
MODE_ATTRACTION = "attraction"
MODE_LORENTZ = "lorentz"
```

Change `OrbitAnalyzer.__init__` signature (analysis.py:77):
```python
def __init__(self, g, energy_window_seconds=ENERGY_WINDOW_SECONDS, dt_per_sample=None, mode=MODE_ATTRACTION):
```
and add one line storing it, right after `self.g = g` (analysis.py:78):
```python
self.mode = mode
```

Change `update` (analysis.py:109-129) so PE/status are skipped in Lorentz mode. Replace analysis.py:113-129:
```python
        ke = _kinetic_energy(velocity)
        pe = _potential_energy(position, magnets, self.g)
        e = ke + pe
        if self._e0 is None:
            self._e0 = e
            # Normalize against |KE|+|PE| rather than E itself, since E can
            # land near zero for a marginally-bound orbit and blow up the ratio.
            self._e_scale = max(abs(ke) + abs(pe), 1e-9)
        rel_err = abs(e - self._e0) / self._e_scale

        self._t_hist.append(t)
        self._ke_hist.append(ke)
        self._pe_hist.append(pe)
        self._e_hist.append(e)
        self._rel_err_hist.append(rel_err)

        self._status = "Bounded" if e < 0 else "Unbound"
```
with:
```python
        ke = _kinetic_energy(velocity)
        self._t_hist.append(t)
        self._ke_hist.append(ke)

        if self.mode == MODE_LORENTZ:
            self._status = None
        else:
            pe = _potential_energy(position, magnets, self.g)
            e = ke + pe
            if self._e0 is None:
                self._e0 = e
                # Normalize against |KE|+|PE| rather than E itself, since E can
                # land near zero for a marginally-bound orbit and blow up the ratio.
                self._e_scale = max(abs(ke) + abs(pe), 1e-9)
            rel_err = abs(e - self._e0) / self._e_scale

            self._pe_hist.append(pe)
            self._e_hist.append(e)
            self._rel_err_hist.append(rel_err)
            self._status = "Bounded" if e < 0 else "Unbound"
```

(The rest of `update` — centroid/max_radius/crossing logic below this block — is mode-agnostic and stays exactly as-is, per the spec.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis.py -v`
Expected: PASS on all tests, including the two new ones and every pre-existing test (which all construct `OrbitAnalyzer` without `mode`, exercising the default).

- [ ] **Step 5: Wire `mode` through from render.py**

In `magnetica/render.py`, change both `OrbitAnalyzer(...)` construction sites to pass `mode=self.mode`:

`render.py:113` (inside `__init__`):
```python
self.analyzer = OrbitAnalyzer(g=ATTRACTION_STRENGTH, dt_per_sample=PARTICLE_DT, mode=self.mode)
```
(Also delete the now-stale comment directly above this line at render.py:110-112 — `"# g/dt_per_sample assume the attraction model's central-force potential; in Lorentz mode the displayed PE/status are not physically meaningful, since compute_bz has no corresponding potential energy."` — the analyzer now handles this itself, so the comment is no longer accurate as a caveat and would read as leftover.)

`render.py:351` (inside `_reset_analyzer`):
```python
self.analyzer = OrbitAnalyzer(g=ATTRACTION_STRENGTH, dt_per_sample=PARTICLE_DT, mode=self.mode)
```

No import change needed in `render.py:11`. `render.py` already defines its own `MODE_ATTRACTION`/`MODE_LORENTZ` constants with the same string values (`"attraction"`/`"lorentz"`, render.py:26-27) that `analysis.py`'s new constants use, and `OrbitAnalyzer` compares `mode` by value — so passing `self.mode` (render.py's own string) straight through, as shown above, is sufficient.

- [ ] **Step 6: Hide PE/Total E/Status in Lorentz mode in the analyzer table**

Replace `_analyzer_table_text` (render.py:364-379):
```python
    def _analyzer_table_text(self):
        a = self.analyzer
        period = f"{a.period:.2f} s" if a.period is not None else "—"
        closure = f"{a.closure_pct:.1f}%" if a.closure_pct is not None else "—"
        convex = "—" if a.is_convex is None else ("Yes" if a.is_convex else "No")
        status_line = f"Status:      {a.status}\n" if a.status is not None else ""
        return (
            f"Orbit Analyzer\n"
            f"Mode:        {MODE_LABELS[self.mode]}\n"
            f"{status_line}"
            f"Period:      {period}\n"
            f"Closure:     {closure}\n"
            f"Convexity:   {convex}\n"
            f"Crossings:   {a.crossing_events}\n"
            f"Self-cross:  {a.crossing_count}\n"
            f"Max radius:  {a.max_radius:.2f}"
        )
```

- [ ] **Step 7: Hide PE/Total E lines on the energy plot in Lorentz mode**

In `_tick` (render.py:442-457), replace:
```python
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
```
with:
```python
        hist = self.analyzer.energy_history
        if hist["t"]:
            self.ke_line.set_data(hist["t"], hist["ke"])
            t_min, t_max = hist["t"][0], hist["t"][-1]
            self.ax_energy.set_xlim(t_min, t_max if t_max > t_min else t_min + 1)

            if self.mode == MODE_LORENTZ:
                self.pe_line.set_data([], [])
                self.e_line.set_data([], [])
                self.rel_err_line.set_data([], [])
                ke_pad = max((max(hist["ke"]) - min(hist["ke"])) * 0.1, 0.1)
                self.ax_energy.set_ylim(min(hist["ke"]) - ke_pad, max(hist["ke"]) + ke_pad)
                self.ax_energy_twin.set_ylim(0, 1)
            else:
                self.pe_line.set_data(hist["t"], hist["pe"])
                self.e_line.set_data(hist["t"], hist["e"])
                rel_err_pct = [v * 100 for v in hist["rel_err"]]
                self.rel_err_line.set_data(hist["t"], rel_err_pct)

                all_energy = hist["ke"] + hist["pe"] + hist["e"]
                e_pad = max((max(all_energy) - min(all_energy)) * 0.1, 0.1)
                self.ax_energy.set_ylim(min(all_energy) - e_pad, max(all_energy) + e_pad)
                self.ax_energy_twin.set_ylim(0, max(rel_err_pct) * 1.2 + 1e-6)
```

Also update the energy-plot title set in `_build_figure` (render.py:209) to not overclaim in Lorentz mode. Change:
```python
self.ax_energy.set_title('Energy monitor — KE + PE (arb. units)', fontsize=9)
```
to a call that's refreshed per-tick instead of fixed at build time — add this line inside `_tick`, right after the `if hist["t"]:` block closes (same indentation as the `hist = ...` line):
```python
        self.ax_energy.set_title(
            'Energy monitor — KE only (Lorentz: no scalar potential)' if self.mode == MODE_LORENTZ
            else 'Energy monitor — KE + PE (arb. units)', fontsize=9)
```
(the `set_title` call in `_build_figure` at render.py:209 can stay as the initial value; `_tick` overwrites it every frame same as it already does for `ax_main`'s title at render.py:395.)

- [ ] **Step 8: Run the full test suite**

Run: `pytest -v`
Expected: PASS on everything. Pay particular attention to `test_analyzer_table_and_side_plots_update_after_ticks` (asserts `"Status" in scene.analyzer_text.get_text()` — still true in attraction mode, the default the test uses) and `test_scene_ticks_without_error_and_analyzer_updates` (asserts `scene.analyzer.status in ("Bounded", "Unbound")` — still true, attraction mode default).

- [ ] **Step 9: Commit**

```bash
git add magnetica/analysis.py magnetica/render.py tests/test_analysis.py
git commit -m "feat: make the orbit analyzer mode-aware, hiding meaningless PE/status in Lorentz mode"
```

---

### Task 6: Dirty-flag field caching

**Files:**
- Modify: `magnetica/render.py` (`__init__`, `on_motion`, `on_scroll`, `_try_add_magnet`, `on_press`, `_toggle_mode`, `_tick`)

**Interfaces:**
- Consumes: `compute_field_arrows`/`strength_norm` from Task 3 (unchanged signatures).
- Produces: no new public interface; `_tick` behavior is observably identical (same mesh/quiver/colorbar end state each frame) but skips recomputation when nothing changed since the last tick.

**Context:** `_tick` recomputes the field grid (`compute_field_arrows`) and `strength_norm`'s percentile unconditionally every ~20ms frame, even when no magnet moved and the mode didn't change. This task adds a `self._field_dirty` flag, set wherever a magnet's position/strength changes or the mode toggles, and only recomputes when set (cleared after each recompute).

- [ ] **Step 1: Add the flag and initialize it dirty**

In `__init__` (render.py, right after `self.mode = MODE_ATTRACTION` at line 100), add:
```python
self._field_dirty = True
```

- [ ] **Step 2: Set the flag on every state change that affects the field**

In `on_motion` (render.py:299-304), after the line `self.magnets[self.dragged_index].position = candidate`, add:
```python
self._field_dirty = True
```

In `on_scroll` (render.py:310-321), after the line `magnet.moment = direction * new_strength`, add:
```python
self._field_dirty = True
```

In `_try_add_magnet` (render.py:290-297), after `self.magnets.append(candidate)`, add:
```python
self._field_dirty = True
```

In `on_press`'s right-click delete branch (render.py:274-281), after `del self.magnets[index]`, add:
```python
self._field_dirty = True
```

In `_toggle_mode` (render.py:329-342), after `self.mode = MODE_LORENTZ if self.mode == MODE_ATTRACTION else MODE_ATTRACTION`, add:
```python
self._field_dirty = True
```

- [ ] **Step 3: Only recompute in `_tick` when dirty**

Replace the block in `_tick` produced by Task 3 Step 4:
```python
bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y, self.mode)
self.mesh.set_array(magnitude.ravel())
self.mesh.set_norm(strength_norm(magnitude))
self.quiver.set_UVC(bx_norm, by_norm)
self.quiver.set_visible(self.mode != MODE_LORENTZ)
self.colorbar.set_label(self._field_label())
```
with:
```python
if self._field_dirty:
    bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y, self.mode)
    self.mesh.set_array(magnitude.ravel())
    self.mesh.set_norm(strength_norm(magnitude))
    self.quiver.set_UVC(bx_norm, by_norm)
    self.quiver.set_visible(self.mode != MODE_LORENTZ)
    self.colorbar.set_label(self._field_label())
    self._field_dirty = False
```

- [ ] **Step 4: Run the full test suite**

Run: `pytest -v`
Expected: PASS on everything, including `test_toggling_mode_keeps_drag_and_scroll_handlers_working` (drags/scrolls after a mode toggle — exercises the dirty flag being set from multiple call sites in sequence) and `test_adding_a_magnet_rebaselines_the_analyzer`/`test_removing_a_magnet_rebaselines_the_analyzer` (exercise add/remove setting the flag). None of these tests assert on `_field_dirty` directly or on mesh/quiver pixel data, so this is a regression check confirming the caching doesn't change observable behavior.

- [ ] **Step 5: Commit**

```bash
git add magnetica/render.py
git commit -m "perf: only recompute the field grid when a magnet or mode actually changed"
```

---

### Task 7: Lorentz tuning experiment

**Files:**
- Create (temporary, not committed): a throwaway script in the repo root or scratch location to sweep parameters
- Possible modify: `magnetica/render.py` (`default_particle_start`, `PARTICLE_ORBIT_OFFSET`, or `xlim`/`ylim` defaults in `render_interactive`) — only if the experiment finds a clear improvement

**Interfaces:**
- Consumes: `Particle`, `compute_bz`, `default_particle_start` — all already exist.
- Produces: either an updated default (Task 7a below) or a documented finding with no code change (Task 7b below) — decided by the experiment's actual output, not presupposed.

**Context:** With Task 1's rotation-based `step` fixing the energy leak, some of today's near-immediate Lorentz escape may no longer occur (the leak was inflating speed, which pushes the particle toward the wall faster). Before touching escape logic, run a headless experiment sweeping initial speed and starting radius to see if a bounded/orbiting trajectory is achievable with the post-Task-1 integrator.

- [ ] **Step 1: Write and run the sweep script**

Write this to a scratch file (e.g. `C:\Users\Puiu\AppData\Local\Temp\claude\...\scratchpad\lorentz_tuning.py` — NOT committed to the repo) and run it with `python`:

```python
import numpy as np
from magnetica.field import Magnet, compute_bz
from magnetica.particle import Particle

magnets = [
    Magnet(position=(-1, 1), moment=(0, 1)),
    Magnet(position=(1, -1), moment=(3, 3)),
    Magnet(position=(1, 1), moment=(-2, 0)),
    Magnet(position=(-1, -1), moment=(1, -1)),
]
xlim = ylim = (-5, 5)
dt = 0.01

def out_of_bounds(pos):
    x, y = pos
    return not (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1])

for start_radius in (0.5, 1.0, 1.5, 2.0, 2.5):
    for speed in (0.5, 1.0, 1.5, 2.0, 3.0):
        p = Particle(position=(start_radius, 0.0), velocity=(0.0, speed), charge=1.0)
        steps_survived = 0
        for step in range(20000):
            px, py = p.position
            bz = compute_bz(magnets, np.array([px]), np.array([py]))[0]
            p.step(bz, dt)
            if out_of_bounds(p.position):
                break
            steps_survived = step
        print(f"radius={start_radius:.1f} speed={speed:.1f} -> survived {steps_survived} steps "
              f"({'BOUNDED (full run)' if steps_survived == 19999 else 'escaped'})")
```

Run: `python C:\Users\Puiu\AppData\Local\Temp\claude\...\scratchpad\lorentz_tuning.py` (adjust the scratch path to the current session's scratchpad directory).

- [ ] **Step 2: Interpret the output and pick a branch**

Read the printed table.

**If at least one `(start_radius, speed)` combination survives the full 20000-step run ("BOUNDED"):** proceed to Step 3a.
**If none survive the full run:** proceed to Step 3b.

- [ ] **Step 3a: (if a bounded combination was found) Apply it as the Lorentz-mode launch default**

Note the surviving `(start_radius, speed)` pair from the sweep output (call them `R_bound` and `V_bound`). In `magnetica/render.py`, `_toggle_mode` (render.py:335) and `__init__` (render.py:103) both call `default_particle_start(magnets)` — this single shared helper currently only serves attraction mode's orbit launch. Add a Lorentz-specific variant rather than changing the shared one (attraction mode's tuning must not regress):

Add to `magnetica/render.py`, near `PARTICLE_ORBIT_OFFSET` (render.py:24):
```python
LORENTZ_ORBIT_RADIUS = <R_bound from the sweep>
LORENTZ_ORBIT_SPEED = <V_bound from the sweep>
```

Add a new function near `default_particle_start` (render.py:34-44):
```python
def default_lorentz_start(magnets, radius=LORENTZ_ORBIT_RADIUS, speed=LORENTZ_ORBIT_SPEED):
    """Launch state tuned (see the Lorentz tuning experiment) to stay
    bounded rather than escape almost immediately."""
    centroid = magnet_positions(magnets).mean(axis=0) if magnets else np.zeros(2)
    return centroid + np.array([radius, 0.0]), np.array([0.0, speed])
```

In both `__init__` and `_toggle_mode`, branch on which start function to call:
```python
start_position, start_velocity = (
    default_lorentz_start(magnets) if self.mode == MODE_LORENTZ
    else default_particle_start(magnets)
)
```
(`__init__` at render.py:103 always uses attraction mode at construction time since `self.mode` defaults to `MODE_ATTRACTION` before this line — so `__init__` can keep calling `default_particle_start(magnets)` unconditionally; only `_toggle_mode` at render.py:335 needs the branch, since it runs *after* `self.mode` has just been flipped to whichever mode is now active.)

- [ ] **Step 3b: (if no combination was bounded) Document the expected-escape finding**

No code change to launch parameters. Add a short note to the README (folded into Task 8's README update — do not create a separate doc) stating that Lorentz mode's default configuration escapes the visible domain for typical initial conditions, since a purely `1/r²`-falling-off out-of-plane field can only steer, never fully re-capture an outward-drifting particle — this is expected physics, not a bug, and matches the audit document's own physical reasoning.

- [ ] **Step 4: If Step 3a was taken, run the full test suite**

Run: `pytest -v`
Expected: PASS. `test_lorentz_mode_ticks_without_error_and_speed_stays_bounded` and `test_lorentz_mode_stops_and_flags_escape_when_particle_leaves_the_frame` both explicitly set `scene.particle.position`/`.velocity` directly after construction (render_integration tests lines 132-146, 194-216), bypassing `default_lorentz_start` entirely, so this change cannot break them.

- [ ] **Step 5: Commit (only if Step 3a produced a code change; if 3b, fold the README note into Task 8's commit instead)**

```bash
git add magnetica/render.py
git commit -m "tune: use an empirically bounded launch state for Lorentz mode"
```

---

### Task 8: Documentation and UX cleanup

**Files:**
- Modify: `magnetica/field.py:15` (error message)
- Modify: `README.md`
- Test: none (text-only changes; existing tests already cover `OverlappingMagnetsError` being raised, just not its message text)

**Interfaces:** none — no code behavior changes in this task beyond the error string.

**Context:** `field.py:15`'s overlap error message ("please change coordinates in the code") is stale — magnets are draggable via the UI now, not hardcoded. `README.md` currently contains only the title (`# Magnetica`) and needs to state that both physics models exist, which one is "real" magnetism, and that the field display now matches whichever model is active (Section A/H of the spec).

- [ ] **Step 1: Fix the stale error message**

In `magnetica/field.py:15`, change:
```python
                    "magnets overlap, please change coordinates in the code"
```
to:
```python
                    "magnets overlap — drag one apart before continuing"
```

- [ ] **Step 2: Run the existing overlap test to confirm it still passes (message text isn't asserted, only the exception type/trigger)**

Run: `pytest tests/test_render_integration.py -k overlap -v` if any such test exists; if not (none currently assert on this message), run the full suite instead: `pytest -v`.
Expected: PASS.

- [ ] **Step 3: Write the README**

Replace the full contents of `README.md`:

```markdown
# Magnetica

An interactive 2D magnet/particle simulator with two switchable physics
models (press `t` to toggle):

- **Attraction mode** (default): a central-force pull toward each magnet,
  integrated with velocity Verlet. This is orbital-mechanics-style
  motion, not real magnetism — a stand-in force chosen because it
  produces stable, visually pleasing bounded orbits.
- **Lorentz mode**: the actual magnetic `F = q(v × B)` force from an
  out-of-plane field `B_z`, integrated with an exact rotation (the
  out-of-plane-B specialization of the Boris pusher) that conserves
  particle speed exactly. This is the model that's actually "magnetic."

The field display always shows whichever field is driving the particle
in the current mode — the central-pull field with direction arrows in
attraction mode, or the `B_z` magnitude heatmap (no arrows — it's a
scalar, out-of-plane field with no in-plane direction to draw) in
Lorentz mode. It never shows a field that isn't the one in effect.

Run with `python -m magnetica.main`.
```

If Task 7 took branch 3b (no bounded Lorentz configuration found), append this paragraph to the README before the "Run with" line:

```markdown
Lorentz mode's default configuration will typically escape the visible
domain rather than settle into a bounded orbit — a field that falls off
as `1/r²` can steer a particle but can't fully recapture one already
drifting outward, so escape is the expected outcome for most starting
conditions, not a bug.
```

- [ ] **Step 4: Commit**

```bash
git add magnetica/field.py README.md
git commit -m "docs: fix stale overlap error message, document both physics models in README"
```

---

## Verification (after all tasks)

Run the full suite once more from the repo root:
```bash
pytest -v
```
Expected: all tests across `tests/test_particle.py`, `tests/test_field.py`, `tests/test_analysis.py`, `tests/test_render_integration.py` pass.

Then launch the app manually to visually confirm:
```bash
python -m magnetica.main
```
Check: attraction mode shows pull-direction arrows toward magnets (not dipole lobes); pressing `t` switches to Lorentz mode, hides the arrows, shows only the `B_z` heatmap, and the energy panel shows a flat KE line with no PE/Total E/rel-error traces and no "Status:" row in the analyzer table; dragging/scrolling a magnet updates the field display in both modes; a self-intersecting orbit (if one occurs) shows a nonzero "Self-cross" count.
