import numpy as np
from collections import deque

MIN_CROSSING_INTERVAL = 0.5   # seconds of simulation time
POINCARE_CAP = 500
LAP_RADIUS_CAP = 500
ENERGY_WINDOW_SECONDS = 30.0
LAP_POINT_CAP = 400   # caps analyze_lap's O(n^2) segment-crossing loop

MODE_ATTRACTION = "attraction"
MODE_LORENTZ = "lorentz"


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
    polyline is convex (every turn the same rotational sign, and no
    self-intersection -- turning-sign alone isn't enough, since a
    star/rosette shape turns one way at every vertex too) and how many
    times it crosses itself. Returns (None, 0) if fewer than 3 points."""
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


class OrbitAnalyzer:
    """Tracks live orbit-shape and energy metrics from a stream of
    (t, position, velocity, magnets) physics substeps. Has no matplotlib
    dependency so it can be tested and reasoned about on its own."""

    def __init__(self, g, energy_window_seconds=ENERGY_WINDOW_SECONDS, dt_per_sample=None, mode=MODE_ATTRACTION):
        self.g = g
        self.mode = mode
        self._max_radius = 0.0
        self._status = "Bounded"

        self._prev_rel = None
        self._last_crossing_t = None
        self._crossing_times = deque(maxlen=2)
        self._crossing_radii = deque(maxlen=2)
        self._period = None
        self._closure_pct = None

        self._poincare_points = deque(maxlen=POINCARE_CAP)

        self._lap_points = deque(maxlen=LAP_POINT_CAP)
        self._is_convex = None
        self._crossing_count = 0

        self._crossing_events = 0  # increments every crossing, unlike _crossing_count (once per lap)
        self._lap_radius_hist = deque(maxlen=LAP_RADIUS_CAP)

        maxlen = None
        if dt_per_sample:
            maxlen = max(int(energy_window_seconds / dt_per_sample), 1)
        self._t_hist = deque(maxlen=maxlen)
        self._ke_hist = deque(maxlen=maxlen)
        self._pe_hist = deque(maxlen=maxlen)
        self._e_hist = deque(maxlen=maxlen)
        self._rel_err_hist = deque(maxlen=maxlen)
        self._e0 = None
        self._e_scale = None

    def update(self, t, position, velocity, magnets):
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)

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

        centroid = _magnets_centroid(magnets)
        rel = position - centroid
        r = float(np.linalg.norm(rel))
        self._max_radius = max(self._max_radius, r)

        if self._prev_rel is not None:
            # Poincare section: a "crossing" is the particle passing through
            # y=0 (relative to the magnets' centroid) on the +x side.
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
        self._lap_points.append((float(position[0]), float(position[1])))

    def _on_crossing(self, t, r, rel, velocity):
        self._last_crossing_t = t
        self._crossing_times.append(t)
        self._crossing_radii.append(r)

        r_hat = rel / r if r > 0 else np.zeros(2)
        v_r = float(np.dot(velocity, r_hat))  # radial velocity at the crossing
        self._poincare_points.append((r, v_r))  # (r, v_r) is the Poincare sample

        self._crossing_events += 1
        self._lap_radius_hist.append((self._crossing_events, r))

        if len(self._crossing_times) == 2:
            self._period = self._crossing_times[1] - self._crossing_times[0]
            r_prev, r_curr = self._crossing_radii
            denom = max(r_prev, r_curr, 1e-9)
            self._closure_pct = 100.0 * (1.0 - abs(r_curr - r_prev) / denom)

            self._is_convex, self._crossing_count = analyze_lap(list(self._lap_points))

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
    def crossing_events(self):
        """Total ray crossings seen so far (one per Poincare sample)."""
        return self._crossing_events

    @property
    def poincare_points(self):
        return list(self._poincare_points)

    @property
    def lap_radius_history(self):
        """(lap number, radius at that crossing) pairs, oldest first."""
        return list(self._lap_radius_hist)

    @property
    def energy_history(self):
        return {
            "t": list(self._t_hist),
            "ke": list(self._ke_hist),
            "pe": list(self._pe_hist),
            "e": list(self._e_hist),
            "rel_err": list(self._rel_err_hist),
        }
