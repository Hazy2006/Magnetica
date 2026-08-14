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
