"""Interactive field + particle visualization."""

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import PowerNorm
from magnetica.field import Magnet, compute_attraction_field, compute_bz, check_no_overlap, OverlappingMagnetsError
from magnetica.particle import Particle
from magnetica.analysis import OrbitAnalyzer

PICK_RADIUS = 0.4       # data-space distance within which a click/scroll grabs a magnet
DEFAULT_MOMENT = (0.0, 1.0)
SCROLL_FACTOR = 1.15     # multiplicative strength change per scroll notch
MIN_STRENGTH = 0.1       # floor so a magnet's direction never becomes undefined

PARTICLE_CHARGE = 1.0
PARTICLE_DT = 0.01
PARTICLE_STEPS_PER_FRAME = 15
SPEED_BOOST_FACTOR = 2.5   # substeps-per-frame multiplier while the speed button is held
TRAIL_LENGTH = 600
ATTRACTION_STRENGTH = 0.5   # g in Particle.step_attracted
PARTICLE_ORBIT_OFFSET = np.array([2.5, 0.0])  # start this far from the magnets' centroid

MODE_ATTRACTION = "attraction"   # Particle.step_attracted -- the central pull driving the on-screen orbit
MODE_LORENTZ = "lorentz"         # Particle.step -- the real magnetic Lorentz-force integrator
MODE_LABELS = {
    MODE_ATTRACTION: "Attraction (central pull)",
    MODE_LORENTZ: "Lorentz (magnetic)",
}


def default_particle_start(magnets, g=ATTRACTION_STRENGTH, offset=PARTICLE_ORBIT_OFFSET):
    """Position/velocity for a roughly circular orbit around the magnets'
    centroid, treating their combined strength as one point mass at the
    starting radius -- an approximation, but enough to launch a bounded
    orbit instead of a collision or an escape."""
    centroid = magnet_positions(magnets).mean(axis=0) if magnets else np.zeros(2)
    total_strength = sum(np.linalg.norm(m.moment) for m in magnets) or 1.0
    radius = np.linalg.norm(offset)
    speed = np.sqrt(g * total_strength / radius)
    tangent = np.array([-offset[1], offset[0]]) / radius
    return centroid + offset, tangent * speed

INSTRUCTIONS = (
    "Left-drag: move magnet    |    a, then click: add magnet    |    "
    "Right-click: remove magnet    |    Scroll over a magnet: adjust strength    |    "
    "Hold the speed button: 2.5x fast-forward    |    t: toggle Lorentz/attraction mode"
)


def magnet_marker_size(magnet):
    strength = np.linalg.norm(magnet.moment)
    return 50 + strength * 40


def magnet_positions(magnets):
    if not magnets:
        return np.empty((0, 2))
    return np.array([m.position for m in magnets])


def compute_field_arrows(magnets, x, y, mode):
    if mode == MODE_LORENTZ:
        magnitude = compute_bz(magnets, x, y)
        return np.zeros_like(x), np.zeros_like(y), magnitude

    fx, fy = compute_attraction_field(magnets, x, y, ATTRACTION_STRENGTH)
    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    mag_safe = np.where(magnitude == 0, 1e-9, magnitude)  # avoid divide-by-zero
    return fx / mag_safe, fy / mag_safe, magnitude


def strength_norm(magnitude):
    # Field falls off as 1/r^3, so linear scaling washes out everything but
    # the magnet cores; PowerNorm compresses the range to keep the falloff
    # visible, and a percentile vmax stops the core singularity from
    # blowing out the rest of the scale.
    vmax = max(np.percentile(magnitude, 99), 1e-9)
    return PowerNorm(gamma=0.35, vmin=0, vmax=vmax)


class InteractiveScene:
    """Owns the interactive plot: magnets, field display, animated particle,
    and mouse/keyboard handling. Event handlers only mutate state; the
    animation timer's tick does all the redrawing."""

    def __init__(self, magnets, xlim, ylim, resolution, title):
        check_no_overlap(magnets)
        self.magnets = magnets
        self.xlim = xlim
        self.ylim = ylim
        xs = np.linspace(*xlim, resolution)
        ys = np.linspace(*ylim, resolution)
        self.x, self.y = np.meshgrid(xs, ys)

        self.dragged_index = None
        self.add_armed = False
        self.speed_boost = False
        self.mode = MODE_ATTRACTION
        self.escaped = False

        start_position, start_velocity = default_particle_start(magnets)
        self.particle = Particle(position=start_position,
                                  velocity=start_velocity,
                                  charge=PARTICLE_CHARGE)
        self.trail_x = deque(maxlen=TRAIL_LENGTH)
        self.trail_y = deque(maxlen=TRAIL_LENGTH)
        self.sim_time = 0.0
        self.analyzer = OrbitAnalyzer(g=ATTRACTION_STRENGTH, dt_per_sample=PARTICLE_DT, mode=self.mode)

        self._base_title = title
        self._build_figure(title)
        self._connect_events()

        self.animation = animation.FuncAnimation(
            self.fig, self._tick, interval=20, blit=False, cache_frame_data=False)

    def _field_label(self):
        return 'B_z, out of plane (arb. units)' if self.mode == MODE_LORENTZ else 'attraction field strength (arb. units)'

    def _build_figure(self, title):
        bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y, self.mode)

        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(
            2, 2, width_ratios=[1, 1.7], height_ratios=[1, 1],
            left=0.06, right=0.97, top=0.93, bottom=0.18,
            wspace=0.30, hspace=0.40,
        )
        self.ax_lap = self.fig.add_subplot(gs[0, 0])
        self.ax_energy = self.fig.add_subplot(gs[1, 0])
        self.ax_main = self.fig.add_subplot(gs[:, 1])

        self.mesh = self.ax_main.pcolormesh(self.x, self.y, magnitude, cmap='inferno',
                                             shading='gouraud', norm=strength_norm(magnitude), zorder=0)
        self.quiver = self.ax_main.quiver(self.x, self.y, bx_norm, by_norm,
                                           color='white', alpha=0.8, zorder=2)
        self.quiver.set_visible(self.mode != MODE_LORENTZ)
        self.scatter = self.ax_main.scatter(*magnet_positions(self.magnets).T,
                                             c='red', s=[magnet_marker_size(m) for m in self.magnets],
                                             edgecolors='black', zorder=5, picker=True)

        self.trail_line, = self.ax_main.plot([], [], color='deepskyblue', alpha=0.6, zorder=4)
        self.particle_dot, = self.ax_main.plot([], [], 'go', markersize=8, zorder=6)

        self.colorbar = self.fig.colorbar(self.mesh, ax=self.ax_main, label=self._field_label())
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

        self.ax_speed = self.fig.add_axes([0.435, 0.075, 0.13, 0.05])
        self.ax_speed.set_xticks([])
        self.ax_speed.set_yticks([])
        self.ax_speed.set_facecolor('dimgray')
        for spine in self.ax_speed.spines.values():
            spine.set_color('white')
        self.speed_label = self.ax_speed.text(
            0.5, 0.5, f'Hold: {SPEED_BOOST_FACTOR:g}x speed',
            transform=self.ax_speed.transAxes, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

        self.ax_mode = self.fig.add_axes([0.59, 0.075, 0.16, 0.05])
        self.ax_mode.set_xticks([])
        self.ax_mode.set_yticks([])
        self.ax_mode.set_facecolor('dimgray')
        for spine in self.ax_mode.spines.values():
            spine.set_color('white')
        self.mode_label = self.ax_mode.text(
            0.5, 0.5, '', transform=self.ax_mode.transAxes, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

        self.analyzer_text = self.ax_main.text(
            0.02, 0.02, '', transform=self.ax_main.transAxes, va='bottom', ha='left',
            fontsize=8, family='monospace', color='white',
            bbox=dict(facecolor='black', alpha=0.6, pad=4))

        self.escape_text = self.ax_main.text(
            0.5, 0.5, '', transform=self.ax_main.transAxes, va='center', ha='center',
            fontsize=13, color='red', fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.75, pad=6))

        self.lap_line, = self.ax_lap.plot([], [], color='deepskyblue', marker='o',
                                           markersize=3, linewidth=1)
        self.ax_lap.set_title('Orbit radius per lap', fontsize=9)
        self.ax_lap.set_xlabel('lap #', fontsize=8)
        self.ax_lap.set_ylabel('radius at crossing', fontsize=8)
        self.ax_lap.tick_params(labelsize=7)
        self.ax_lap.text(
            0.5, 0.97, 'flat = periodic   drifting = precessing   jagged = chaotic',
            transform=self.ax_lap.transAxes, ha='center', va='top',
            fontsize=6, color='gray')

        self.ke_line, = self.ax_energy.plot([], [], color='orange', label='KE')
        self.pe_line, = self.ax_energy.plot([], [], color='deepskyblue', label='PE')
        self.e_line, = self.ax_energy.plot([], [], color='white', label='Total E')
        self.ax_energy.set_title('Energy monitor — KE + PE (arb. units)', fontsize=9)
        self.ax_energy.set_xlabel('time (s)', fontsize=8)
        self.ax_energy.set_ylabel('energy (arb. units)', fontsize=8)
        self.ax_energy.tick_params(labelsize=7)

        self.ax_energy_twin = self.ax_energy.twinx()
        self.rel_err_line, = self.ax_energy_twin.plot(
            [], [], color='red', linestyle='--', label='Rel. error (%)')
        self.ax_energy_twin.set_ylabel('relative error (%)', fontsize=8)
        self.ax_energy_twin.tick_params(labelsize=7)

        energy_lines = [self.ke_line, self.pe_line, self.e_line, self.rel_err_line]
        self.ax_energy.legend(energy_lines, [l.get_label() for l in energy_lines],
                               fontsize=6, loc='upper left')

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    # --- picking helpers ---

    def _nearest_magnet(self, xdata, ydata):
        if not self.magnets:
            return None
        distances = [np.hypot(m.position[0] - xdata, m.position[1] - ydata)
                     for m in self.magnets]
        index = int(np.argmin(distances))
        if distances[index] <= PICK_RADIUS:
            return index
        return None

    def _resolve_overlap(self, index, candidate):
        """Push `candidate` back outside every other magnet's radius, so a
        drag can slide a magnet up to its neighbours but never through
        or on top of them."""
        magnet = self.magnets[index]
        for j, other in enumerate(self.magnets):
            if j == index:
                continue
            min_dist = magnet.radius + other.radius
            offset = candidate - other.position
            dist = np.linalg.norm(offset)
            if dist < min_dist:
                if dist == 0:
                    offset = magnet.position - other.position
                    dist = np.linalg.norm(offset)
                    if dist == 0:
                        offset = np.array([1.0, 0.0])
                        dist = 1.0
                candidate = other.position + offset / dist * min_dist
        return candidate

    # --- event handlers: mutate state only, the animation tick redraws ---

    def on_press(self, event):
        if event.button == 1 and event.inaxes == self.ax_speed:
            self.speed_boost = True
            return

        if event.inaxes != self.ax_main or event.xdata is None:
            return

        if event.button == 3:  # right-click: remove
            index = self._nearest_magnet(event.xdata, event.ydata)
            if index is not None:
                if self.dragged_index == index:
                    self.dragged_index = None
                del self.magnets[index]
                self._reset_analyzer()
            return

        if event.button == 1:
            if self.add_armed:
                self._try_add_magnet(event.xdata, event.ydata)
                self.add_armed = False
                return
            self.dragged_index = self._nearest_magnet(event.xdata, event.ydata)

    def _try_add_magnet(self, xdata, ydata):
        candidate = Magnet(position=(xdata, ydata), moment=DEFAULT_MOMENT)
        try:
            check_no_overlap(self.magnets + [candidate])
        except OverlappingMagnetsError:
            return  # spot is taken; ignore the click rather than crash
        self.magnets.append(candidate)
        self._reset_analyzer()

    def on_motion(self, event):
        if self.dragged_index is None or event.inaxes != self.ax_main or event.xdata is None:
            return
        candidate = np.array([event.xdata, event.ydata])
        candidate = self._resolve_overlap(self.dragged_index, candidate)
        self.magnets[self.dragged_index].position = candidate

    def on_release(self, event):
        self.dragged_index = None
        self.speed_boost = False

    def on_scroll(self, event):
        if event.inaxes != self.ax_main or event.xdata is None:
            return
        index = self._nearest_magnet(event.xdata, event.ydata)
        if index is None:
            return
        magnet = self.magnets[index]
        strength = np.linalg.norm(magnet.moment)
        direction = magnet.moment / strength if strength > 0 else np.array([0.0, 1.0])
        factor = SCROLL_FACTOR if event.button == 'up' else 1 / SCROLL_FACTOR
        new_strength = max(strength * factor, MIN_STRENGTH)
        magnet.moment = direction * new_strength

    def on_key(self, event):
        if event.key == 'a':
            self.add_armed = not self.add_armed
        elif event.key == 't':
            self._toggle_mode()

    def _toggle_mode(self):
        """Switch physics model. Resets the particle to the default launch
        state, the trail, and the analyzer so no history mixes across the
        two force laws."""
        self.mode = MODE_LORENTZ if self.mode == MODE_ATTRACTION else MODE_ATTRACTION

        start_position, start_velocity = default_particle_start(self.magnets)
        self.particle = Particle(position=start_position,
                                  velocity=start_velocity,
                                  charge=PARTICLE_CHARGE)
        self.trail_x = deque(maxlen=TRAIL_LENGTH)
        self.trail_y = deque(maxlen=TRAIL_LENGTH)
        self.escaped = False
        self._reset_analyzer()

    def _reset_analyzer(self):
        """Rebaselines the energy monitor. The magnet configuration is part
        of the potential-energy calculation, so adding/removing a magnet
        (or switching physics models) is a legitimate energy discontinuity,
        not integration error -- without this, the analyzer keeps comparing
        against a baseline from before the change and the relative error
        it reports never recovers."""
        self.analyzer = OrbitAnalyzer(g=ATTRACTION_STRENGTH, dt_per_sample=PARTICLE_DT, mode=self.mode)

    # --- redraw ---

    def _status_message(self):
        if self.add_armed:
            return "Add mode armed — click anywhere to place a magnet"
        return ''

    def _out_of_bounds(self, position):
        x, y = position
        return not (self.xlim[0] <= x <= self.xlim[1] and self.ylim[0] <= y <= self.ylim[1])

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

    def _tick(self, frame):
        self.status_text.set_text(self._status_message())

        positions = magnet_positions(self.magnets)
        self.scatter.set_offsets(positions)
        self.scatter.set_sizes([magnet_marker_size(m) for m in self.magnets])

        bx_norm, by_norm, magnitude = compute_field_arrows(self.magnets, self.x, self.y, self.mode)
        self.mesh.set_array(magnitude.ravel())
        self.mesh.set_norm(strength_norm(magnitude))
        self.quiver.set_UVC(bx_norm, by_norm)
        self.quiver.set_visible(self.mode != MODE_LORENTZ)
        self.colorbar.set_label(self._field_label())

        self.ax_speed.set_facecolor('orangered' if self.speed_boost else 'dimgray')
        self.mode_label.set_text(f't: {MODE_LABELS[self.mode]}')
        self.ax_main.set_title(f'{self._base_title} — {MODE_LABELS[self.mode]}')

        steps = PARTICLE_STEPS_PER_FRAME
        if self.speed_boost:
            steps = int(round(PARTICLE_STEPS_PER_FRAME * SPEED_BOOST_FACTOR))

        if not self.escaped:
            for _ in range(steps):
                if self.mode == MODE_LORENTZ:
                    px, py = self.particle.position
                    bz = compute_bz(self.magnets, np.array([px]), np.array([py]))[0]
                    self.particle.step(bz, PARTICLE_DT)
                    if self._out_of_bounds(self.particle.position):
                        # Lorentz mode has no restoring force, so leaving the
                        # frame means it's genuinely escaping, not a close
                        # encounter to bounce back from (see clamp_to_bounds
                        # in attraction mode). Stop rather than let step()'s
                        # per-step energy leak keep compounding unnoticed.
                        self.escaped = True
                        self.particle.clamp_to_bounds(self.xlim, self.ylim)  # pin the dot at the wall it left through
                        break
                else:
                    self.particle.step_attracted(self.magnets, PARTICLE_DT, g=ATTRACTION_STRENGTH)
                    self.particle.clamp_to_bounds(self.xlim, self.ylim)
                self.sim_time += PARTICLE_DT
                self.analyzer.update(self.sim_time, self.particle.position, self.particle.velocity, self.magnets)

            self.trail_x.append(self.particle.position[0])
            self.trail_y.append(self.particle.position[1])
            self.trail_line.set_data(self.trail_x, self.trail_y)
            self.particle_dot.set_data([self.particle.position[0]], [self.particle.position[1]])

        self.escape_text.set_text(
            "PARTICLE ESCAPED — simulation stopped\npress t to reset" if self.escaped else '')

        self.analyzer_text.set_text(self._analyzer_table_text())

        lap_hist = self.analyzer.lap_radius_history
        if lap_hist:
            laps, radii = zip(*lap_hist)
            self.lap_line.set_data(laps, radii)
            lap_min, lap_max = min(laps), max(laps)
            r_min, r_max = min(radii), max(radii)
            r_pad = max((r_max - r_min) * 0.1, 0.1)
            self.ax_lap.set_xlim(lap_min - 0.5, lap_max + 0.5 if lap_max > lap_min else lap_min + 1.5)
            self.ax_lap.set_ylim(r_min - r_pad, r_max + r_pad)

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
        self.ax_energy.set_title(
            'Energy monitor — KE only (Lorentz: no scalar potential)' if self.mode == MODE_LORENTZ
            else 'Energy monitor — KE + PE (arb. units)', fontsize=9)

        return (self.mesh, self.quiver, self.scatter,
                self.trail_line, self.particle_dot, self.status_text,
                self.analyzer_text, self.mode_label, self.escape_text, self.lap_line,
                self.ke_line, self.pe_line, self.e_line, self.rel_err_line)

    def show(self):
        plt.show()


def render_interactive(magnets, xlim=(-5, 5), ylim=(-5, 5), resolution=25, title="Magnetica — field of magnets"):
    """Render magnets and their combined field with a live animated particle
    and mouse/keyboard controls (see INSTRUCTIONS)."""
    scene = InteractiveScene(magnets, xlim, ylim, resolution, title)
    scene.show()
    return scene
