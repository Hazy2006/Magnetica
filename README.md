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

Lorentz mode's tuned launch state significantly delays escape compared
to an untuned start, but the particle will still eventually leave the
visible domain -- a field that falls off as 1/r² can steer a particle
but can't fully recapture one already drifting outward, so eventual
escape is expected physical behavior, not a bug.

Run with `python -m magnetica.main`.
