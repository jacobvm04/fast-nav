"""Robot kinematics as a strategy: everything that differs between drive types
lives here, one class per kinematics, registered in KINEMATICS.

Each kinematics owns two things:

1. A small set of Metal inline functions (`metal`) with fixed signatures that
   the shared step/lidar/expert kernels in sim.py call. Kernels are compiled
   once per kinematics, so there is no per-step branching and the collision,
   odometry, termination, and auto-reset machinery stays single-source.

2. The Python mirrors that trainers, envs, and the policy need: per-dimension
   action limits (`action_scale`), the same action clamp the kernel applies
   (`clamp`), normalized linear speed for reward shaping (`speed`), and the
   goal-vector rotation into the observation frame (`rel_goal`).

Action convention -- every kinematics has a 2-dim action:
  holonomic:     (vx, vy)   velocity in the believed world frame, |v| <= v_max
  diffdrive:     (v, omega) forward velocity + yaw rate, |v| <= v_max, |w| <= w_max
  diffdrive_vel: (vx, vy)   desired velocity in the body frame, |v| <= v_max;
                 an in-kernel P-steering controller (the same one the diffdrive
                 expert uses) converts it to (v, omega) before unicycle
                 integration. Decouples direction choice (learned, smooth
                 holonomic-shaped labels) from steering dynamics (fixed
                 controller, proven at ~99.5% expert success).

Frame convention, shared by the kernels (see sim.py):
  heading      true body heading in the world frame (holonomic: unused, 0)
  odom[:, 2]   orientation of the *believed* command/sensor frame:
               holonomic -> accumulated heading error of the believed world
               frame; diffdrive -> believed heading (anchored true at reset)
  kin_frame()  maps (heading, odom theta) to the TRUE world orientation of the
               frame commands execute in and lidar rays are indexed in.

Metal contract (signatures; holonomic is the reference implementation):
  constant int KIN_NU            uniforms per env the step kernel consumes
  kin_execute(a0, a1, vmax, wmax, ascale, anoise, n0, n1, &e0, &e1, &w)
      clamp + actuation noise -> executed twist: (e0, e1) m/s in the command
      frame, w rad/s yaw rate. The kernel integrates R(frame)*(e0, e1) with
      the frame rotating by w within the step.
  kin_frame(th, odo_th) -> float
      true world orientation of the command/sensor frame.
  kin_reset(u, &th, &odo_th)
      episode start: true heading + believed-frame anchor (u: uniform [0,1),
      only drawn when KIN_NU > 4).
  kin_expert(ax, ay, speed, th, beta, wmax, tgain, prev0, prev1, &a0, &a1)
      desired world velocity (ax, ay) of magnitude `speed` -> action command,
      low-pass smoothed toward prev where it helps.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

_HOLONOMIC_METAL = """
constant int KIN_NU = 4;

inline void kin_execute(float a0, float a1, float vmax, float wmax,
                        float ascale, float anoise, float n0, float n1,
                        thread float& e0, thread float& e1, thread float& w) {
    float vn = metal::sqrt(a0 * a0 + a1 * a1);
    if (vn > vmax) { a0 *= vmax / vn; a1 *= vmax / vn; }
    e0 = a0 * ascale + anoise * n0;
    e1 = a1 * ascale + anoise * n1;
    w = 0.0f;
}

// the believed frame is meant to be world-aligned; odo_th is its accumulated
// error, so it IS the true orientation of the command/sensor frame
inline float kin_frame(float th, float odo_th) { return odo_th; }

inline void kin_reset(float u, thread float& th, thread float& odo_th) {
    th = 0.0f;
    odo_th = 0.0f;
}

inline void kin_expert(float ax, float ay, float speed, float th,
                       float beta, float wmax, float tgain,
                       float prev0, float prev1,
                       thread float& a0, thread float& a1) {
    a0 = prev0 + beta * (ax - prev0);
    a1 = prev1 + beta * (ay - prev1);
}
"""

_DIFFDRIVE_METAL = """
constant int KIN_NU = 5;  // extra uniform: random episode-start heading

inline void kin_execute(float a0, float a1, float vmax, float wmax,
                        float ascale, float anoise, float n0, float n1,
                        thread float& e0, thread float& e1, thread float& w) {
    e0 = metal::clamp(a0, -vmax, vmax) * ascale + anoise * n0;
    e1 = 0.0f;  // no lateral motion
    // yaw noise scaled so noise/limit matches the linear axis
    w = metal::clamp(a1, -wmax, wmax) * ascale + anoise * (wmax / vmax) * n1;
}

// commands execute exactly in the true body frame; the believed heading
// (odom theta) drifting away from it is what corrupts the observations
inline float kin_frame(float th, float odo_th) { return th; }

inline void kin_reset(float u, thread float& th, thread float& odo_th) {
    th = 6.2831853f * u - 3.14159265f;
    odo_th = th;  // believed heading starts exact, drifts within the episode
}

inline void kin_expert(float ax, float ay, float speed, float th,
                       float beta, float wmax, float tgain,
                       float prev0, float prev1,
                       thread float& a0, thread float& a1) {
    float a = metal::atan2(ay, ax) - th;  // heading error toward desired direction
    a = a - 6.2831853f * metal::floor((a + 3.14159265f) / 6.2831853f);  // [-pi, pi)
    a1 = metal::clamp(tgain * a, -wmax, wmax);  // steer at the desired direction
    // cos^4 gate: hard slowdown while misaligned tightens turn arcs around
    // corners (un-smoothed on purpose -- lagged speed is what cuts corners)
    float c = metal::max(metal::cos(a), 0.0f);
    a0 = speed * c * c * c * c;
}
"""


_DIFFDRIVE_VEL_METAL = """
constant int KIN_NU = 5;  // extra uniform: random episode-start heading
// steering controller constants -- part of the drive definition, matched to
// the diffdrive expert's validated conversion (P-gain 4, cos^4 speed gate)
constant float KIN_TGAIN = 4.0f;

inline void kin_execute(float a0, float a1, float vmax, float wmax,
                        float ascale, float anoise, float n0, float n1,
                        thread float& e0, thread float& e1, thread float& w) {
    // command = body-frame desired velocity, norm-clamped like holonomic
    float vn = metal::sqrt(a0 * a0 + a1 * a1);
    if (vn > vmax) { a0 *= vmax / vn; a1 *= vmax / vn; vn = vmax; }
    // P-steering: heading error is the command's body angle directly
    float alpha = metal::atan2(a1, a0);
    // turn-direction convention: near straight-backward the command's tiny
    // y-component decides left vs right, so policy approximation noise flips
    // the turn every step (measured 52% flip rate vs the expert's 0%). Always
    // turning left in the backward cone moves the decision boundary to -150
    // degrees, where the regression target is large and rarely mispredicted.
    if (alpha < -2.618f) alpha += 6.2831853f;
    float wz = metal::clamp(KIN_TGAIN * alpha, -wmax, wmax);
    float c = metal::max(metal::cos(alpha), 0.0f);
    float v = vn * c * c * c * c;
    // actuation noise applies to the executed (v, omega), as in diffdrive
    e0 = v * ascale + anoise * n0;
    e1 = 0.0f;
    w = wz * ascale + anoise * (wmax / vmax) * n1;
}

inline float kin_frame(float th, float odo_th) { return th; }

inline void kin_reset(float u, thread float& th, thread float& odo_th) {
    th = 6.2831853f * u - 3.14159265f;
    odo_th = th;
}

// expert command = the desired world velocity rotated into the body frame;
// the controller in kin_execute handles smoothing-free conversion
inline void kin_expert(float ax, float ay, float speed, float th,
                       float beta, float wmax, float tgain,
                       float prev0, float prev1,
                       thread float& a0, thread float& a1) {
    float c = metal::cos(th), s = metal::sin(th);
    a0 = c * ax + s * ay;
    a1 = -s * ax + c * ay;
}
"""


class Holonomic:
    name = "holonomic"
    metal = _HOLONOMIC_METAL
    n_uniform = 4
    # world rotated by any angle is the same task: rotation augmentation valid
    rotation_augment = True

    @staticmethod
    def action_scale(cfg) -> np.ndarray:
        return np.array([cfg.v_max, cfg.v_max], dtype=np.float32)

    @staticmethod
    def clamp(act: mx.array, cfg) -> mx.array:
        norm = mx.maximum(mx.sqrt(mx.sum(mx.square(act), axis=-1, keepdims=True)), 1e-6)
        return act * mx.minimum(1.0, cfg.v_max / norm)

    @staticmethod
    def speed(act: mx.array, cfg) -> mx.array:
        """Normalized linear speed in [0, 1] after the sim's clamp."""
        a = Holonomic.clamp(act, cfg)
        return mx.sqrt(mx.sum(mx.square(a), axis=-1)) / cfg.v_max

    @staticmethod
    def rel_goal(goal: mx.array, odom: mx.array) -> mx.array:
        """Goal vector in the observation frame (believed world frame)."""
        return goal - odom[:, :2]


class DiffDrive:
    name = "diffdrive"
    metal = _DIFFDRIVE_METAL
    n_uniform = 5
    # observations are body-frame, which already factors out world rotation;
    # only the reflection part of the dihedral augmentation applies
    rotation_augment = False

    @staticmethod
    def action_scale(cfg) -> np.ndarray:
        return np.array([cfg.v_max, cfg.w_max], dtype=np.float32)

    @staticmethod
    def clamp(act: mx.array, cfg) -> mx.array:
        lim = mx.array([cfg.v_max, cfg.w_max])
        return mx.clip(act, -lim, lim)

    @staticmethod
    def speed(act: mx.array, cfg) -> mx.array:
        return mx.minimum(mx.abs(act[..., 0]), cfg.v_max) / cfg.v_max

    @staticmethod
    def rel_goal(goal: mx.array, odom: mx.array) -> mx.array:
        """Goal vector rotated into the believed body frame."""
        r = goal - odom[:, :2]
        c, s = mx.cos(odom[:, 2]), mx.sin(odom[:, 2])
        return mx.stack([c * r[:, 0] + s * r[:, 1], -s * r[:, 0] + c * r[:, 1]], axis=1)


class DiffDriveVel:
    """Diff drive commanded by body-frame desired velocity through a fixed
    P-steering controller (see _DIFFDRIVE_VEL_METAL). Learns like holonomic,
    deploys like diffdrive (cmd_vel + steering shim)."""

    name = "diffdrive_vel"
    metal = _DIFFDRIVE_VEL_METAL
    n_uniform = 5
    rotation_augment = False  # body-frame observations

    action_scale = Holonomic.action_scale
    clamp = Holonomic.clamp
    speed = Holonomic.speed  # |desired| / v_max: upper bound on executed speed
    rel_goal = DiffDrive.rel_goal


KINEMATICS = {k.name: k for k in (Holonomic, DiffDrive, DiffDriveVel)}


def get(name: str):
    if name not in KINEMATICS:
        raise KeyError(f"unknown kinematics {name!r}; available: {sorted(KINEMATICS)}")
    return KINEMATICS[name]
