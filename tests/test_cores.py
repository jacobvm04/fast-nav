"""Memory-core invariants: the deploy step path (KV ring / carried hidden) must
reproduce the training unroll, and zeroing the flat state must be a cold start.
Run on both cores so any new core inherits the same contract."""

import mlx.core as mx
import numpy as np

from fastnav.policy import CORES, RecurrentNavPolicy
from fastnav.sim import SimConfig


def stepped(policy, obs_seq, done_seq):
    """Sequential step() with the standard reset rule (mask_state after done)."""
    b, t_len, _ = obs_seq.shape
    state = policy.new_state(b)
    feats = []
    for t in range(t_len):
        if t > 0:
            state = policy.mask_state(state, 1.0 - done_seq[:, t - 1])
        f, state = policy._step_feature(obs_seq[:, t], state)
        feats.append(f)
    return mx.stack(feats, axis=1), state


def unrolled(policy, obs_seq, done_seq):
    b = obs_seq.shape[0]
    return policy.features(obs_seq, mx.zeros((b, policy.h0_size)), done_seq)[0]


def check_core(core: str, core_opts: dict, atol: float):
    mx.random.seed(3)
    cfg = SimConfig(kinematics="diffdrive", n_rays=16)
    policy = RecurrentNavPolicy(cfg, hidden=32, head="discrete_w", core=core,
                                core_opts=core_opts)
    mx.eval(policy.parameters())
    b, t_len = 8, 50
    obs_seq = mx.random.uniform(shape=(b, t_len, cfg.obs_dim + 2)) * 4.0
    done_seq = (mx.random.uniform(shape=(b, t_len, 1)) < 0.06).astype(mx.float32)

    # 1. step path == unroll path (same dones, same window)
    f_step, state = stepped(policy, obs_seq, done_seq)
    f_unroll = unrolled(policy, obs_seq, done_seq)
    d = float(mx.abs(f_step - f_unroll).max())
    assert d < atol, f"{core}: step/unroll diverge: {d}"

    # 2. zeroing the state == cold start: continuing after a wipe must equal a
    # fresh policy run on the suffix alone
    f2, _ = stepped(policy, obs_seq, mx.zeros_like(done_seq))
    cut = t_len // 2
    state = policy.new_state(b)
    for t in range(cut):
        _, state = policy._step_feature(obs_seq[:, t], state)
    state = policy.mask_state(state, mx.zeros((b, 1)))  # the wipe
    feats = []
    for t in range(cut, t_len):
        f, state = policy._step_feature(obs_seq[:, t], state)
        feats.append(f)
    fresh, _ = stepped(policy, obs_seq[:, cut:], mx.zeros((b, t_len - cut, 1)))
    d = float(mx.abs(mx.stack(feats, axis=1) - fresh).max())
    assert d < atol, f"{core}: zeroed state is not a cold start: {d}"

    # 3. done inside the sequence == concatenation independence: tokens after a
    # done must not see anything before it
    done_mid = mx.zeros((b, t_len, 1))
    done_mid[:, cut - 1] = 1.0
    f_joint = unrolled(policy, obs_seq, done_mid)
    f_suffix = unrolled(policy, obs_seq[:, cut:], mx.zeros((b, t_len - cut, 1)))
    d = float(mx.abs(f_joint[:, cut:] - f_suffix).max())
    assert d < atol, f"{core}: attention/state leaks across episode boundary: {d}"
    print(f"{core}{core_opts}: step/unroll parity, cold reset, boundary isolation OK")


if __name__ == "__main__":
    check_core("gru", {}, atol=1e-6)
    # transformer tolerance is numerical, not logical: the deploy ring is fp16
    # while the training unroll is fp32 (indexing/masking bugs show as O(1)
    # diffs, far above it). Window < sequence exercises ring wrap + unroll mask.
    check_core("transformer", {"context": 16, "layers": 2, "heads": 4}, atol=0.05)
    check_core("transformer", {"context": 64, "layers": 3, "heads": 4}, atol=0.05)
    print("all core checks passed")
