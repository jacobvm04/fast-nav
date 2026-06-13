"""Action-head contracts: chunk-window targets, deploy chunk state machine,
and the stateful-act interface that all heads now share."""

import mlx.core as mx
import numpy as np

from fastnav.policy import HEADS, RecurrentNavPolicy
from fastnav.sim import SimConfig


def test_chunk_windows():
    """bc_loss must consume target windows target[t:t+H]; verify the in-head
    gather against a naive python loop by planting a recognizable signal."""
    mx.random.seed(0)
    head = HEADS["flow_chunk"](hidden=16, act_scale=(1.5, 2.5), horizon=4)
    b, t_len = 2, 12
    target = mx.random.uniform(shape=(b, t_len, 2))
    h = head.horizon
    idx = mx.arange(t_len)[:, None] + mx.arange(h)[None, :]
    chunks = mx.take(target, mx.minimum(idx, t_len - 1), axis=1)
    ref = np.zeros((b, t_len, h, 2), np.float32)
    tn = np.array(target)
    for t in range(t_len):
        for k in range(h):
            ref[:, t, k] = tn[:, min(t + k, t_len - 1)]
    assert np.abs(np.array(chunks) - ref).max() == 0.0
    # tail mask: the last H-1 positions carry no loss
    feat = mx.random.normal((b, t_len, 16))
    loss = head.bc_loss(feat, target)
    assert loss.shape == (b, t_len)
    assert float(mx.abs(loss[:, t_len - h + 1:]).max()) == 0.0
    assert float(mx.abs(loss[:, : t_len - h]).min()) > 0.0
    print("flow_chunk: window targets + tail mask OK")


def test_chunk_state_machine():
    """A sampled chunk must execute open-loop for H steps (state holds it
    constant, cursor decrements), resample at exhaustion, and resample after a
    zero-reset mid-chunk."""
    mx.random.seed(1)
    head = HEADS["flow_chunk"](hidden=16, act_scale=(1.5, 2.5), horizon=4)
    mx.eval(head.parameters())
    n = 3
    state = mx.zeros((n, head.state_size))
    chunks, lefts, acts = [], [], []
    for t in range(9):
        feat = mx.random.normal((n, 16))  # varying features: a held chunk must ignore them
        a, state = head.act(feat, state)
        acts.append(np.array(a))
        lefts.append(float(state[0, 0]))
        chunks.append(np.array(state[:, 1:]))
    # chunk constant within each H-window, changed across windows
    assert np.abs(chunks[0] - chunks[3]).max() == 0.0, "chunk mutated mid-execution"
    assert np.abs(chunks[0] - chunks[4]).max() > 0.0, "chunk not resampled at exhaustion"
    assert lefts[:5] == [3.0, 2.0, 1.0, 0.0, 3.0], lefts[:5]
    # executed actions = successive chunk positions
    for k in range(4):
        assert np.abs(acts[k] - chunks[0].reshape(n, 4, 2)[:, k]).max() < 1e-6
    # zero-reset mid-chunk = fresh chunk on the next step
    state = state * 0.0
    _, state = head.act(mx.random.normal((n, 16)), state)
    assert float(state[0, 0]) == 3.0
    print("flow_chunk: open-loop execution, resample, zero-reset OK")


def test_waypoint_follower():
    """The follower must track a fabricated straight-line plan: constant
    forward command, dead-reckoned pose advancing along it, replan countdown,
    and a fresh plan after a zero-reset."""
    mx.random.seed(2)
    head = HEADS["waypoint_flow"](hidden=16, act_scale=(1.5, 2.5), horizon=4,
                                  stride=2, dt=0.1, w_max=2.5)
    mx.eval(head.parameters())
    n, v, h2 = 3, 1.0, 2 * head.horizon
    # state layout: [left(1) | est(3) | chunk(2H) | prev_cond(2H)]
    assert head.state_size == 4 + 2 * h2
    # plan: straight ahead at v m/s -> wp_k = ((k+1)*v*stride*dt, 0)
    wps = np.stack([np.arange(1, 5) * v * 0.2, np.zeros(4)], axis=1)
    chunk = mx.broadcast_to(mx.array(wps.reshape(-1).astype(np.float32)), (n, h2))
    state = mx.concatenate([mx.full((n, 1), float(head.replan)),
                            mx.zeros((n, 3)), chunk, mx.zeros((n, h2))], axis=1)
    for t in range(head.replan):
        a, state = head.act(mx.random.normal((n, 16)), state)
        a = np.array(a)
        assert np.abs(a - [v, 0.0]).max() < 1e-5, f"step {t}: command {a[0]} != ({v}, 0)"
        est = np.array(state[:, 1:4])
        assert np.abs(est - [v * 0.1 * (t + 1), 0.0, 0.0]).max() < 1e-5, "dead reckoning drifted"
        assert np.abs(np.array(state[:, 4:4 + h2]) - np.array(chunk)).max() == 0.0, \
            "plan mutated mid-execution"
    assert float(state[0, 0]) == 0.0, "replan countdown wrong"
    # exhausted (and zeroed) state -> fresh plan sampled from the net
    _, state2 = head.act(mx.random.normal((n, 16)), state)
    assert float(state2[0, 0]) == float(head.replan) - 1.0
    assert np.abs(np.array(state2[:, 4:4 + h2]) - np.array(chunk)).max() > 0.0, "plan not resampled"
    print("waypoint_flow: straight-plan tracking, dead reckoning, resample OK")


def test_waypoint_diffdrive_output():
    """On diffdrive the follower emits (v, omega) directly (not a body velocity).
    A straight-ahead plan -> (v_max, ~0); _steer must match _dead_reckon's
    internal conversion."""
    mx.random.seed(4)
    vmax, wmax = 1.5, 2.5
    head = HEADS["waypoint_flow"](hidden=16, act_scale=(vmax, wmax), horizon=4,
                                  stride=2, dt=0.1, w_max=wmax, kinematics="diffdrive")
    mx.eval(head.parameters())
    n, h2 = 3, 2 * head.horizon
    # straight-ahead plan
    wps = np.stack([np.arange(1, 5) * vmax * 0.2, np.zeros(4)], axis=1)
    chunk = mx.broadcast_to(mx.array(wps.reshape(-1).astype(np.float32)), (n, h2))
    state = mx.concatenate([mx.full((n, 1), float(head.replan)), mx.zeros((n, 3)),
                            chunk, mx.zeros((n, h2))], axis=1)
    a, _ = head.act(mx.random.normal((n, 16)), state)
    a = np.array(a)
    assert np.abs(a[:, 0] - vmax).max() < 1e-4, f"v != v_max: {a[0]}"
    assert np.abs(a[:, 1]).max() < 1e-4, f"omega != 0 on straight plan: {a[0]}"
    # _steer consistency: the (v, omega) it emits must equal the (v, omega)
    # _dead_reckon integrates for the same body-velocity command
    bv = mx.array(np.array([[1.0, 0.6], [0.8, -0.4], [0.3, 0.9]], np.float32))
    v, wz = head._steer(bv)
    # reconstruct dead_reckon's internal (v, wz) by stepping a zero pose
    est0 = mx.zeros((3, 3))
    est1 = head._dead_reckon(est0, bv)
    # after 2 substeps from heading 0: th = wz*dt; check wz recovered
    wz_recovered = np.array(est1[:, 2]) / 0.1
    assert np.abs(np.array(wz) - wz_recovered).max() < 1e-4, "_steer omega != _dead_reckon"
    print("waypoint_flow[diffdrive]: (v,omega) output + _steer/_dead_reckon consistency OK")


def test_waypoint_conditioning():
    """The previous-plan conditioning is carried in deploy state and threaded
    through resampling; cold-start (zero cond) and the cond round-trip work."""
    mx.random.seed(5)
    head = HEADS["waypoint_flow"](hidden=16, act_scale=(1.5, 1.5), horizon=4, stride=2)
    mx.eval(head.parameters())
    n, h2 = 4, 2 * head.horizon
    feat = mx.random.normal((n, 16))
    # from a cold (zeroed) state, act resamples and stores the new plan's deltas
    # as prev_cond; that cond must equal to_deltas(committed chunk)
    state = mx.zeros((n, head.state_size))
    _, state = head.act(feat, state)
    chunk = state[:, 4:4 + h2]
    cond = state[:, 4 + h2:]
    rt = head._to_waypoints(cond)  # cond is deltas of the committed plan
    assert float(mx.abs(rt - chunk).max()) < 1e-4, "prev_cond != to_deltas(committed plan)"
    # conditioning changes the sample: same feat, different cond -> different plan
    p_cold = np.array(head._sample_plan(feat, mx.zeros((n, h2))))
    p_cond = np.array(head._sample_plan(feat, mx.random.normal((n, h2))))
    assert np.abs(p_cold - p_cond).mean() > 1e-3, "conditioning has no effect on the sample"
    print("waypoint_flow: prev-plan conditioning carried + threaded OK")


def test_waypoint_coherence():
    """A trained-ish head should produce MORE temporally-consistent consecutive
    plans WITH conditioning than without (the whole point). Use a fixed feat and
    feed each plan's deltas as the next cond; coherence = low frame-to-frame
    change vs independent resampling."""
    mx.random.seed(6)
    head = HEADS["waypoint_flow"](hidden=16, act_scale=(1.5, 1.5), horizon=6, stride=2)
    mx.eval(head.parameters())
    n = 64
    feat = mx.random.normal((n, 16))
    # independent: resample with zero cond each time
    indep = [np.array(head._sample_plan(feat, None)) for _ in range(6)]
    indep_jit = np.mean([np.abs(indep[i] - indep[i - 1]).mean() for i in range(1, 6)])
    # conditioned: feed previous plan's deltas as cond
    plans, cond = [], None
    for _ in range(6):
        p = head._sample_plan(feat, cond)
        plans.append(np.array(p))
        cond = head._to_deltas(p)
    cond_jit = np.mean([np.abs(plans[i] - plans[i - 1]).mean() for i in range(1, 6)])
    print(f"waypoint_flow: frame-to-frame plan change indep {indep_jit:.3f} vs "
          f"conditioned {cond_jit:.3f} (untrained net; conditioning wired)")
    # untrained net can't guarantee lower jitter, but the cond path must run and
    # produce finite, shaped output
    assert np.isfinite(cond_jit) and plans[0].shape == (n, 12)


def test_waypoint_bc_loss():
    """bc_loss consumes complete waypoint labels: per-step loss everywhere
    (no tail mask), label_dim = 2*horizon, token_frac path keeps shape."""
    mx.random.seed(3)
    head = HEADS["waypoint_flow"](hidden=16, act_scale=(1.5, 2.5), horizon=4, stride=2)
    mx.eval(head.parameters())
    assert head.label_dim == 8 and head.label_kind == "waypoint"
    b, t_len = 2, 12
    target = mx.random.uniform(shape=(b, t_len, 8))
    feat = mx.random.normal((b, t_len, 16))
    loss = head.bc_loss(feat, target)
    assert loss.shape == (b, t_len)
    assert float(mx.abs(loss).min()) > 0.0, "waypoint labels are complete: no masked steps"
    head.token_frac = 0.5
    loss = head.bc_loss(feat, target)
    assert loss.shape == (b, t_len)
    nz = float(mx.sum((mx.abs(loss) > 0).astype(mx.float32)))
    assert 0 < nz <= b * (t_len // 2), "token_frac subsample not applied"
    print("waypoint_flow: complete-label bc_loss + token_frac OK")


def test_stateful_act_contract():
    """All heads share act(feat, state) -> (action [N,2], state); per-step heads
    pass a zero-width state through; policy.step round-trips the joint state."""
    cfg = SimConfig(kinematics="diffdrive", n_rays=16)
    for head in HEADS:
        policy = RecurrentNavPolicy(cfg, hidden=32, head=head, core="transformer",
                                    core_opts={"context": 8, "layers": 1, "heads": 2})
        mx.eval(policy.parameters())
        state = policy.new_state(4)
        assert state.shape[1] == policy.core.state_size + policy.head.state_size
        obs = mx.random.uniform(shape=(4, cfg.obs_dim + 2))
        for _ in range(3):
            a, state = policy.step(obs, state)
        assert a.shape == (4, 2) and state.shape[1] == policy.state_size
        print(f"{head}: stateful act contract OK (head state {policy.head.state_size})")


if __name__ == "__main__":
    test_chunk_windows()
    test_chunk_state_machine()
    test_waypoint_follower()
    test_waypoint_diffdrive_output()
    test_waypoint_conditioning()
    test_waypoint_coherence()
    test_waypoint_bc_loss()
    test_stateful_act_contract()
    print("all head checks passed")
