"""Collect expert demonstrations for behavior cloning.

Stores transposed-to-env-major arrays so episodes are easy to slice:
  obs      [N, T, obs_dim] float16   (lidar | rel_goal | pos)
  actions  [N, T, 2]       float16   expert velocity commands
  done     [N, T]          bool      episode boundary AFTER this transition
  scene    [N]             int16
"""

import argparse
import time

import numpy as np

from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--out", default="data/demos.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pack = ScenePack.load_dir(args.scenes)
    sim = Sim(pack, num_envs=args.envs, cfg=SimConfig(), seed=args.seed)
    sim.reset()

    n, t, d = args.envs, args.steps, sim.cfg.obs_dim
    obs_buf = np.empty((t, n, d), dtype=np.float16)
    act_buf = np.empty((t, n, 2), dtype=np.float16)
    done_buf = np.empty((t, n), dtype=bool)

    t0 = time.perf_counter()
    obs = sim.obs()
    for i in range(t):
        a = sim.expert_actions()
        obs_buf[i] = np.array(obs, copy=False).astype(np.float16)
        act_buf[i] = np.array(a, copy=False).astype(np.float16)
        obs, term, trunc = sim.step(a)
        done_buf[i] = np.array(term, copy=False) | np.array(trunc, copy=False)
    dt = time.perf_counter() - t0

    eps = int(done_buf.sum())
    print(f"collected {n * t:_} transitions ({eps:_} episodes) in {dt:.1f}s "
          f"({n * t / dt:,.0f} steps/s incl. host copies)")
    np.savez_compressed(
        args.out,
        obs=obs_buf.transpose(1, 0, 2),
        actions=act_buf.transpose(1, 0, 2),
        done=done_buf.T,
        scene=np.array(sim.scene).astype(np.int16),
        scene_names=np.array(pack.names),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
