"""Throughput benchmark: env steps/sec across batch sizes and modes.

Modes:
  random    pure sim (step + lidar), random actions precomputed
  expert    expert kernel + sim step
  expert+np expert + sim + obs copied to numpy each step (training-loop realistic)
"""

import argparse
import time

import mlx.core as mx
import numpy as np

from fastnav.scene import FieldConfig, Scene, ScenePack, build_scene, make_synthetic_occupancy
from fastnav.sim import Sim, SimConfig


def load_pack(scene_dir: str | None) -> ScenePack:
    if scene_dir:
        return ScenePack.load_dir(scene_dir)
    occ, origin = make_synthetic_occupancy(seed=3)
    return ScenePack([build_scene("synth0", occ, origin, FieldConfig())])


def bench(sim: Sim, mode: str, n_steps: int, warmup: int = 50) -> float:
    sim.reset()
    if mode == "random":
        acts = [mx.random.uniform(low=-1.5, high=1.5, shape=(sim.num_envs, 2)) for _ in range(32)]
        mx.eval(*acts)

    def one(t):
        if mode == "random":
            a = acts[t % 32]
        else:
            a = sim.expert_actions()
        obs, term, trunc = sim.step(a)
        if mode == "expert+np":
            return np.array(obs), np.array(term), np.array(trunc)
        mx.eval(obs, term)
        return None

    for t in range(warmup):
        one(t)
    mx.synchronize()
    t0 = time.perf_counter()
    for t in range(n_steps):
        one(t)
    mx.synchronize()
    dt = time.perf_counter() - t0
    return n_steps * sim.num_envs / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None, help="dir of preprocessed scene .npz (default: synthetic)")
    ap.add_argument("--rays", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--envs", type=int, nargs="+", default=[1024, 4096, 8192, 16384, 32768])
    args = ap.parse_args()

    pack = load_pack(args.scenes)
    print(f"scenes: {pack.names}  grid {pack.grid_hw}  geo {pack.geo_hw}  rays={args.rays}")
    print(f"{'num_envs':>9} {'random':>14} {'expert':>14} {'expert+np':>14}   (env steps/sec)")
    for n in args.envs:
        cfg = SimConfig(n_rays=args.rays)
        sim = Sim(pack, num_envs=n, cfg=cfg)
        row = [f"{n:>9}"]
        for mode in ("random", "expert", "expert+np"):
            sps = bench(sim, mode, args.steps)
            row.append(f"{sps:>13,.0f}".replace(",", "_"))
        print(" ".join(row))


if __name__ == "__main__":
    main()
