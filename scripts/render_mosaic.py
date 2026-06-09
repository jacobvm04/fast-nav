"""Render a mosaic of expert-driven envs to mp4 (or a live window with --live)."""

import argparse

import cv2
import numpy as np

from fastnav.render import MosaicRenderer
from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--out", default="mosaic.mp4")
    ap.add_argument("--envs", type=int, default=24, help="tiles in the mosaic")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--ego", action="store_true", help="pair each tile with the egocentric obs view")
    args = ap.parse_args()

    pack = ScenePack.load_dir(args.scenes)
    # env i is pinned to scene i % S, so the first `envs` ids cycle through scenes
    sim = Sim(pack, num_envs=max(args.envs, 64), cfg=SimConfig(), seed=0)
    sim.reset()
    ren = MosaicRenderer(sim, list(range(args.envs)), cols=args.cols)

    writer = None
    if not args.live:
        four = cv2.VideoWriter_fourcc(*"mp4v")
        first = ren.frame(np.array(sim.pos), np.array(sim.goal), np.array(sim.lidar), np.array(sim.scene), ego=args.ego)
        writer = cv2.VideoWriter(args.out, four, args.fps, (first.shape[1], first.shape[0]))

    for t in range(args.frames):
        sim.step(sim.expert_actions())
        img = ren.frame(np.array(sim.pos), np.array(sim.goal), np.array(sim.lidar), np.array(sim.scene), ego=args.ego)
        if args.live:
            cv2.imshow("fast-nav expert mosaic", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            writer.write(img)
    if writer is not None:
        writer.release()
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
