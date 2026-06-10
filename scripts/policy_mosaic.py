"""Mosaic video of a trained policy (feedforward, recurrent, or PPO checkpoint)
on chosen scenes; --failures hunts and replays its failure cases."""

import argparse

import mlx.core as mx

from fastnav.policy import NavPolicy
from fastnav.scene import ScenePack
from fastnav.sim import SimConfig
from fastnav.videos import policy_mosaic_video


def load_policy(path: str, cfg: SimConfig):
    """Build the right policy class from the checkpoint's keys."""
    keys = set(mx.load(path).keys())
    if "log_std" in keys:
        from fastnav.ppo import PPONavPolicy
        policy = PPONavPolicy(cfg)
    elif any(k.startswith("gru.") for k in keys):
        from fastnav.policy import RecurrentNavPolicy
        policy = RecurrentNavPolicy(cfg)
    else:
        policy = NavPolicy(cfg)
    policy.load_weights(path)
    mx.eval(policy.parameters())
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--include", nargs="+", default=["Baked_sc2_*", "Baked_sc3_*"])
    ap.add_argument("--checkpoint", default="checkpoints/ppo/policy_best.safetensors")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--collisions-only", action="store_true")
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--out", default="policy_mosaic.mp4")
    args = ap.parse_args()

    pack = ScenePack.load_dir(args.scenes, include=args.include)
    cfg = SimConfig()
    policy = load_policy(args.checkpoint, cfg)
    path = policy_mosaic_video(pack, policy, cfg=cfg, failures=args.failures,
                               collisions_only=args.collisions_only,
                               n_tiles=args.tiles, cols=args.cols, frames=args.frames,
                               out_path=args.out)
    print(f"wrote {path}" if path else "no failures found")


if __name__ == "__main__":
    main()
