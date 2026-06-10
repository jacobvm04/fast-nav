"""Expand a trained policy's lidar input from R to 2R rays by weight surgery.

New ray j sits at angle 2*pi*j/(2R): even j coincides with old ray j/2, odd j
falls halfway between old rays j//2 and j//2+1 — its encoder column is their
mean. All lidar columns are halved so the summed activation (twice as many
rays covering the same field) matches the original network's statistics.
The policy's obs-normalization vector (_scale) is dropped; the rebuilt policy
constructs the right-sized one.
"""

import argparse

import mlx.core as mx
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="checkpoints/ppo_big/policy_best.safetensors")
    ap.add_argument("--out", default="checkpoints/ppo_big_128init.safetensors")
    ap.add_argument("--rays", type=int, default=64, help="source ray count")
    args = ap.parse_args()

    w = dict(mx.load(args.src).items())
    enc = np.array(w["enc.weight"])  # [enc_out, R + tail]
    r = args.rays
    tail = enc[:, r:]
    lidar = enc[:, :r]

    cols = []
    for j in range(2 * r):
        if j % 2 == 0:
            c = lidar[:, j // 2]
        else:
            c = 0.5 * (lidar[:, j // 2] + lidar[:, (j // 2 + 1) % r])
        cols.append(0.5 * c)
    new_enc = np.concatenate([np.stack(cols, axis=1), tail], axis=1).astype(np.float32)

    w["enc.weight"] = mx.array(new_enc)
    mx.save_safetensors(args.out, w)
    print(f"{args.src} [{enc.shape}] -> {args.out} [{new_enc.shape}]")


if __name__ == "__main__":
    main()
