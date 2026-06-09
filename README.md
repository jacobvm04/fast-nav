# fast-nav

Ultra-fast playground for indoor robot point-navigation with 2D lidar, built for RL/BC
research where rollout throughput is the bottleneck. Runs entirely on Apple-silicon GPU
via MLX custom Metal kernels.

**Measured on M2 Max (12-core, 32GB), 6 ReplicaCAD apartment scenes, 64 lidar rays:**

| num_envs | random actions | expert policy | expert + obs→numpy |
|---------:|---------------:|--------------:|-------------------:|
| 4 096    | 11.4M steps/s  | 10.4M         | 9.8M               |
| 16 384   | 26.4M steps/s  | 23.6M         | 20.0M              |
| 32 768   | 33.6M steps/s  | 31.2M         | 24.5M              |

Through the PufferLib env wrapper (obs written into pufferlib numpy buffers every step):
~9.7M steps/s at 8 192 envs. Expert success rate: 100.00% over 431k episodes across all
six scenes (zero timeouts).

## How it's fast

- **No 3D at runtime.** Each scene mesh is sliced once at the robot's height band
  (0.10–1.30 m), rasterized to a 2.5 cm occupancy grid, and converted to a signed
  **Euclidean distance field (EDF)**. 2D lidar = sphere tracing through the EDF
  (~15–30 bilinear samples per ray); collision = one EDF lookup + gradient projection.
- **Whole batch per dispatch.** Three Metal kernels per step (expert, step+auto-reset,
  lidar at N×R threads), built once via `mx.fast.metal_kernel`. Python overhead is
  amortized over 8k–32k envs; arrays stay resident in unified memory.
- **Zero per-episode planning.** Per scene, K=16 goals get precomputed
  **clearance-weighted geodesic distance fields** (Dijkstra, offline). The expert is
  smoothed gradient descent on the field — fully batched, naturally wall-avoiding,
  curved paths. The same fields provide episode sampling by difficulty (start tables
  sorted by geodesic distance) and, later, oracle distance for reward shaping.

## Robot model

Holonomic kinematic point (radius 0.18 m), no orientation state, continuous world-frame
velocity command (≤1.5 m/s), dt = 0.1 s. Lidar rays are fixed in the world frame.
Observation = `[lidar (R) | goal − pos (2) | pos (2)]` — ground-truth odometry given;
the learning problem is obstacle avoidance + navigation, not state estimation.

## Usage

```bash
uv sync
uv run python scripts/preprocess_replicacad.py   # GLB -> data/scenes/*.npz (+ debug PNGs)
uv run python tests/test_correctness.py          # lidar vs numpy ref, collision, expert
uv run python scripts/bench.py --scenes data/scenes
uv run python scripts/collect_demos.py --envs 2048 --steps 1024   # 2M transitions in <1s
uv run python scripts/render_mosaic.py --live    # live expert mosaic (or --out x.mp4)
```

Assets (6 baked apartment variations, ~56 MB, no auth):

```bash
uv run hf download ai-habitat/ReplicaCAD_baked_lighting --repo-type dataset \
  --include "stages_uncompressed/Baked_sc0_staging_0[0-2].glb" \
            "stages_uncompressed/Baked_sc[1-3]_staging_00.glb" \
            "configs/scenes/*.scene_instance.json" "urdf_uncompressed/**" \
  --local-dir data/replica_cad_baked
```

More scenes: the repo has 84 variations (`Baked_sc{0-3}_staging_{00-20}`); download more
and rerun the preprocess script. Doors are treated as open (door leaves skipped);
articulated furniture (fridge, kitchen counter, cabinets) is composited closed from the
URDFs.

## Layout

```
fastnav/scene.py        occupancy -> EDF -> geodesic fields -> episode tables; ScenePack
fastnav/sim.py          Metal kernels + batched Sim (step / lidar / expert)
fastnav/render.py       mosaic renderer (viz-only)
fastnav/puffer_env.py   PufferLib native vectorized env (rewards = 0, by design for now)
scripts/                preprocess, bench, collect_demos, render_mosaic
tests/                  kernel correctness vs numpy reference
```

## Next steps (not yet built)

- Reward design + PPO via pufferlib (3.0 trainer takes the env instance directly;
  note pufferlib 4.0 dropped the Python env API, so we pin 3.0.0)
- BC on `data/demos.npz`, DAgger (expert is queryable every step at full speed)
- Cross-scene generalization splits (train sc0–sc2 variants, eval sc3)
- Discrete-action wrapper if continuous PPO is finicky
