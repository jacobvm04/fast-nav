"""Grouped bar chart: expert-policy throughput + GPU utilization vs batch size.

GPU saturation is sampled sudo-free from IOAccelerator's "Device Utilization %"
(ioreg) on a background thread while the benchmark loop runs.
"""

import argparse
import re
import subprocess
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from fastnav.scene import ScenePack
from fastnav.sim import Sim, SimConfig

_UTIL_RE = re.compile(r'"Device Utilization %"=(\d+)')


class GpuUtilSampler:
    """Polls GPU busy % via ioreg until stopped."""

    def __init__(self, period: float = 0.15):
        self.period = period
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                                 capture_output=True, text=True).stdout
            m = _UTIL_RE.search(out)
            if m:
                self.samples.append(int(m.group(1)))
            self._stop.wait(self.period)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")


def bench_expert(pack: ScenePack, num_envs: int, duration: float = 4.0, warmup: float = 1.0):
    sim = Sim(pack, num_envs=num_envs, cfg=SimConfig())
    sim.reset()
    t_end = time.perf_counter() + warmup
    while time.perf_counter() < t_end:
        obs, term, _ = sim.step(sim.expert_actions())
        mx.eval(obs, term)
    mx.synchronize()
    mx.reset_peak_memory()
    steps = 0
    with GpuUtilSampler() as util:
        t0 = time.perf_counter()
        t_end = t0 + duration
        while time.perf_counter() < t_end:
            obs, term, _ = sim.step(sim.expert_actions())
            mx.eval(obs, term)
            steps += 1
        mx.synchronize()
        dt = time.perf_counter() - t0
    peak_gb = mx.get_peak_memory() / 1024**3
    return steps * num_envs / dt, peak_gb, util.mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--out", default="throughput.png")
    ap.add_argument("--envs", type=int, nargs="+",
                    default=[1024, 4096, 16384, 65536, 131072, 262144,
                             524288, 1048576, 2097152, 4194304])
    args = ap.parse_args()

    pack = ScenePack.load_dir(args.scenes)
    sps, mem, util = [], [], []
    for n in args.envs:
        s, m, u = bench_expert(pack, n)
        sps.append(s)
        mem.append(m)
        util.append(u)
        print(f"{n:>8} envs: {s:>12,.0f} steps/s  peak {m:.2f} GB  gpu {u:.0f}%")

    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=150)
    ax2 = ax.twinx()
    x = np.arange(len(args.envs))
    bw = 0.38
    b1 = ax.bar(x - bw / 2, np.array(sps) / 1e6, width=bw, color="#3a7bd5",
                zorder=3, label="throughput (M steps/s)")
    b2 = ax2.bar(x + bw / 2, util, width=bw, color="#f0a35e",
                 zorder=3, label="GPU utilization (%)")
    ymax = ax.get_ylim()[1]
    for b, m in zip(b1, mem):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.015,
                f"{m:.2f}\nGB", ha="center", va="bottom", fontsize=7.5, color="#2a5ba8")
    for b, u in zip(b2, util):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                 f"{u:.0f}%", ha="center", va="bottom", fontsize=7.5, color="#b06a1e")
    ax.set_xticks(x, [f"{n // 1024}k" if n >= 1024 else str(n) for n in args.envs])
    ax.set_xlabel("batch size (parallel envs)")
    ax.set_ylabel("env steps / second (millions)", color="#2a5ba8")
    ax2.set_ylabel("GPU utilization %", color="#b06a1e")
    ax2.set_ylim(0, 119)
    ax.margins(y=0.14)
    ax.set_title("fast-nav: throughput vs GPU saturation by batch size\n"
                 "M2 Max · 6 ReplicaCAD scenes · 64 rays · expert policy · "
                 "labels: peak memory / mean GPU busy")
    ax.yaxis.grid(True, color="#dddddd", zorder=0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
