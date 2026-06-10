"""Selectively download ProcTHOR scenes from hssd/ai2thor-hab (disk-friendly).

Downloads scene_instance.json configs first, then only the stage GLBs, object
configs, and object GLBs those scenes actually reference.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "hssd/ai2thor-hab"
ROOT = "ai2thor-hab"


def get(path: str, local_dir: str, retries: int = 5) -> Path:
    import time

    for attempt in range(retries):
        try:
            return Path(hf_hub_download(REPO, path, repo_type="dataset", local_dir=local_dir))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="+", default=["1", "2", "3", "4", "5"],
                    help="room-count buckets (1-9, a=10, b=11, c=12)")
    ap.add_argument("--per-room", type=int, default=8)
    ap.add_argument("--split", default="Train", choices=["Train", "Val", "Test"])
    ap.add_argument("--out", default="data/ai2thor-hab")
    args = ap.parse_args()

    files = list_repo_files(REPO, repo_type="dataset")
    picked = []
    for room in args.rooms:
        prefix = f"{ROOT}/configs/scenes/ProcTHOR/{room}/"
        cands = sorted(f for f in files if f.startswith(prefix) and f"-{args.split}-" in f)
        if not cands:
            cands = sorted(f for f in files if f.startswith(prefix))
        picked.extend(cands[: args.per_room])
    print(f"selected {len(picked)} scene configs")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    fileset = set(files)
    stages, objects = set(), set()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for fut in as_completed([pool.submit(get, c, args.out) for c in picked]):
            cfg = json.loads(fut.result().read_text())
            st = cfg["stage_instance"]["template_name"]  # 'stages/ProcTHOR/1/X'
            stages.add(st.split("stages/")[-1])
            for inst in cfg.get("object_instances", []):
                objects.add(inst["template_name"].split("/")[-1])

    print(f"unique stages: {len(stages)}, unique objects: {len(objects)}", flush=True)
    todo = []
    n_missing = 0
    for st in sorted(stages):
        todo += [f"{ROOT}/configs/stages/{st}.stage_config.json", f"{ROOT}/assets/stages/{st}.glb"]
    for ob in sorted(objects):
        cfg_p = f"{ROOT}/configs/objects/{ob}.object_config.json"
        glb_p = f"{ROOT}/assets/objects/{ob}.glb"
        if cfg_p not in fileset or glb_p not in fileset:
            n_missing += 1
            continue
        todo += [cfg_p, glb_p]

    done = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = [pool.submit(get, p, args.out) for p in todo]
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(todo)} files", flush=True)
    print(f"done: {len(todo)} files ({n_missing} objects had no asset in repo)", flush=True)


if __name__ == "__main__":
    main()
