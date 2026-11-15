"""Run the measurement pipeline on VSI-Bench videos (GPU), then answer the questions from the saved 3D memory (CPU).

  1. download one VSI-Bench shard (arkitscenes | scannet | scannetpp) from the Hugging Face hub
  2. measure_twice.scene_memory: 32 frames per video, Grounding DINO + SAM masks, every reconstructor on all frames and
     on two random 16-frame halves -> one .npz per scene in $MT_OUT/vsi/   (one worker per visible GPU)
  3. measure_twice.programs: geometric programs answer every question in every run -> answers JSON

The paper used two NVIDIA T4 GPUs (16 GB each). Long ARKitScenes videos can exceed T4 memory for pi3-X; those runs are
skipped and the questions drop out of the complete-output set, exactly as reported in the paper.

python scripts/run_pipeline.py --shard scannet --models mapanything,pi3x,da3m --limit 2      # smoke test
python scripts/run_pipeline.py --shard scannet --models mapanything,pi3x,da3m                 # full shard
"""
import argparse, glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from measure_twice.utils import log, WORK, OUT, hf_token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, choices=["arkitscenes", "scannet", "scannetpp"])
    ap.add_argument("--models", default="mapanything,pi3x,da3m")
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes (smoke test)")
    ap.add_argument("--part", default="0/1", help="k/n: take scenes k, k+n, ... (split a shard across machines)")
    ap.add_argument("--budget_h", type=float, default=11.0, help="stop starting new scenes after this many hours")
    ap.add_argument("--gpus", type=int, default=0, help="number of workers (default: all visible GPUs)")
    a = ap.parse_args()
    import torch
    from huggingface_hub import hf_hub_download
    n_gpu = a.gpus or max(torch.cuda.device_count(), 1)
    tok = hf_token()
    z = hf_hub_download("nyu-visionx/VSI-Bench", f"{a.shard}.zip", repo_type="dataset", token=tok, local_dir=os.path.join(WORK, "vsi_zip"))
    vroot = os.path.join(WORK, "vsi", a.shard); os.makedirs(vroot, exist_ok=True)
    subprocess.run(f'cd "{vroot}" && unzip -q -o "{z}"', shell=True, check=False)
    cats = json.load(open(os.path.join(ROOT, "measure_twice", "data", "vsi_cats.json")))
    k, n = map(int, a.part.split("/"))
    scenes = sorted(s for s in cats if s.startswith(a.shard + "/"))[k::n]
    scenes = scenes[: a.limit] if a.limit else scenes
    log(f"{a.shard} part {a.part}: {len(scenes)} scenes on {n_gpu} worker(s), models {a.models}")
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    ps = []
    for g in range(n_gpu):
        mine = scenes[g::n_gpu]
        if not mine:
            continue
        cmd = [sys.executable, "-m", "measure_twice.scene_memory", "--root", vroot, "--models", a.models,
               "--budget_h", str(a.budget_h), "--scenes", *mine]
        ps.append(subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=str(g)), stdout=open(os.path.join(OUT, f"worker{g}.log"), "w"),
                                   stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in ps):
        time.sleep(60)
        log("scenes saved:", len(glob.glob(os.path.join(OUT, "vsi", "*.npz"))))
    out = os.path.join(OUT, f"answers_{a.shard}.json")
    subprocess.run([sys.executable, "-m", "measure_twice.programs", "--npz", os.path.join(OUT, "vsi"),
                    "--questions", os.path.join(ROOT, "measure_twice", "data", "vsi_test.txt"), "--out", out], env=env, check=True)
    log("answers ->", out)


if __name__ == "__main__":
    main()
