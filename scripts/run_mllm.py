"""Run a spatial MLLM on VSI-Bench with its own confidence (GPU), for the comparison in Table 3.

Same uniformly sampled frames as the measurement pipeline, official VSI-Bench answer prompts, greedy decoding. Saved per
question: answer, option probabilities (multiple choice) or mean token log-likelihood (numbers).

python scripts/run_mllm.py --model Qwen/Qwen3-VL-8B-Instruct --frames 16 --out results/mllm/qwen3vl8b_16f/run.jsonl
python scripts/run_mllm.py --model Qwen/Qwen3-VL-8B-Instruct --frames 32 --part 0/4 --out results/mllm/qwen3vl8b_32f/shard0.jsonl
python scripts/run_mllm.py --model rayruiyang/VST-7B-RL --frames 16 --out results/mllm/vst7b_16f/run.jsonl
"""
import argparse, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from measure_twice.utils import WORK, hf_token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--frames", type=int, default=16); ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1", help="k/n scene shard"); ap.add_argument("--quant", default="none")
    ap.add_argument("--degrade", default="", help="e.g. lowlight:2, defocus:1, codec:3 (see measure_twice.degradations)")
    ap.add_argument("--budget_h", type=float, default=11.0)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    tok = hf_token(); snapshot_download(a.model, token=tok)
    Q = [json.loads(l) for l in open(os.path.join(ROOT, "measure_twice", "data", "vsi_test.txt"))]
    Q = [q for q in Q if q["question_type"] != "route_planning"]
    k, n = map(int, a.part.split("/"))
    scenes = sorted({(q["dataset"], q["scene_name"]) for q in Q}); mine = set(scenes[k::n])
    vroot = os.path.join(WORK, "vsi")
    for sd in sorted({d for d, _ in mine}):
        z = hf_hub_download("nyu-visionx/VSI-Bench", f"{sd}.zip", repo_type="dataset", token=tok, local_dir=os.path.join(WORK, "vsi_zip"))
        os.makedirs(os.path.join(vroot, sd), exist_ok=True)
        subprocess.run(f'cd "{os.path.join(vroot, sd)}" && unzip -q -o "{z}"', shell=True, check=False)
    ids = os.path.join(WORK, f"ids_{k}_{n}.json")
    json.dump([q["id"] for q in Q if (q["dataset"], q["scene_name"]) in mine], open(ids, "w"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cmd = [sys.executable, "-m", "measure_twice.mllm_worker", "--model", a.model, "--root", vroot, "--ids", ids, "--out", a.out,
           "--frames", str(a.frames), "--quant", a.quant, "--budget_h", str(a.budget_h)] + (["--degrade", a.degrade] if a.degrade else [])
    subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
                                              PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))


if __name__ == "__main__":
    main()
