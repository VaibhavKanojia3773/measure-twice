"""Reproduce every table of the paper on a CPU from the released per-run answers (no GPU, no video download).

  results/vsi_answers.json             9 runs per question (3 reconstructors x {all 32 frames, two random halves of 16})
  results/vsi_answers_degraded.json    the same pipeline on 25 ScanNet++ scenes under low light, defocus and H.264
  results/mllm/*/                      Qwen3-VL-8B and VST-7B-RL answers with their own confidence

Writes results/analysis/*.json and prints the headline numbers. Takes about 20-40 minutes (100 scene splits per signal).

python scripts/reproduce_paper.py            # everything
python scripts/reproduce_paper.py --quick    # Table 1 distance row only (a few minutes)
"""
import argparse, glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "results", "analysis")
PY = sys.executable


def run(args):
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    ans = os.path.join(RES, "vsi_answers.json")
    if a.quick:
        run([PY, "-m", "measure_twice.calibration", "--answers", ans, "--families", "mapanything,pi3x",
             "--tasks", "object_abs_distance", "--out", os.path.join(OUT, "quick_distance.json")])
        return
    # Tables 1 and 2: uncertainty signals with two reconstructors, and the three-reconstruction signal ablation
    run([PY, "-m", "measure_twice.calibration", "--answers", ans, "--families", "mapanything,pi3x", "--out", os.path.join(OUT, "two_models.json")])
    run([PY, "-m", "measure_twice.calibration", "--answers", ans, "--families", "mapanything,pi3x,da3m", "--out", os.path.join(OUT, "three_models.json")])
    # Table 3: spatial MLLMs next to the measurements (16 frames, and 32 frames for Qwen3-VL when available)
    for name, d in (("qwen3vl8b_16f", "qwen3vl8b_16f"), ("vst7b_16f", "vst7b_16f"), ("qwen3vl8b_32f", "qwen3vl8b_32f")):
        files = sorted(glob.glob(os.path.join(RES, "mllm", d, "*.jsonl")))
        if files:
            run([PY, "-m", "measure_twice.mllm_compare", "--mllm", *files, "--answers", ans, "--out", os.path.join(OUT, f"mllm_{name}.json")])
    # Table 4: physical degradation
    run([PY, "-m", "measure_twice.degradation_eval", "--clean", ans, "--degraded", os.path.join(RES, "vsi_answers_degraded.json"),
         "--out", os.path.join(OUT, "degradation.json")])
    # robustness analyses (complete-output subset, interval widths, own-answer ablation)
    run([PY, os.path.join("scripts", "robustness.py")])
    summary()


def summary():
    J = lambda f: json.load(open(os.path.join(OUT, f)))
    A = J("two_models.json"); d = A["object_abs_distance"]["signals"]
    print("\n================ headline numbers ================")
    print(f"distance common set: {A['object_abs_distance']['n_common']} questions")
    print(f"{'signal':26s} {'rho':>5s} {'cov%':>6s} {'width m':>8s} {'AURC':>6s}")
    for s, nm in (("const", "constant log-width"), ("detector", "detector score"), ("native:mean", "native 3D confidence"),
                  ("between", "model disagreement"), ("fam:mapanything", "resampling, MapAnything"), ("fam:pi3x", "resampling, pi3-X"),
                  ("comb", "resampling, both (ours)")):
        c, se = d[s]["conf"], d[s]["sel"]
        print(f"{nm:26s} {se['spearman']:5.2f} {100 * c['cov']:6.1f} {c['width']:8.2f} {se['aurc']:6.3f}")
    c = d["comb"]["conf"]
    print(f"\nthreshold decisions (1/2/3 m): settled {100 * c['dec_frac']:.1f}% of pairs with {100 * c['dec_err']:.1f}% error; "
          f"matched control {100 * c['ctl_err']:.1f}%; point estimate on all pairs {100 * c['pt_err']:.1f}%")
    if os.path.exists(os.path.join(OUT, "mllm_qwen3vl8b_16f.json")):
        M = J("mllm_qwen3vl8b_16f.json")["tasks"]["object_abs_distance"]
        g = lambda o, k: 100 * (o[f"matched_{k}_random"] - o[f"matched_{k}_aurc"])
        print(f"distance ranking gain: Qwen3-VL own confidence {g(M, 'mllm'):.1f}, measured spread {g(M, 'geo'):.1f}")


if __name__ == "__main__":
    main()
