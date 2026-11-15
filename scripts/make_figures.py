"""Paper figures, all computed from the canonical answers and measure_twice.calibration (no hand-entered numbers).
  figs/teaser.pdf        a real VSI-Bench scene: one question the runs agree on (answered), one they do not (withheld)
  figs/signals.pdf       which signal knows the error: rank correlation of each uncertainty signal with the error, per task
  figs/risk_coverage.pdf score of the answered questions as the least certain ones are withheld

python scripts/make_figures.py   (teaser only if VSI-Bench frames are available, see README)
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from measure_twice.calibration import table, mra, choice, NUM, conformal

OUT = os.path.join(ROOT, "figures")
FAMS = ["mapanything", "pi3x"]
# validated categorical slots (light surface); fixed assignment across all figures
OURS, SINGLE, NATIVE, DET = "#2a78d6", "#1baf7a", "#eb6834", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
J = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
ANS = J("vsi_answers.json")
if not os.path.exists(os.path.join(ROOT, "results", "analysis", "two_models.json")):
    sys.exit("run `python scripts/reproduce_paper.py` first (it writes results/analysis/two_models.json)")
# 32 sampled frames of scannet/scene0356_00 (written by measure_twice.scene_memory); not redistributed
FRAMES = os.environ.get("MT_TEASER_FRAMES", os.path.join(ROOT, "outputs", "frames", "scannet__scene0356_00"))


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def teaser():
    """scannet/scene0356_00: 'table-bed' (runs agree, answered) and 'radiator-bed' (runs disagree, withheld)."""
    t = "object_abs_distance"; eps = NUM[t]; g = lambda x: np.log(np.asarray(x, float) + eps)
    T = table([r for r in ANS if r["type"] == t], t, FAMS)
    # instantiate one split of the main analysis (same seed and random stream as measure_twice.calibration): the first split in which
    # this scene is a test scene. sigma0, q and the answer threshold come only from that split's other scenes.
    splits = []; conformal(T, t, "comb", thresholds=(1.0, 2.0, 3.0), keep=splits)
    k_split, sp = next((i, x) for i, x in enumerate(splits) if "scannet/scene0356_00" in x["test"])
    s0, q = sp["s0"], sp["q"]
    tau = float(np.median([r["sig"]["comb"] for r in T if r["scene"] in sp["dev"]]))   # answer the most certain half
    print("teaser: split", k_split, "s0 %.3f q %.3f answer if sigma <= %.3f" % (s0, q, tau))
    z = np.load(os.path.join(ROOT, "results/V02/out/vsi/scannet__scene0356_00.npz"), allow_pickle=True)
    cats = [str(c) for c in z["cats"]]
    cases = [(4650, 19, ("table", "bed"), "Distance between the table and the bed?"),
             (4652, 24, ("radiator", "bed"), "Distance between the radiator and the bed?")]
    fig = plt.figure(figsize=(6.6, 2.35))
    for row, (qid, fr, objs, text) in enumerate(cases):
        r = next(x for x in T if x["id"] == qid)
        a = next(x for x in ANS if x["id"] == qid)
        ax = fig.add_axes([0.0, 0.52 - 0.5 * row, 0.24, 0.46])
        ax.imshow(Image.open(os.path.join(FRAMES, f"{fr:02d}.jpg")))
        for f, c, s, b in zip(z["det_frame"], z["det_cat"], z["det_score"], z["det_box"]):
            if f == fr and cats[c] in objs and s > 0.45:
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="white", lw=1.3))
                ax.text(b[0] + 4, b[1] + 16, cats[c], color="white", fontsize=6.5, bbox=dict(fc=INK, ec="none", pad=0.8))
        ax.axis("off")
        ax2 = fig.add_axes([0.30, 0.60 - 0.5 * row, 0.44, 0.30])
        runs = [(k, v) for k, v in sorted(a["runs"].items()) if k.startswith(tuple(FAMS)) and v is not None]
        lo = max(np.exp(g(r["point"]) - q * (r["sig"]["comb"] + s0)) - eps, 0.0)
        hi = np.exp(g(r["point"]) + q * (r["sig"]["comb"] + s0)) - eps
        ax2.add_patch(Rectangle((max(lo, 0.02), -0.5), hi - max(lo, 0.02), 1.0, color=OURS, alpha=0.18, lw=0))
        if lo < 0.02:                            # interval continues below the axis: mark the clipping
            ax2.annotate("", xy=(0.02, 0), xytext=(0.032, 0), arrowprops=dict(arrowstyle="->", color=OURS, lw=1.0))
        for i, (k, v) in enumerate(runs):
            ma = k.startswith("mapanything")
            ax2.plot(max(v, 0.02), 0.25 if ma else -0.25, "o" if ma else "^", ms=4.5, mfc=INK2, mec="white", mew=0.6, zorder=3)
        ax2.plot(r["point"], 0, "D", ms=5, color=INK, zorder=4)
        ax2.plot(r["gt"], 0, "*", ms=9, mfc="white", mec=INK, mew=0.9, zorder=5)
        ax2.set_xscale("log"); ax2.set_xlim(0.02, 8); ax2.set_ylim(-0.7, 0.7); ax2.set_yticks([])
        ax2.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4], ["0.05", "0.1", "0.25", "0.5", "1", "2", "4 m"], fontsize=6.5)
        ax2.minorticks_off(); clean(ax2); ax2.spines["left"].set_visible(False)
        ax2.set_title(f"“{text}”", fontsize=7.5, loc="left", color=INK, pad=3)
        verdict = "ANSWER" if r["sig"]["comb"] <= tau else "WITHHOLD"
        m = mra(np.array([r["point"]]), np.array([r["gt"]]))[0]
        txt = (f"{verdict}\n{r['point']:.2f} m, interval [{lo:.2f}, {hi:.2f}] m\n"
               f"truth {r['gt']:.1f} m, run spread $\\sigma$={r['sig']['comb']:.2f}")
        fig.text(0.758, 0.75 - 0.5 * row, txt, fontsize=7, va="center", color=INK,
                 bbox=dict(fc="white", ec=OURS if verdict == "ANSWER" else INK2, lw=1.0, boxstyle="round,pad=0.4"))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", mfc=INK2, mec="white", ms=5, label="MapAnything runs"),
         Line2D([], [], ls="", marker="^", mfc=INK2, mec="white", ms=5, label="$\\pi^3$-X runs"),
         Line2D([], [], ls="", marker="D", color=INK, ms=4.5, label="median answer"),
         Line2D([], [], ls="", marker="*", mfc="white", mec=INK, ms=8, label="ground truth"),
         Rectangle((0, 0), 1, 1, color=OURS, alpha=0.18, label="calibrated 90% interval")]
    fig.legend(handles=h, ncol=5, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.52, -0.12), columnspacing=1.0, handletextpad=0.3)
    fig.savefig(f"{OUT}/teaser.pdf", bbox_inches="tight", pad_inches=0.08); plt.close(fig)


def signals():
    A = J(os.path.join("analysis", "two_models.json"))
    tasks = [("object_abs_distance", "Abs. distance"), ("object_size_estimation", "Object size"),
             ("room_size_estimation", "Room area"), ("object_counting", "Counting")]
    rho = lambda t, s: A[t]["signals"][s]["sel"]["spearman"]
    series = [("Detector score", DET, [rho(t, "detector") if "detector" in A[t]["signals"] else np.nan for t, _ in tasks]),
              ("Native 3D confidence", NATIVE, [rho(t, "native:mean") for t, _ in tasks]),
              ("Frame resampling, one model", SINGLE, [np.mean([rho(t, f"fam:{f}") for f in FAMS]) for t, _ in tasks]),
              ("Frame resampling, both models", OURS, [rho(t, "comb") for t, _ in tasks])]
    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    w = 0.19; x = np.arange(len(tasks))
    for k, (name, col, v) in enumerate(series):
        xs = x + (k - 1.5) * (w + 0.02)
        ax.bar(xs, v, w, color=col, label=name, zorder=2)
        for xi, vi in zip(xs, v):
            if np.isfinite(vi):
                ax.text(xi, vi + (0.012 if vi >= 0 else -0.012), f"{vi:.2f}".replace("0.", ".").replace("-.", "−."),
                        ha="center", va="bottom" if vi >= 0 else "top", fontsize=5.3, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x, [n for _, n in tasks]); ax.set_ylabel("rank correlation with the error", fontsize=6.8)
    ax.set_ylim(-0.2, 0.62); ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True); clean(ax)
    ax.legend(ncol=2, fontsize=5.8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.8)
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/signals.pdf"); plt.close(fig)


def curve(score, conf, rng, reps=20):
    cs = []
    for _ in range(reps):
        o = np.lexsort((rng.random(len(conf)), -conf)); cs.append(np.cumsum(score[o]) / np.arange(1, len(score) + 1))
    return np.arange(1, len(score) + 1) / len(score), np.mean(cs, 0)


def risk_coverage():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.9))
    for ax, (t, name) in zip(axes[:2], (("object_abs_distance", "Abs. distance"), ("object_size_estimation", "Object size"))):
        T = table([r for r in ANS if r["type"] == t], t, FAMS)
        m = mra(np.array([r["point"] for r in T]), np.array([r["gt"] for r in T]))
        for lab, sig, col, ls, lw in (("frame resampling, both models (ours)", "comb", OURS, "-", 1.5),
                                       ("native 3D confidence", "native:mean", NATIVE, "-", 1.0),
                                       ("detector score", "detector", DET, "-", 1.0)):
            cov, c = curve(m, -np.array([r["sig"][sig] for r in T]), rng); ax.plot(cov * 100, c * 100, color=col, ls=ls, lw=lw, label=lab)
        cov, c = curve(m, m, rng); ax.plot(cov * 100, c * 100, color=INK2, ls=":", lw=1.0, label="oracle")
        ax.axhline(m.mean() * 100, color=INK2, ls="--", lw=0.8, label="random order")
        ax.set_title(f"{name}, MRA ({len(T)} questions)", fontsize=7.3, color=INK, pad=3)
    R = [r for r in ANS if r["type"].startswith("object_rel_direction")]
    acc, conf = [], []
    for r in R:
        v = [x for k, x in r["runs"].items() if x is not None and k.startswith(tuple(FAMS))]
        n = sum(1 for k in r["runs"] if k.startswith(tuple(FAMS)))
        if not v:
            acc.append(0.0); conf.append(0.0); continue
        vals, cnt = np.unique(v, return_counts=True); acc.append(float(vals[np.argmax(cnt)] == r["gt"])); conf.append(cnt.max() / n)
    acc, conf = np.array(acc), np.array(conf)
    cov, c = curve(acc, conf, rng); axes[2].plot(cov * 100, c * 100, color=OURS, lw=1.5)
    cov, c = curve(acc, acc, rng); axes[2].plot(cov * 100, c * 100, color=INK2, ls=":", lw=1.0)
    axes[2].axhline(acc.mean() * 100, color=INK2, ls="--", lw=0.8)
    axes[2].set_title(f"Rel. direction, accuracy (all {len(R)})", fontsize=7.3, color=INK, pad=3)
    for ax in axes:
        ax.set_xlabel("questions answered (%)", fontsize=6.8); ax.set_xlim(5, 100); ax.set_ylim(0, 102)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); clean(ax)
    axes[0].set_ylabel("score of answered (%)", fontsize=6.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=6.2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03), handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0, 1, 0.88)); fig.savefig(f"{OUT}/risk_coverage.pdf"); plt.close(fig)


def evidence():
    """One figure for the main evidence: (a) which signal knows the error, (b) selective answering for distances,
    (c) the spatial MLLM's own confidence against the measured vote on the same relative-direction questions."""
    import glob as _g
    from measure_twice.mllm_compare import load_mllm, mllm_conf, score as mscore
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.05), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    A = J(os.path.join("analysis", "two_models.json"))
    tasks = [("object_abs_distance", "Distance"), ("object_size_estimation", "Size"), ("room_size_estimation", "Room"), ("object_counting", "Count")]
    rho = lambda t, s: A[t]["signals"][s]["sel"]["spearman"]
    series = [("detector score", DET, [rho(t, "detector") if "detector" in A[t]["signals"] else np.nan for t, _ in tasks]),
              ("native 3D confidence", NATIVE, [rho(t, "native:mean") for t, _ in tasks]),
              ("resampling, one model", SINGLE, [np.mean([rho(t, f"fam:{f}") for f in FAMS]) for t, _ in tasks]),
              ("resampling, both (ours)", OURS, [rho(t, "comb") for t, _ in tasks])]
    ax = axes[0]; w = 0.19; x = np.arange(len(tasks))
    for k, (name, col, v) in enumerate(series):
        ax.bar(x + (k - 1.5) * (w + 0.02), v, w, color=col, label=name, zorder=2)
    ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(x, [n for _, n in tasks], fontsize=6.8)
    ax.set_ylabel("rank corr. with error", fontsize=6.8); ax.set_ylim(-0.2, 0.6)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True); clean(ax)
    ax.legend(ncol=2, fontsize=5.4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.98), handlelength=0.9, columnspacing=0.6, handletextpad=0.3)
    ax.set_title("(a) which signal knows the error", fontsize=7.3, color=INK, pad=17)
    # (b) distance selective answering
    ax = axes[1]; t = "object_abs_distance"
    T = table([r for r in ANS if r["type"] == t], t, FAMS)
    m = mra(np.array([r["point"] for r in T]), np.array([r["gt"] for r in T]))
    for sig, col, lw in (("comb", OURS, 1.5), ("native:mean", NATIVE, 1.0), ("detector", DET, 1.0)):
        cov, c = curve(m, -np.array([r["sig"][sig] for r in T]), rng); ax.plot(cov * 100, c * 100, color=col, lw=lw)
    cov, c = curve(m, m, rng); ax.plot(cov * 100, c * 100, color=INK2, ls=":", lw=1.0, label="oracle")
    ax.axhline(m.mean() * 100, color=INK2, ls="--", lw=0.8, label="random order")
    ax.set_title(f"(b) distance, {len(T)} questions", fontsize=7.3, color=INK, pad=3); ax.set_ylabel("MRA of answered (%)", fontsize=6.8)
    ax.legend(fontsize=5.6, frameon=False, loc="lower left", handlelength=1.4)
    # (c) relative direction: MLLM probability vs measured vote, same questions
    ax = axes[2]
    M = load_mllm(sorted(_g.glob(os.path.join(ROOT, "results", "mllm", "qwen3vl8b_16f", "*.jsonl"))))
    gs, gc, qs, qc = [], [], [], []
    for r in ANS:
        if r["type"].startswith("object_rel_direction") and r["id"] in M:
            v = [x for k, x in r["runs"].items() if x is not None and k.startswith(tuple(FAMS))]
            if not v:
                continue
            vals, cnt = np.unique(v, return_counts=True)
            gs.append(float(vals[np.argmax(cnt)] == r["gt"])); gc.append(cnt.max() / sum(1 for k in r["runs"] if k.startswith(tuple(FAMS))))
            q = M[r["id"]]; qs.append(mscore(q["type"], q.get("pred"), q["gt"])); qc.append(mllm_conf(q))
    for s_, c_, col, lab in ((gs, gc, OURS, "measured, vote share"), (qs, qc, NATIVE, "Qwen3-VL, own probability")):
        s_ = np.array(s_); cov, c = curve(s_, np.array(c_), rng); ax.plot(cov * 100, c * 100, color=col, lw=1.4 if col == OURS else 1.1, label=lab)
        ax.axhline(s_.mean() * 100, color=col, ls="--", lw=0.7)
    ax.set_title(f"(c) rel. direction, {len(gs)} questions", fontsize=7.3, color=INK, pad=3); ax.set_ylabel("accuracy of answered (%)", fontsize=6.8)
    ax.legend(fontsize=5.6, frameon=False, loc="lower center", handlelength=1.4)
    for ax in axes[1:]:
        ax.set_xlabel("questions answered (%)", fontsize=6.8); ax.set_xlim(5, 100); ax.set_ylim(20, 102)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); clean(ax)
    fig.tight_layout(pad=0.3, w_pad=0.8); fig.savefig(f"{OUT}/evidence.pdf"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if os.path.isdir(FRAMES):
        teaser()
    else:
        print("teaser skipped: no frames at", FRAMES)
    evidence(); print("figures ->", OUT)
