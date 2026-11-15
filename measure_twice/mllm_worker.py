"""Open spatial MLLMs on VSI-Bench with their own confidence, for comparison and deferral against geometric answers.

Same 32 uniformly sampled frames as the geometric pipeline, passed as images. Official VSI-Bench answer prompts:
  numerical  "Please answer the question using a single word or phrase."
  choice     "Answer with the option's letter from the given choices directly."
Saved per question (JSONL): generated text, parsed answer, option-letter probabilities (choice), mean token log-prob of the
generated answer (numerical), number of prompt tokens.

python -m measure_twice.mllm_worker --model Qwen/Qwen3-VL-8B-Instruct --root data/vsi --ids ids.json --out out.jsonl --budget_h 5
"""
import argparse, glob, json, os, re, time
import numpy as np
import torch
from .utils import log, WORK
from .scene_memory import frames_of, resize

NUM = ("object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation")


def prompt(q):
    if q["question_type"] in NUM:
        return q["question"] + "\nPlease answer the question using a single word or phrase."
    return q["question"] + "\nOptions:\n" + "\n".join(q["options"]) + "\nAnswer with the option's letter from the given choices directly."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--root", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--budget_h", type=float, default=5.0)
    ap.add_argument("--frames", type=int, default=16); ap.add_argument("--quant", default="none", help="none = fp16 over all GPUs (paper); 4bit = NF4 on one GPU"); ap.add_argument("--degrade", default=""); ap.add_argument("--max_pixels", type=int, default=448 * 336)
    a = ap.parse_args()
    t0 = time.time()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.model, max_pixels=a.max_pixels, min_pixels=128 * 128)
    kw = dict(dtype=torch.float16, device_map={"": 0})
    if a.quant == "4bit":                             # one model per T4: NF4 weights, fp16 compute
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    elif a.quant == "none":
        kw["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(a.model, **kw).eval()
    log("loaded", a.model, a.quant, f"{torch.cuda.memory_allocated() / 1e9:.1f} GB")
    tok = proc.tokenizer
    letter_ids = {L: tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"}
    Q = {q["id"]: q for q in (json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "data", "vsi_test.txt")))}
    ids = json.load(open(a.ids))
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    by_scene = {}
    for i in ids:
        if i not in done:
            q = Q[i]; by_scene.setdefault((q["dataset"], q["scene_name"]), []).append(q)
    n = 0
    with open(a.out, "a") as fo:
        for (ds, sc), qs in by_scene.items():
            if time.time() - t0 > a.budget_h * 3600:
                log("budget reached"); break
            vids = glob.glob(f"{a.root}/**/{sc}.mp4", recursive=True)
            if not vids:
                log("no video", sc); continue
            ims = [im for _, im in frames_of(vids[0], a.frames)]
            if a.degrade:                                   # same physical degradation as the geometric pipeline
                from . import degradations as DG
                from PIL import Image
                kind, lvl = a.degrade.split(":")
                ims = [Image.fromarray(x) for x in DG.apply(kind, [np.asarray(resize(im)) for im in ims], int(lvl), tmp=os.path.join(WORK, f"codec_{sc}"))]
            for q in qs:
                g = None
                for use in (ims, ims[::2]):                    # on out-of-memory, retry with half the frames
                    try:
                        msgs = [{"role": "user", "content": [{"type": "image"} for _ in use] + [{"type": "text", "text": prompt(q)}]}]
                        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        inp = proc(text=[text], images=use, return_tensors="pt").to(model.device)
                        with torch.no_grad():
                            g = model.generate(**inp, max_new_tokens=16, do_sample=False, output_scores=True, return_dict_in_generate=True)
                        break
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache(); g = None
                if g is None:
                    log("OOM twice", q["id"]); continue
                tq = time.time()
                new = g.sequences[0, inp["input_ids"].shape[1]:]
                ans = tok.decode(new, skip_special_tokens=True).strip()
                lp = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(g.scores, new) if t not in tok.all_special_ids]
                rec = dict(id=q["id"], type=q["question_type"], gt=q["ground_truth"], cond=a.degrade or "clean", text=ans, n_prompt=int(inp["input_ids"].shape[1]), n_frames=len(use),
                           mean_logprob=float(np.mean(lp)) if lp else None)
                if q["question_type"] in NUM:
                    m = re.search(r"-?\d+(\.\d+)?", ans.replace(",", ""))
                    rec["pred"] = float(m.group(0)) if m else None
                else:
                    p = torch.softmax(g.scores[0][0].float(), -1)
                    probs = {L: float(p[i]) for L, i in letter_ids.items()}
                    k = len(q["options"])
                    sub = {L: probs[L] for L in "ABCD"[:k]}; z = sum(sub.values()) or 1.0
                    rec["probs"] = {L: v / z for L, v in sub.items()}
                    m = re.match(r"\s*\(?([A-D])", ans)
                    rec["pred"] = m.group(1) if m else max(sub, key=sub.get)
                fo.write(json.dumps(rec) + "\n"); fo.flush(); n += 1
            log(f"{ds}/{sc}: {len(qs)} questions, total {n}, {(time.time() - t0) / 60:.0f} min")
    log("done", n, "questions")


if __name__ == "__main__":
    main()
