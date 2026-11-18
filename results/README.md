# Released measurements

Everything needed to reproduce the paper's tables on a CPU. Question ids, types and ground truth follow
[VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) (test split).

## `vsi_answers.json`

One record per answerable VSI-Bench question (route planning excluded), 4,936 in total.

```json
{"id": 0, "type": "object_counting", "ds": "arkitscenes", "scene": "41069025", "cond": "clean", "gt": "4",
 "runs":   {"mapanything_all": 4.0, "mapanything_s0": 4.0, "mapanything_s1": 4.0, "pi3x_all": 4.0, ...},
 "native": {"mapanything_all": 1.0, ...},
 "det":    {"mapanything_all": 0.73, ...}}
```

| field | meaning |
|---|---|
| `runs` | the answer of every run: `<model>_<frames>` with model in `mapanything`, `pi3x`, `da3m` and frames `all` (32 frames), `s0`, `s1` (two random halves of 16). `null` if the run failed (no detection, out of memory). |
| `native` | the reconstructor's own median per-point confidence on the queried objects (the *native 3D confidence* baseline) |
| `det` | mean Grounding DINO score of the queried objects (the *detector score* baseline) |

Numbers are in metres (distance), centimetres (size), square metres (room area) or counts; choices are option letters.

## `vsi_answers_degraded.json`

The same format for 25 ScanNet++ scenes under low light (1/32 exposure with shot and read noise), defocus (3 px) and
H.264 (QP 37); `cond` names the degradation.

## `mllm/`

JSONL, one line per question:
`{"id", "type", "gt", "text", "pred", "n_frames", "n_prompt", "mean_logprob" (numbers) | "probs" (choices)}`.

| folder | model | frames |
|---|---|---|
| `qwen3vl8b_16f/` | Qwen3-VL-8B-Instruct, fp16 | 16 |
| `vst7b_16f/` | VST-7B-RL, fp16 | 16 |
| `*_degraded/` | the same models on the degraded videos | 16 |

`part1.jsonl` is the half of the benchmark evaluated in the paper's Table 3 (2,007 questions with a measured answer);
`part2.jsonl` completes the benchmark. The measurements themselves use 32 frames.
