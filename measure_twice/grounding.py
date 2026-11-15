"""Open-vocabulary object masks: Grounding DINO boxes (IDEA-Research/grounding-dino-base) refined by SAM (facebook/sam-vit-large)."""
import numpy as np
import torch
from .utils import log


class Grounder:
    def __init__(self, device="cuda", box_thr=0.30, text_thr=0.25):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
        self.dev, self.box_thr, self.text_thr = device, box_thr, text_thr
        self.dp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.dm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device).eval()
        self.sp = SamProcessor.from_pretrained("facebook/sam-vit-large")
        self.sm = SamModel.from_pretrained("facebook/sam-vit-large").to(device).eval()

    @torch.no_grad()
    def detect(self, img, cats):
        """img: PIL RGB; cats: list of category strings. Returns [(cat, score, box xyxy, mask HxW bool)]."""
        out = []
        for i in range(0, len(cats), 8):                    # Grounding DINO's text length limit: query in groups
            group = cats[i:i + 8]
            text = ". ".join(c.lower() for c in group) + "."
            inp = self.dp(images=img, text=text, return_tensors="pt").to(self.dev)
            o = self.dm(**inp)
            kw = dict(target_sizes=[img.size[::-1]], text_threshold=self.text_thr)
            try:
                res = self.dp.post_process_grounded_object_detection(o, inp.input_ids, threshold=self.box_thr, **kw)[0]
            except TypeError:
                res = self.dp.post_process_grounded_object_detection(o, inp.input_ids, box_threshold=self.box_thr, **kw)[0]
            labels = res.get("text_labels", res.get("labels"))
            for lab, sc, bx in zip(labels, res["scores"].tolist(), res["boxes"].tolist()):
                lab = lab if isinstance(lab, str) else str(lab)
                cat = _match(lab, group)
                if cat is not None:
                    out.append([cat, float(sc), [float(v) for v in bx]])
        if not out:
            return []
        boxes = [[b for _, _, b in out]]
        inp = self.sp(img, input_boxes=boxes, return_tensors="pt").to(self.dev)
        o = self.sm(**inp, multimask_output=False)
        masks = self.sp.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                                           inp["reshaped_input_sizes"].cpu())[0][:, 0].numpy()
        return [(c, s, b, m) for (c, s, b), m in zip(out, masks)]


def _match(label, cats):
    """Map Grounding DINO's predicted phrase back to one of the queried categories."""
    label = label.strip().lower()
    for c in sorted(cats, key=len, reverse=True):
        if c.lower() == label:
            return c
    for c in sorted(cats, key=len, reverse=True):
        if c.lower() in label or label in c.lower():
            return c
    return None
