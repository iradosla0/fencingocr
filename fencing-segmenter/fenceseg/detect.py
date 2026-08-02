"""YOLOv5 detection, batched.

The original ran `model(frame)` one frame at a time and then called
`results.pandas().xyxy[0]`, which builds a DataFrame per frame. Both are
avoidable: AutoShape accepts a list of images and batches them, and the raw
`results.xyxy[i]` tensors carry the same information without the pandas
round-trip.

Class map baked into best.pt (read straight out of the checkpoint):
    0 fencer   1 overlayLeft   2 overlayRight   3 scoreLeft   4 scoreRight
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

FENCER = "fencer"
OVERLAY_LEFT = "overlayLeft"
OVERLAY_RIGHT = "overlayRight"
SCORE_LEFT = "scoreLeft"
SCORE_RIGHT = "scoreRight"


@dataclass
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    conf: float
    name: str

    def as_int(self):
        return int(self.xmin), int(self.ymin), int(self.xmax), int(self.ymax)


class Detector:
    def __init__(
        self,
        weights: Path,
        device: str = "cuda:0",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        half: bool = True,
        imgsz: int = 640,
    ):
        import torch

        self.torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.imgsz = imgsz
        self.half = bool(half) and self.device.type == "cuda"

        model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=str(weights), verbose=False
        )
        model.conf = conf_thres
        model.iou = iou_thres
        model.max_det = 20
        model.agnostic = False
        model.to(self.device)
        if self.half:
            model.half()
        model.eval()

        self.model = model
        self.names: Dict[int, str] = dict(model.names) if isinstance(model.names, dict) \
            else {i: n for i, n in enumerate(model.names)}

    def detect(self, frames: Sequence[np.ndarray]) -> List[List[Box]]:
        """Run one batched forward pass. Returns per-frame lists of Box."""
        torch = self.torch
        # AutoShape wants RGB.
        rgb = [f[:, :, ::-1] for f in frames]
        with torch.inference_mode():
            results = self.model(rgb, size=self.imgsz)

        out: List[List[Box]] = []
        for i in range(len(frames)):
            det = results.xyxy[i]
            boxes: List[Box] = []
            if det is not None and len(det):
                arr = det.detach().float().cpu().numpy()
                for x1, y1, x2, y2, conf, cls in arr:
                    boxes.append(
                        Box(float(x1), float(y1), float(x2), float(y2),
                            float(conf), self.names.get(int(cls), str(int(cls))))
                    )
            out.append(boxes)
        return out


def group(boxes: Sequence[Box]) -> Dict[str, List[Box]]:
    """Bucket a frame's detections by class name."""
    g: Dict[str, List[Box]] = {}
    for b in boxes:
        g.setdefault(b.name, []).append(b)
    return g


def best(boxes: Sequence[Box], name: str) -> Box | None:
    """Highest-confidence box of a given class, or None.

    The original took whichever scoreLeft/scoreRight row pandas happened to
    iterate over last. Taking the most confident one is strictly better and
    identical whenever there is only one, which is the normal case.
    """
    cands = [b for b in boxes if b.name == name]
    if not cands:
        return None
    return max(cands, key=lambda b: b.conf)
