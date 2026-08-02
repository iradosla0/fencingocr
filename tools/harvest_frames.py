"""Harvest the frames that are worth labelling next.

The model is trained on ~100 images. The cheapest large accuracy win is not a
new architecture, it is another few hundred images chosen so that each one
teaches the model something it currently gets wrong. Labelling 300 random
frames from a stream mostly re-teaches what it already knows; labelling 300
frames selected by disagreement does not.

This script runs the current weights over one or more videos and saves frames
that fall into the categories the model is demonstrably weak on, together with
a pre-filled YOLO label file containing the current predictions. In a labelling
tool you then correct rather than draw from scratch, which is roughly 5x
faster.

Selection buckets:

  uncertain   a detection sits in the ambiguous confidence band. These are the
              frames sitting on the decision boundary.
  miscount    fencer count is not 2 during what looks like active play, or
              exactly one of scoreLeft/scoreRight fired. A single-sided score
              detection is almost always a genuine miss, and it is exactly the
              failure that makes the guard drop a frame.
  novel_layout the score box appeared in a grid cell this run has not seen
              before. This is the direct answer to "different overlay formats
              where the score is in different positions": it finds the formats
              your training set does not cover, instead of you hunting for them.
  no_overlay  scoreLeft/scoreRight missing while two fencers are present.

Usage:
    python tools/harvest_frames.py stream.mp4 --weights best.pt --out dataset/candidates
    python tools/harvest_frames.py a.mp4 b.mp4 --weights best.pt --per-bucket 150

Output layout is ready for a YOLOv5 dataset:
    candidates/images/<video>_<idx>_<bucket>.jpg
    candidates/labels/<video>_<idx>_<bucket>.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenceseg import decode  # noqa: E402
from fenceseg.detect import (FENCER, OVERLAY_LEFT, OVERLAY_RIGHT, SCORE_LEFT,
                             SCORE_RIGHT, Detector, best)  # noqa: E402

CLASS_INDEX = {
    FENCER: 0,
    OVERLAY_LEFT: 1,
    OVERLAY_RIGHT: 2,
    SCORE_LEFT: 3,
    SCORE_RIGHT: 4,
}


def to_yolo(box, w, h):
    cx = (box.xmin + box.xmax) / 2.0 / w
    cy = (box.ymin + box.ymax) / 2.0 / h
    bw = (box.xmax - box.xmin) / w
    bh = (box.ymax - box.ymin) / h
    return CLASS_INDEX[box.name], cx, cy, bw, bh


def layout_cell(box, w, h, grid=10):
    return (int((box.xmin + box.xmax) / 2 / w * grid),
            int((box.ymin + box.ymax) / 2 / h * grid))


def classify(boxes, w, h, seen_layouts, lo, hi):
    fencers = [b for b in boxes if b.name == FENCER]
    sl = best(boxes, SCORE_LEFT)
    sr = best(boxes, SCORE_RIGHT)

    if (sl is None) != (sr is None):
        return "miscount"

    if sl is not None and sr is not None:
        for b in (sl, sr):
            cell = (b.name, *layout_cell(b, w, h))
            if cell not in seen_layouts:
                seen_layouts.add(cell)
                return "novel_layout"

    if any(lo <= b.conf <= hi for b in boxes):
        return "uncertain"

    if len(fencers) == 2 and sl is None and sr is None:
        return "no_overlay"

    if len(fencers) not in (0, 2):
        return "miscount"

    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("dataset/candidates"))
    ap.add_argument("--sample-fps", type=float, default=0.2,
                    help="harvest rate; low is fine, adjacent frames are near-duplicates")
    ap.add_argument("--per-bucket", type=int, default=120)
    ap.add_argument("--conf-lo", type=float, default=0.25)
    ap.add_argument("--conf-hi", type=float, default=0.60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    img_dir = args.out / "images"
    lbl_dir = args.out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    det = Detector(args.weights, args.device, conf_thres=0.15)
    counts = Counter()
    seen_layouts = set()

    for video in args.videos:
        meta = decode.probe(video)
        w, h = meta["width"], meta["height"]
        print(f"[harvest] {video.name}")
        for batch in decode.iter_batches(video, args.sample_fps,
                                         args.batch_size, None, meta):
            frames = [f for _, _, f in batch]
            for (idx, ts, frame), boxes in zip(batch, det.detect(frames)):
                bucket = classify(boxes, w, h, seen_layouts,
                                  args.conf_lo, args.conf_hi)
                if bucket is None or counts[bucket] >= args.per_bucket:
                    continue
                counts[bucket] += 1
                stem = f"{video.stem}_{idx:06d}_{bucket}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                with (lbl_dir / f"{stem}.txt").open("w") as f:
                    for b in boxes:
                        if b.name not in CLASS_INDEX:
                            continue
                        c, cx, cy, bw, bh = to_yolo(b, w, h)
                        f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print("\nharvested:")
    for k, v in counts.most_common():
        print(f"  {k:14s} {v}")
    print(f"\n-> {args.out}")
    print("Labels are PREDICTIONS, not ground truth. Correct them before training.")


if __name__ == "__main__":
    raise SystemExit(main())
