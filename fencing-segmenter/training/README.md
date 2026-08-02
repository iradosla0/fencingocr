# Retraining notes

Everything here comes from reading the training configuration stored inside
`best.pt` itself, so it describes what actually happened, not what was
intended.

## What the checkpoint says

| field | value |
|---|---|
| base weights | `yolov5m.pt` |
| config | `models/yolov5m.yaml` (depth 0.67, width 0.75, ~21M params) |
| classes | 5 — `fencer`, `overlayLeft`, `overlayRight`, `scoreLeft`, `scoreRight` |
| image size | 640 |
| batch size | 16 |
| epochs | 600 |
| precision | fp16 (stripped, optimizer removed) |
| `fliplr` | **0.5** |
| `mosaic` | 1.0 |
| `translate` / `scale` | 0.1 / 0.5 |

## The three things to change, in order of expected payoff

### 1. Turn off horizontal flip

`fliplr: 0.5` with left/right semantic classes is a labelling contradiction
baked into the augmentation. Every flipped image tells the model that the box
it should call `scoreLeft` is on the right of the frame, while the label still
says `scoreLeft`. Roughly half of all training samples carried this
contradiction.

This is very likely a meaningful part of why the detector is inconsistent, and
it costs nothing to fix. Use `training/hyp.fencing.yaml`.

If you want to keep flip augmentation, YOLOv5 supports it correctly only if
you also supply a flip index that swaps paired class IDs when the image is
mirrored. Setting `fliplr: 0.0` is the safe option.

### 2. Add data where the model is actually wrong

600 epochs of a 21M-parameter model on ~100 images is heavily into
memorisation. More epochs will not help; more *informative* images will.

```bash
python tools/harvest_frames.py stream1.mp4 stream2.mp4 \
    --weights best.pt --out dataset/candidates --per-bucket 150
```

This writes frames bucketed by `uncertain`, `miscount`, `novel_layout` and
`no_overlay`, each with a pre-filled label file containing the current
predictions. Correcting predictions is far quicker than drawing boxes, so a
500-image expansion is a few hours rather than a weekend.

Bias the harvest across as many *different broadcasts* as you can find. Ten
frames from twenty events beats two hundred frames from one event — the thing
you are short of is overlay-format diversity, not frame count.

Aim for a genuine validation split held out **by event**, not by frame.
Frames from the same broadcast are near-duplicates; a random split will report
a mAP far above what you actually get on a new stream, which is the most
likely reason the model feels better in validation than in practice.

### 3. Consider a smaller model

`yolov5m` on this dataset is oversized. With the boxes being large,
high-contrast and nearly axis-aligned, `yolov5s` will very likely match it
while running roughly twice as fast, giving you a second speedup on top of the
pipeline changes. Worth an A/B once the dataset is larger.

## Suggested command

```bash
python train.py \
  --img 640 --batch 16 --epochs 150 \
  --data data.yaml \
  --hyp ../training/hyp.fencing.yaml \
  --weights yolov5s.pt \
  --patience 30 \
  --name fencing_v2
```

150 epochs with early stopping rather than a fixed 600. With a larger dataset
the useful stopping point is much earlier, and `--patience` finds it.

## A note on what retraining will *not* fix

The digit reading. That was Tesseract, not the detector, and the notebook's
Tesseract configuration had a structural bug — `--psm 10` means "single
character", which cannot ever return a two-digit score. See the OCR section of
the top-level README. Improving the detector will not touch that; it is fixed
separately in `fenceseg/ocr.py`.
