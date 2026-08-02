# Fencing bout segmenter (rework of `streamlined-version1.ipynb`)

Segments a fencing livestream into per-bout video files, named from the
broadcast's own name plates.

```
Alexander Massialas USA vs. Race Imboden USA.mp4
```

---

## The constraint that governed everything

> *the conditions for the start and finish remain unchanged unless approved*

They are unchanged. `fenceseg/segment.py` contains the same guard, the same
`start` condition, the same `end` condition, the same `prev_scores` update
order, and the same truth test on `prev_scores`.

This is not asserted, it is tested. `tools/verify_state_machine.py` holds a
verbatim transcription of the original `process_frame` and fuzzes both
implementations against each other:

```
$ python tools/verify_state_machine.py
sequences checked: 26096
mismatches:        0
PASS - refactored state machine is identical to the original.
```

Coverage is exhaustive over short sequences from a reduced alphabet, plus
20,000 uniform-random sequences, plus 2,000 sequences shaped like real bouts
with dropouts and occlusions injected. Run it after any edit to
`segment.py`.

**What did change is the quality of the values fed into those conditions.**
The conditions consume `fencer_count`, `score_left_int` and `score_right_int`.
Making the OCR read those more correctly is the accuracy improvement you
asked for; the logic acting on them is untouched. Where a read is not
trustworthy the OCR now returns `None`, which is precisely what the original
produced on OCR failure, and the state machine handles it identically.

---

## Quick start

```bash
pip install -r requirements.txt
apt-get install -y ffmpeg tesseract-ocr

# local file
python -m fenceseg.cli run stream.mp4 --weights best.pt

# FencingTV (URL = the m3u8 Request URL from DevTools, see below)
python -m fenceseg.cli run "https://.../index.m3u8" --weights best.pt --hwaccel cuda

# boundaries only, no cutting - useful for checking names first
python -m fenceseg.cli run stream.mp4 --weights best.pt --analyse-only
```

`run_pipeline.ipynb` is a thin notebook driver for Kaggle/Colab.

---

## FencingTV

FencingTV serves HLS, and the playlist URL is not discoverable from the page,
so it still comes from you, exactly as in `andrefisch/VideoDownloadScript`:

1. Open the video page, press F12
2. Network tab, filter for `.m3u8`
3. Click the entry with the random-looking name — **not** the ones containing
   `rendition`
4. Headers tab → copy the **Request URL**

Two consequences of it being an m3u8 rather than a YouTube page:

- **No title metadata.** `%(title)s` gives you nothing usable, so an explicit
  output name is required. This is irrelevant to the final bout filenames,
  which come from the overlay.
- **Thousands of small fragments.** yt-dlp fetches them one at a time by
  default. `concurrent_fragment_downloads=8` (the `--fragments` flag) is the
  single biggest download speedup available, often an order of magnitude on a
  multi-hour stream.

---

## Why it is faster

Measured against the structure of the original, largest first.

### 1. Stop decoding frames you throw away

The original:

```python
while cap.isOpened():
    ret, frame = cap.read()           # full decode + colour convert, every frame
    if frame_count % int(fps) == 0:   # ...then discard ~29 of every 30
```

For a 3-hour 30fps stream that is ~324,000 full decodes to obtain ~10,800
usable frames. **Decoding, not inference, was the dominant cost of the entire
pipeline.**

Now ffmpeg applies the `fps` filter inside its own graph, so frames are
dropped before colour conversion and only sampled frames cross into Python.
`--hwaccel cuda` moves the decode to the GPU on top of that. Same frames
analysed, roughly 20–30x less decode work.

### 2. Delete the pre-processing that existed for no reason

The original split every stream into 3-hour chunks with a full remux, then
re-encoded every chunk through `h264_nvenc` if the video was ≤1280 wide or
AV1. Analysis reads frames through ffmpeg either way, and ffmpeg decodes AV1
and any resolution natively. That re-encode pass — frequently hours of GPU
time per stream — bought nothing and is gone. So is the chunking, along with
the offset arithmetic it required.

### 3. Stop downloading the video twice

Cell 5 re-downloaded the entire livestream in order to cut it, having already
downloaded it in cell 3. It cuts from the file on disk now.

### 4. Batch the detector, drop the pandas round-trip

`model(frame)` one frame at a time, then `results.pandas().xyxy[0]` building a
DataFrame per frame. Now frames go through in batches of 32 (`--batch-size`,
throughput only — it cannot affect which frames are evaluated or what is
decided) and the raw result tensors are read directly.

### 5. Stop forking Tesseract twice per frame

Every `pytesseract` call spawns a subprocess: roughly 40–80ms of pure
overhead, twice per sampled frame, ~10,800 frames per stream. That is on the
order of 20 minutes per stream spent on process creation.

`DigitTemplateBank` learns the broadcast's own digit glyphs from
high-confidence Tesseract reads, then matches by normalised cross-correlation
— microseconds, in-process. After a few hundred frames Tesseract is barely
called. The pipeline reports the split at the end:

```
[ocr] template hits: 18422  tesseract calls: 611  low-confidence reads dropped: 97
```

### 6. Delete the non-local-means denoise

`cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)` is one of the most expensive
filters in OpenCV. It is designed to suppress *sensor noise in photographs*. A
broadcast score overlay is synthetic graphics composited digitally — there is
no sensor noise to remove. It was costing tens of milliseconds per call, twice
per frame, to blur away the thin strokes it was meant to preserve.

### 7. Only run OCR on frames that can matter

Frames failing the guard (`fencer_count != 2`, or a missing score box)
contribute nothing in the original — they do not even update `prev_scores`.
So OCR is not run on them at all now. On a typical stream, with breaks,
crowd shots and piste changes, this skips a large fraction of frames outright.

---

## Why the OCR is more accurate

The original score reader:

```python
gray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
denoised  = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
_, binar  = cv2.threshold(denoised, 0, 255, THRESH_BINARY + THRESH_OTSU)
sharpened = cv2.filter2D(binar, -1, [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
text      = pytesseract.image_to_string(sharpened,
                config=r'--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789')
```

**`--psm 10` means "treat the image as a single character".** Fencing scores
run to 15. Every two-digit score was being handed to a mode that structurally
cannot return two digits. This is the largest accuracy bug in the notebook and
it bites precisely at the score that triggers your end condition. Now `--psm 7`
(single text line).

**Otsu output polarity was never normalised.** Tesseract expects dark text on
a light background. Depending on the overlay skin, Otsu produces either
polarity — so an entire class of your overlay formats was being fed inverted
images. This is very likely a chunk of what you were reading as "the model is
undertrained on different overlay formats": for some of those formats the
detector was fine and the OCR was inverted. Now polarity is measured (glyph
strokes are the minority of pixels, so the background must dominate) and
normalised.

**No upscaling.** Score digits are ~15–25px tall in a 1080p broadcast;
Tesseract's LSTM engine wants roughly 30–40px of x-height. Crops are now
Lanczos-upscaled to a fixed target height before thresholding.

**The sharpening kernel ran after binarisation**, on an image that was already
pure black and white. It could only add speckle. Removed.

**No confidence gate.** A garbage read was returned as a confident integer, and
a misread of 13 as 15 ends a bout early. Reads now come back with a
confidence, and sub-threshold reads return `None` — the skip-this-frame path
the state machine already had. At 1 Hz you get another attempt one second
later, so abstaining costs almost nothing while a wrong answer costs a bad cut.

**No range check.** Any integer Tesseract emitted was accepted. A fencing score
cannot exceed 15; out-of-range reads are now rejected rather than propagated.

The learned template bank raises accuracy as well as speed, because the
templates come from the exact font and rendering of the stream being
processed rather than from Tesseract's general-purpose model. In degradation
testing it never returned a *wrong* digit — under heavy noise and blur it
abstains, which is the failure mode you want.

### Temporal voting (off by default)

`--temporal-vote` enables single-frame outlier rejection: at 1 Hz a real score
moves one touch at a time, so a jump from 2 to 13 in one second is a misread.
It is deliberately narrow — a reset to 0-0 and a score of 15 are whitelisted
unconditionally, because your end condition depends on both, and the allowed
jump widens with time since the last accepted read so a long occlusion does
not cause it to reject the correct higher score afterwards.

**It is off by default** because it changes which frames yield a value and can
therefore shift a boundary by one sampling interval. That edges toward your
"conditions unchanged" constraint, so it is your call, not mine. A/B it on a
stream you know before enabling.

---

## Filenames

Your model already has `overlayLeft` and `overlayRight` classes trained. The
notebook never used them. That is where the names come from — no extra
training needed.

Name plates are static for the length of a bout, so the pipeline reads them on
a subsample of frames and takes the **modal reading across the whole bout**. A
one-frame misread is outvoted by the ~200 frames of a typical bout, which
makes the names far more reliable than any single-frame read.

Parsing handles the conventions that actually appear:

| plate text | result |
|---|---|
| `MASSIALAS Alexander USA` | `Alexander Massialas USA` |
| `Andreas GEORGIADIS GRE` | `Andreas Georgiadis GRE` |
| `SZABO-LAZAR Andrei ROU` | `Andrei Szabo-Lazar ROU` |
| `O'CONNOR Sean IRL` | `Sean O'Connor IRL` |
| `A. GEORGIADIS GRE` | `A. Georgiadis GRE` |
| `GEORGIADIS GRE` | `Georgiadis GRE` |

Per your instruction it never blocks: whatever is legible goes in the
filename, an unreadable side becomes `Unknown`, and a bout with neither plate
readable falls back to `bout_007`. Collisions get ` (2)`, ` (3)`. Use
`--underscores` if you want `Alexander_Massialas_USA` instead.

Run with `--analyse-only` first to review names before committing to cuts —
overriding a filename is much cheaper than re-cutting.

---

## The undertrained model

I read the training configuration out of `best.pt`. `yolov5m`, 640px, batch
16, **600 epochs**, 5 classes, and:

```
fliplr: 0.5
```

**Your classes are left/right semantic** — `scoreLeft`, `scoreRight`,
`overlayLeft`, `overlayRight` — and horizontal flip mirrors the image while
keeping the label. Roughly half your training samples were telling the model
that a box labelled `scoreLeft` appears on the right-hand side of the frame.
On a ~100-image dataset that is a large fraction of your signal working
directly against the thing you need the model to learn.

This is very likely a real contributor to the inconsistency you are seeing,
and it costs nothing to fix. `training/hyp.fencing.yaml` sets `fliplr: 0.0`,
raises `translate` (the augmentation that *does* target "score in different
positions"), and pulls `mosaic` back from 1.0 so the model sees some intact
broadcast frames. `training/README.md` covers the rest — including that 600
epochs of a 21M-parameter model on 100 images is memorisation, and that your
validation split should be held out **by event**, not by frame, or the
reported mAP will be far above what you get on a new stream.

For expanding the dataset:

```bash
python tools/harvest_frames.py stream1.mp4 stream2.mp4 \
    --weights best.pt --out dataset/candidates --per-bucket 150
```

This saves the frames the model is *least sure about* — ambiguous confidence,
one-sided score detections, and score boxes appearing in grid positions not
seen before — each with a pre-filled label file containing the current
predictions. Correcting predictions is roughly 5x faster than drawing boxes,
and the `novel_layout` bucket finds the overlay formats your training set does
not cover instead of you hunting for them by hand.

Ten frames from twenty different events beats two hundred from one. What you
are short of is format diversity, not frame count.

---

## Layout

```
fenceseg/
  config.py     all tunables in one dataclass
  download.py   yt-dlp / FencingTV HLS, concurrent fragments
  decode.py     ffmpeg sampling pipe (the main speedup)
  detect.py     batched YOLOv5, fp16
  ocr.py        score + name reading, template bank, temporal filter
  segment.py    *** the state machine - conditions unchanged ***
  naming.py     plate parsing, voting, filename construction
  cut.py        ffmpeg extraction, copy or frame-accurate
  pipeline.py   single-pass orchestration
  cli.py        command line
tools/
  verify_state_machine.py   parity fuzz test vs the original
  harvest_frames.py         active-learning frame selection
training/
  hyp.fencing.yaml          corrected augmentation
  README.md                 retraining guidance
run_pipeline.ipynb          notebook driver
```

---

## Two things I could not test, and one small behaviour change

I have no GPU, no Torch and no fencing footage in this environment. What is
verified: the state machine parity fuzz (26,096 sequences, exact match), the
OCR preprocessing chain against synthetic score crops in both polarities
including two-digit scores, the template bank round-trip and its behaviour
under noise and blur, the name parser across six plate formats, and that
everything imports and compiles. What is **not** verified end-to-end: the
Torch/YOLOv5 loading path, the ffmpeg decode pipe against a real file, and
Tesseract's actual confidence values on your overlays — `--score-min-conf 55`
is a starting point and you should tune it on one stream with `--analyse-only`
before running a batch.

**The one behaviour change worth flagging.** The original sampled every
`int(fps)`-th frame. On 29.97fps footage `int(fps)` is 29, so it sampled every
0.9676s and drifted about 3.5 seconds per hour relative to wall-clock
timestamps. The `fps=1` filter samples at exactly 1.000s. The conditions are
unchanged and the decision rate is the same 1 Hz, but the specific frames
examined differ slightly and the timestamps are now correct rather than
drifting. I treated this as a bug fix; `--sample-fps` is exposed if you want
to experiment, and if you would rather I reproduced the original drift
exactly, say so and I will.

There was also a genuine bug in the original cell 4: `process_video` returns
`(waiting_for_start, segment_duration)`, but the caller assigned that tuple
straight back into `waiting_for_start`. From the second segment onward
`waiting_for_start` was a non-empty tuple, which is always truthy, so the
start condition could fire when the machine was not actually waiting for a
start. Removing the chunking removed the bug with it.

---

## Batch mode: processing several streams concurrently

For a whole season's worth of archived streams rather than one at a time.

```bash
# a .txt file of local paths and/or m3u8 URLs, one per line
python -m fenceseg.cli run streams.txt --weights best.pt --workers 4
```

Read `fenceseg/batch.py`'s module docstring before setting `--workers` above
1 — the short version:

- **One video's decode can't be split across GPUs or engines** — H.264 frames
  reference earlier frames, so decode is inherently sequential per file.
  Concurrency here means *multiple videos* running at once, not one video
  going faster.
- **The GPU forward pass doesn't truly run in parallel either** — kernel
  launches from multiple threads into one CUDA context queue onto the same
  stream. The real overlap comes from ffmpeg decode and Tesseract OCR, both
  subprocesses that release Python's GIL, so while one video's frames sit on
  the GPU, other videos' decode/OCR keep the CPU busy.
- **Hardware decode parallelism is capped by your GPU's NVDEC engine count**
  — 1 on a T4, usually 1-2 on consumer GeForce cards (fewer engines enabled
  than the same-gen datacenter part), 4 on an L4, 5 on an A100 (which has no
  NVENC at all — irrelevant for the default `copy` cut mode, but relevant if
  you ever switch to `reencode`).
- **Don't guess your card's number** — measure it:
  ```bash
  python tools/probe_concurrency.py your_video.mp4 --max-n 8
  ```
  This launches N concurrent `-hwaccel cuda` decodes for increasing N and
  reports where aggregate throughput stops climbing. That row is roughly
  your hardware ceiling; also budget CPU cores on top of it, since OCR and
  ffmpeg's software-side work compete for cores across concurrent videos too.

**What's safe to share, what isn't.** The loaded YOLO model is shared across
all concurrent videos (stateless during inference, avoids N redundant loads).
Everything per-broadcast and mutable — digit template banks, `prev_scores`,
`waiting_for_start` — is built fresh per video, same as single-video mode
already did, so no cross-stream contamination is possible.

**Output layout differs from single-video mode.** Bouts land in
`cfg.outdir/<video-stem>/`, not flattened into `cfg.outdir` directly — two
different broadcasts producing the same filename (same fencers meeting
twice, or two unreadable plates both falling back to `bout_003`) would
otherwise silently overwrite each other.

`tools/verify_batch.py` checks the concurrency contract itself — shared
detector built exactly once, one video's failure doesn't sink the others,
outputs namespaced correctly, and that videos genuinely overlap rather than
serialising behind an accidental lock — all without needing a GPU to run:

```
$ python tools/verify_batch.py
ALL 5 PASSED
```

---

## Name reading: targeted seeks instead of continuous sampling

Rebuilt around two observations: bouts of very different lengths mean a
fixed time offset (e.g. "5 minutes in") frequently lands after a short bout
has already ended, and the name-plate box position is a static graphic for
the whole broadcast, so it doesn't need re-detecting every time it's read.

**Phase 1 (`analyse`)** is otherwise unchanged - full detection still runs on
every sampled frame, because `fencer_count` drives the state machine guard
and that condition is never touched. What's new: every frame's
`overlayLeft`/`overlayRight` box coordinates - already computed as part of
that same forward pass, previously discarded - now feed a
`PositionClusterer` that locks onto the stable position once it's seen
enough times.

**Phase 2 (`resolve_bout_names`)** runs after boundaries are known. For each
bout it seeks to a handful of instants - near the start, the middle, near
the end, as *fractions of that bout's own duration* with a small buffer kept
away from the exact boundary (broadcasts often show a transition graphic
right at the instant a bout starts or ends) - and reads the name plate only
there. If Phase 1 locked the overlay position, no detector call happens at
all: crop directly from known coordinates. A small burst of frames around
each anchor guards against any single frame being blurry or mid-transition,
and `naming.vote()`'s existing majority-agreement logic resolves the
readings, generalizing "2 out of 3 agree" to however many were actually
collected.

Verified against real encoded video (not just synthetic arrays): a burned-in
timestamp round-tripped exactly through `decode.extract_frame`'s seek, and a
burned-in name plate was read correctly at every one of 9 targeted seeks
across a simulated bout, unanimous vote, correct filename. Anchor placement
was checked across bout durations from 0.5 seconds to 8 minutes - it never
lands outside the bout, which a fixed offset can't guarantee.

---

## Optional: adaptive start-marker detection

Off by default (`start_mode="legacy"`, i.e. the protected condition above is
exactly what runs). Turn on with `--start-mode clock_or_score` if you want
the start marker placed earlier than the original 0-0 detection, based on
either signal:

- A **confirmed** clock-decrease -> marker = that instant minus 30s
- A **confirmed** score-change, if it happens first (or the clock is
  unreadable this stream) -> marker = that instant minus 60s, and the clock
  signal is disregarded for that bout even if it fires afterward

"Confirmed" means the same new value was read on `--start-score-confirm-frames`
(default 2) or `--start-clock-confirm-frames` (default 2) consecutive frames
- applied to BOTH signals, not just the score one, since a single misread
  digit can produce a false reading either way. Verified directly:
  a synthetic one-frame score blip and a synthetic one-frame clock blip both
  correctly fail to trigger a start in `tools/verify_adaptive_start.py`.

**The end condition is completely untouched** - `fenceseg/segment.py`'s
protected `BoutStateMachine` is not modified at all by this feature. The new
logic lives entirely in `fenceseg/adaptive_start.py`, a separate file, and
reproduces the same end condition independently rather than importing it, so
there's no runtime coupling to the protected class. What guarantees the two
don't drift apart is a test, not the architecture: `verify_adaptive_start.py`
runs 3,000 randomized sequences through both classes and requires identical
end-firing behaviour on every one - currently 0 mismatches.

**There is no trained detector class for the clock/timer.** The model has
five classes - fencer, overlayLeft, overlayRight, scoreLeft, scoreRight -
and none of them is the timer. `ClockReader` locates it heuristically
instead: it searches a few candidate regions relative to the score boxes'
own position (inline with them, above, below) during the first several
frames, locks onto whichever one's OCR output repeatedly parses as a
plausible clock value, and gives up if none do - the exact same
search-then-lock pattern already used for the name-plate corner badge.
Verified against real encoded video: a synthetic countdown was located and
locked without ever being told where it was, from a starting position that
was never hardcoded anywhere in the code.

This is inherently less certain than everything else in this pipeline, which
all locate their target via the trained model first. If it proves unreliable
on real footage, the robust fix is adding a `timer` class to the detector and
retraining (see `training/README.md`) - `ClockReader` would then become
unnecessary. Until then, the score-change signal works independently of the
clock entirely, so a stream where the clock never locks still gets usable
(if less precisely timed) start markers.

---

## Waiting N seconds after the end condition before placing the end marker

Already covered by the existing `pad_end` config field / `--pad-end` flag -
`build_bouts()` applies `end = detected_end_timestamp + pad_end` regardless
of which start mode is in use:

```bash
python -m fenceseg.cli run stream.mp4 --weights best.pt --pad-end 10
```

**One real interaction this surfaced and fixes:** if you combine `--pad-end`
with `--start-mode clock_or_score`, the adaptive start marker's
overlap-prevention clamp needs to know about the padding too - otherwise a
fast-confirming next-bout start could land earlier than the previous bout's
actual (padded) exported end, producing two overlapping cut files.
`pipeline.py` now folds `pad_end` into the clamp floor it passes to
`AdaptiveBoutStateMachine`, and `tools/verify_adaptive_start.py` reproduces
the exact overlap scenario as a permanent regression test - the clock
confirms fast enough that the naive marker would land at t=73, but the
padded previous end is t=110, and the clamped result correctly comes out at
110, not 73.

`pad_start` is the same idea for the legacy start condition, applied
symmetrically (`start = detected_start_timestamp - pad_start`).
