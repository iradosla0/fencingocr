"""Process several videos concurrently.

Read this before setting `max_workers` above the default of 2 - the right
number depends on what your hardware actually has, not a guess.

What concurrency here does and does not buy you
-------------------------------------------------
The GPU forward pass does NOT run four videos at once just because four
threads call into it. Kernel launches from multiple Python threads into one
CUDA context queue onto the same default stream and execute in order - there
is no explicit multi-stream setup here, deliberately, because it would add
real complexity for a stage that is already the smallest slice of total time
(see the README's per-stage table: detection is ~1-3 min out of a
12-35 min single-stream run).

What genuinely overlaps is everything that is NOT the GPU:

  * ffmpeg decode runs as a subprocess per video (decode.py). Subprocess I/O
    releases the GIL, so while video A's frames sit in the pipe waiting to be
    read, video B's ffmpeg process is decoding on the CPU (or a separate
    NVDEC hardware context - see below) in parallel.
  * Every Tesseract call is also a subprocess fork. Same story: while video
    A waits on the GPU, video B's Tesseract calls run.

Decode is the one stage where multiple *hardware* decode engines can give you
genuine parallelism, not just GIL-release overlap - but only up to however
many physical NVDEC engines your card has. NVIDIA's own documentation confirms
the driver load-balances multiple concurrent decoding sessions across whatever
NVDEC engines exist on the card, with negligible context-switch cost even when
oversubscribed. A single stream only ever uses one engine (confirmed by
NVIDIA's own forum: a single decode session on an A100, which has 5 NVDEC
engines, showed only one engine active). So the number of engines is a cap on
how many videos can decode at full hardware speed *simultaneously*, not a
speed multiplier for any one of them:

    T4                1 NVDEC  - concurrency helps OCR/decode-CPU overlap only,
                                  little decode parallelism to gain
    RTX 20/30/40-series consumer  usually 1-2 NVDEC active (fewer than the
                                  same-generation datacenter part - GeForce
                                  cards have some engines disabled)
    L4                 4 NVDEC  - four streams can decode at full hardware
                                  speed at once
    A100                5 NVDEC  - five streams, but no NVENC at all: if you
                                  ever use cut_mode='reencode' this falls back
                                  to slow CPU libx264 encoding regardless of
                                  how many videos you run concurrently

Don't guess your card's engine count from memory - it varies by exact SKU and
NVIDIA has changed how many are enabled on consumer cards over time. Use
tools/probe_concurrency.py to measure directly against your actual hardware
before picking max_workers for a real batch run.

What is safe to share across concurrent videos, and what is not
-----------------------------------------------------------------
The Detector (the loaded YOLO model) is stateless during inference - nothing
in Detector.detect() mutates shared state - so one instance is built here and
passed to every video's analyse() call, avoiding N redundant model loads.

Everything else is per-video mutable state and must NOT be shared:
ScoreReader's digit template bank, the bout state machine's prev_scores /
waiting_for_start, NameReader. pipeline.analyse() already constructs all of
these fresh on every call, so passing a shared Detector into it and nothing
else is sufficient - no additional locking is needed here.

CPU budget
----------
Tesseract subprocess calls and ffmpeg's software-side work (colour-space
conversion, muxing) compete for CPU cores across however many videos are
running at once. max_workers set higher than your decode-engine count can
still help by overlapping OCR/CPU work, but pushing it far past your core
count will start hurting instead of helping. As a starting point, keep
max_workers <= min(hardware_decode_engines, cpu_cores // 2) and measure.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from . import decode
from .config import Config
from .cut import cut_all
from .detect import Detector
from .pipeline import analyse, build_bouts, resolve_bout_names, write_report


def process_many(
    videos: List[Path],
    cfg: Config,
    max_workers: int = 2,
    detector: Optional[Detector] = None,
) -> Dict[Path, List[Path]]:
    """Analyse and cut several videos concurrently, sharing one Detector.

    Bouts for each video are written to a subfolder named after that video's
    stem (cfg.outdir / video.stem), not flattened into cfg.outdir directly.
    Two different broadcasts can produce the same filename (the same pair of
    fencers meeting twice, or two unreadable name plates both falling back to
    "bout_003") - namespacing by source video avoids one run's cut silently
    overwriting another's.

    Returns {video_path: [written_bout_paths]} for videos that succeeded. A
    video that raised is omitted from the return value; check the printed
    summary (or wrap this call and inspect exceptions yourself if you need
    programmatic access to failures).
    """
    if not videos:
        return {}

    owns_detector = detector is None
    if owns_detector:
        print(f"[batch] loading shared detector once for {len(videos)} videos")
        detector = Detector(cfg.weights, cfg.device, cfg.conf_thres,
                            cfg.iou_thres, cfg.half, cfg.imgsz)

    results: Dict[Path, List[Path]] = {}
    errors: Dict[Path, BaseException] = {}

    def _one(video: Path) -> List[Path]:
        records, boundaries, stats, overlay_clusterer = analyse(video, cfg, detector=detector)
        bouts = build_bouts(records, boundaries, cfg)
        meta = decode.probe(video)
        bouts = resolve_bout_names(video, bouts, overlay_clusterer, cfg, meta,
                                   detector=detector)
        write_report(video, records, boundaries, bouts, stats, cfg)
        out_subdir = Path(cfg.outdir) / video.stem
        jobs = [(b.start, b.end, b.filename) for b in bouts]
        return cut_all(video, jobs, out_subdir, cfg.cut_mode)

    print(f"[batch] {len(videos)} video(s), max_workers={max_workers}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, v): v for v in videos}
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                written = fut.result()
                results[v] = written
                print(f"[batch] done:   {v.name}  ({len(written)} bouts)")
            except BaseException as e:  # noqa: BLE001 - report, don't hide
                errors[v] = e
                print(f"[batch] FAILED: {v.name}: {e}")

    elapsed = time.time() - t0
    print(f"\n[batch] {len(results)}/{len(videos)} succeeded in "
          f"{elapsed / 60:.1f} min")
    if errors:
        print(f"[batch] {len(errors)} failed:")
        for v, e in errors.items():
            print(f"  {v.name}: {type(e).__name__}: {e}")

    return results
