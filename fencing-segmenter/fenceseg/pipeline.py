"""Single-pass analysis and cutting.

Structure of the rework: the notebook made three passes over the media
(download -> split+reencode -> analyse -> re-download -> cut). This makes one
analysis pass that collects everything, then cuts directly from the file that
is already on disk.

Per sampled frame the pass collects:
  * fencer count, presence of both score boxes  -> state machine guard
  * left/right score integers                   -> state machine conditions
  * left/right name plate text                  -> filenames

Name plates are OCR'd on a subsample (they do not change within a bout) and
resolved by majority vote once the bout's boundaries are known.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import decode, detect, naming
from .adaptive_start import AdaptiveBoutStateMachine
from .config import Config
from .detect import (FENCER, OVERLAY_LEFT, OVERLAY_RIGHT, SCORE_LEFT,
                     SCORE_RIGHT, Box, Detector, best)
from .ocr import ClockReader, NameReader, PositionClusterer, ScoreReader, TemporalScoreFilter
from .segment import BoutStateMachine, FrameObs, pair_boundaries, unterminated


@dataclass
class Record:
    idx: int
    ts: float
    fencers: int
    sl: Optional[int]
    sr: Optional[int]


@dataclass
class Bout:
    index: int
    start: float
    end: float
    filename: str
    left_plate: Optional[str]
    right_plate: Optional[str]


def analyse(video: Path, cfg: Config,
           detector: Optional[Detector] = None
           ) -> Tuple[List[Record], List, dict, PositionClusterer]:
    """Run one video through decode -> detect -> score OCR -> state machine.

    Name reading does NOT happen here any more. It used to run inline every
    NAME_EVERY sampled frames throughout each bout - continuous re-detection
    and re-OCR of a name plate that (a) never changes within a bout and (b)
    sits in a screen position fixed for the whole broadcast. Two things
    follow from that: name reads don't need to be frequent, they need to be
    well-chosen, and the overlay box's POSITION doesn't need re-detecting on
    every frame once it's been seen enough times to trust.

    Both of those get handled by moving name reading to a second pass
    (resolve_bout_names, below) that runs AFTER bout boundaries are known:
    it seeks directly to a handful of chosen instants per bout - near the
    start, the middle, near the end, as fractions of THAT bout's own
    duration - rather than sampling continuously. Fixed offsets (e.g. "5
    minutes in") don't work here: plenty of real bouts finish well under 5
    minutes, so a fixed offset would frequently land after the bout has
    already ended, reading the wrong bout's names entirely.

    What THIS function contributes to that second pass, at effectively zero
    extra cost: the detector already runs on every sampled frame here to get
    fencer_count and score boxes (fencer_count is load-bearing for the state
    machine guard and must keep running every frame - that condition is
    unchanged, see segment.py). Every one of those forward passes also
    produces overlayLeft/overlayRight box coordinates, which used to be
    discarded. They're fed into a PositionClusterer here instead, for free -
    no extra detector calls - so that once it locks onto a stable position,
    the second pass can crop directly from known coordinates without needing
    to run the detector again at all on its targeted seek frames.

    `detector` lets a caller supply an already-loaded model so several videos
    can share one GPU-resident YOLO instance instead of each loading its own
    (see batch.py). Inference on a shared Detector is safe to call from
    multiple threads concurrently: the model is in eval mode and nothing in
    Detector.detect() mutates shared state. Everything else constructed
    below - ScoreReader, the state machine, the overlay clusterer - is
    per-video mutable state and must NOT be shared across concurrent videos,
    so a fresh instance is built here every call regardless of whether the
    detector was shared.
    """
    meta = decode.probe(video)
    print(f"[analyse] {video.name}  {meta['width']}x{meta['height']} "
          f"{meta['fps']:.2f}fps  {meta['duration'] / 60:.1f}min  "
          f"codec={meta['codec']}")

    if detector is None:
        detector = Detector(cfg.weights, cfg.device, cfg.conf_thres,
                            cfg.iou_thres, cfg.half, cfg.imgsz)
    score_reader = ScoreReader(cfg.score_min_conf, cfg.score_target_h,
                               cfg.use_templates, cfg.template_match_thres)
    overlay_clusterer = PositionClusterer()
    tfilter = TemporalScoreFilter() if cfg.temporal_vote else None

    use_adaptive_start = cfg.start_mode == "clock_or_score"
    if use_adaptive_start:
        sm = AdaptiveBoutStateMachine(
            waiting_for_start=True,
            score_confirm_frames=cfg.start_score_confirm_frames,
            clock_confirm_frames=cfg.start_clock_confirm_frames,
            score_lookback_s=cfg.start_score_lookback_s,
            clock_lookback_s=cfg.start_clock_lookback_s,
            # The clamp floor must be measured from the ACTUAL exported end
            # (after pad_end), not the raw detected end timestamp - otherwise
            # a padded previous-bout end plus an early adaptive start marker
            # for the next bout can produce overlapping cut files. pad_end is
            # applied downstream in build_bouts(), so it's folded in here too
            # rather than left for that later, separate step to (not) catch.
            min_gap_after_previous_end=cfg.pad_end + cfg.start_min_gap_after_previous_end_s,
        )
        clock_reader = ClockReader(cfg.clock_min_conf, cfg.clock_search_attempts,
                                   cfg.clock_lock_threshold)
    else:
        sm = BoutStateMachine(waiting_for_start=True)
        clock_reader = None
    records: List[Record] = []
    t0 = time.time()
    n = 0

    for batch in decode.iter_batches(video, cfg.sample_fps, cfg.batch_size,
                                     cfg.hwaccel, meta):
        frames = [f for _, _, f in batch]
        dets = detector.detect(frames)

        for (idx, ts, frame), boxes in zip(batch, dets):
            fencers = sum(1 for b in boxes if b.name == FENCER)
            bl = best(boxes, SCORE_LEFT)
            br = best(boxes, SCORE_RIGHT)

            # Free bookkeeping: whenever an overlay box appears in this
            # frame's detections - regardless of the fencer/score guard
            # below - feed it into the position clusterer. No extra detector
            # call, just arithmetic on output already computed.
            ol = best(boxes, OVERLAY_LEFT)
            orr = best(boxes, OVERLAY_RIGHT)
            if ol is not None:
                overlay_clusterer.key_for(ol, frame.shape)
            if orr is not None:
                overlay_clusterer.key_for(orr, frame.shape)

            sl = sr = None

            # Only pay for OCR on frames the guard would let through. Frames
            # failing the guard contribute nothing in the original either.
            if fencers == 2 and bl is not None and br is not None:
                sl = score_reader.read(frame, bl)
                sr = score_reader.read(frame, br)
                if tfilter is not None:
                    sl = tfilter.filter("L", sl, ts)
                    sr = tfilter.filter("R", sr, ts)

            records.append(Record(idx, ts, fencers, sl, sr))

            # Clock reading is only ever consulted while waiting for a bout
            # to start (AdaptiveBoutStateMachine ignores clock_seconds once
            # a bout is underway). Gating on sm.waiting_for_start here, not
            # just on clock_reader being present, skips the OCR cost for the
            # much larger fraction of the video spent mid-bout, where the
            # value would be computed and then immediately discarded.
            clock_val = (clock_reader.read(frame, bl, br)
                        if clock_reader is not None and sm.waiting_for_start
                        else None)

            if use_adaptive_start:
                sm.step(FrameObs(
                    sample_index=idx, timestamp=ts,
                    fencer_count=fencers,
                    has_score_left=bl is not None,
                    has_score_right=br is not None,
                    score_left=sl, score_right=sr,
                ), clock_seconds=clock_val)
            else:
                sm.step(FrameObs(
                    sample_index=idx, timestamp=ts,
                    fencer_count=fencers,
                    has_score_left=bl is not None,
                    has_score_right=br is not None,
                    score_left=sl, score_right=sr,
                ))
            n += 1

        if n % (cfg.batch_size * 20) < cfg.batch_size:
            el = time.time() - t0
            rate = n / el if el else 0
            done = n * (1.0 / cfg.sample_fps)
            pct = 100.0 * done / meta["duration"] if meta["duration"] else 0
            print(f"[analyse] {pct:5.1f}%  {n} frames  {rate:.1f} f/s  "
                  f"{len(sm.boundaries)} boundaries")

    stats = dict(score_reader.stats)
    stats["temporal_rejected"] = tfilter.rejected if tfilter else 0
    stats["frames"] = n
    stats["seconds"] = round(time.time() - t0, 1)
    stats["duration"] = meta["duration"]
    stats["overlay_left_locked"] = overlay_clusterer.dominant_key(OVERLAY_LEFT) is not None
    stats["clock_mode"] = clock_reader.stats["mode"] if clock_reader is not None else "disabled"
    stats["overlay_right_locked"] = overlay_clusterer.dominant_key(OVERLAY_RIGHT) is not None
    return records, sm.boundaries, stats, overlay_clusterer


def build_bouts(records: List[Record], boundaries: List, cfg: Config) -> List[Bout]:
    """Construct Bout objects with timing only. Names come from
    resolve_bout_names, which must run after this and before write_report/cut
    - filenames here are placeholders.
    """
    pairs = pair_boundaries(boundaries)

    dangling = unterminated(boundaries)
    if dangling is not None:
        print(f"[bouts] a start at {dangling.timestamp:.0f}s was never closed "
              f"by an end condition; not exported")

    bouts: List[Bout] = []
    for i, (s, e) in enumerate(pairs, start=1):
        bouts.append(Bout(
            index=i,
            start=max(0.0, s.timestamp - cfg.pad_start),
            end=e.timestamp + cfg.pad_end,
            filename=f"bout_{i:03d}",
            left_plate=None,
            right_plate=None,
        ))
    return bouts


# ---------------------------------------------------------------------------
# Phase 2: name resolution via targeted seeks, not continuous sampling
# ---------------------------------------------------------------------------

# Anchor points as fractions through each bout's own duration, plus a buffer
# in seconds kept away from the exact start/end boundary. Buffering matters:
# broadcasts very often show a transition graphic (a "FENCE" title card, a
# victory animation) at the EXACT instant a bout starts or ends - which is
# precisely the state-machine boundary timestamp. Reading right at that
# instant risks hitting the one moment the name plate is least readable.
_ANCHOR_BUFFER_S = 2.0
_ANCHOR_FRACTIONS = (0.0, 0.5, 1.0)  # start, middle, end - offset by the buffer below
# Small burst around each anchor (seconds), so one blurry/transitional frame
# doesn't sink that anchor's contribution to the vote.
_BURST_OFFSETS_S = (0.0, 0.6, -0.6)


def _anchor_timestamps(start: float, end: float) -> List[float]:
    duration = end - start
    buf = min(_ANCHOR_BUFFER_S, duration * 0.25)
    raw = [start + buf, start + duration * 0.5, end - buf]
    # Very short bouts can push these past each other or out of [start, end];
    # clamp and dedupe rather than read the same instant multiple times.
    clamped = sorted({round(max(start, min(end, t)), 2) for t in raw})
    return clamped


def resolve_bout_names(video: Path, bouts: List[Bout], overlay_clusterer: PositionClusterer,
                       cfg: Config, meta: dict, detector: Optional[Detector] = None
                       ) -> List[Bout]:
    """Fill in left_plate/right_plate/filename for each bout via targeted seeks.

    No continuous per-frame sampling. For each bout: a handful of instants
    near its start, middle, and end (as fractions of THAT bout's duration,
    not a fixed offset - a fixed "5 minutes in" would miss plenty of real
    bouts that finish well under 5 minutes entirely). Each instant is a
    direct ffmpeg seek-and-decode-one-frame, not a re-run of the full decode
    pipeline.

    If overlay_clusterer locked a stable box position during analyse() (the
    normal case - it's built from every sampled frame's detections, not just
    a few), no detector call is needed at all here: crop directly from the
    known coordinates. Only if a side never locked - a genuinely degenerate
    case - does this fall back to running the detector on that one seeked
    frame, still far cheaper than the old continuous-sampling approach.

    One NameReader instance is shared across every bout in this video, so
    its adaptive corner-badge search (see ocr.NameReader) can lock on early
    and stay locked for the rest of the video rather than re-searching from
    scratch each bout.
    """
    name_reader = NameReader(cfg.name_min_conf, cfg.name_target_h)
    taken: set = set()
    w, h = meta["width"], meta["height"]

    locked = {
        OVERLAY_LEFT: overlay_clusterer.locked_box(OVERLAY_LEFT, (h, w)),
        OVERLAY_RIGHT: overlay_clusterer.locked_box(OVERLAY_RIGHT, (h, w)),
    }
    for side_name, box in locked.items():
        if box is None:
            print(f"[names] {side_name} position never locked during analysis - "
                  f"falling back to a one-off detection on each seeked frame "
                  f"for that side (slower, but still far cheaper than "
                  f"continuous per-frame sampling)")

    def _box_for(side_class: str, frame) -> Optional[Box]:
        cached = locked[side_class]
        if cached is not None:
            x1, y1, x2, y2 = cached
            return Box(x1, y1, x2, y2, conf=1.0, name=side_class)
        if detector is None:
            return None
        dets = detector.detect([frame])[0]
        return best(dets, side_class)

    for bout in bouts:
        left_reads: List[Optional[str]] = []
        right_reads: List[Optional[str]] = []

        for anchor in _anchor_timestamps(bout.start, bout.end):
            for off in _BURST_OFFSETS_S:
                ts = max(bout.start, min(bout.end, anchor + off))
                frame = decode.extract_frame(video, ts, meta)
                if frame is None:
                    continue
                lb = _box_for(OVERLAY_LEFT, frame)
                rb = _box_for(OVERLAY_RIGHT, frame)
                if lb is not None:
                    left_reads.append(name_reader.read(frame, lb))
                if rb is not None:
                    right_reads.append(name_reader.read(frame, rb))

        left = naming.vote(left_reads)
        right = naming.vote(right_reads)
        fname = naming.bout_filename(left, right, bout.index, cfg.space_char)
        fname = naming.dedupe(fname, taken)

        bout.left_plate = left
        bout.right_plate = right
        bout.filename = fname

    return bouts


def write_report(video: Path, records, boundaries, bouts, stats, cfg: Config) -> Path:
    out = Path(cfg.workdir) / f"{video.stem}_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": str(video),
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "stats": stats,
        "boundaries": [
            {"label": b.label, "sample_index": b.sample_index,
             "timestamp": b.timestamp} for b in boundaries
        ],
        "bouts": [asdict(b) for b in bouts],
    }
    out.write_text(json.dumps(payload, indent=2))

    # The original's plain-text log, kept for compatibility.
    txt = Path(cfg.workdir) / f"start_and_end_{video.stem}.txt"
    with txt.open("w") as f:
        for b in boundaries:
            hh = time.strftime("%H:%M:%S", time.gmtime(b.timestamp))
            f.write(f"{b.label}: Frame {b.sample_index}, Timestamp: {hh}\n")
    return out


def process(video: Path, cfg: Config) -> List[Path]:
    from .cut import cut_all

    records, boundaries, stats, overlay_clusterer = analyse(video, cfg)
    bouts = build_bouts(records, boundaries, cfg)

    meta = decode.probe(video)
    bouts = resolve_bout_names(video, bouts, overlay_clusterer, cfg, meta)

    report = write_report(video, records, boundaries, bouts, stats, cfg)

    print(f"\n[analyse] {stats['frames']} frames in {stats['seconds']}s "
          f"({stats['frames'] / max(stats['seconds'], 1e-6):.1f} f/s) "
          f"for {stats['duration'] / 60:.0f} min of video")
    print(f"[ocr] template hits: {stats['template_hits']}  "
          f"tesseract calls: {stats['tesseract_calls']}  "
          f"low-confidence reads dropped: {stats['rejected']}  "
          f"position clusters: {stats['clusters_created']}")
    print(f"[names] overlay box locked without re-detection - "
          f"left: {stats['overlay_left_locked']}  right: {stats['overlay_right_locked']}")
    if stats['clock_mode'] != 'disabled':
        print(f"[start] clock reader status: {stats['clock_mode']}")
        if stats['clock_mode'] == 'none':
            print(f"[start] WARNING: clock was never located - the clock-decrease "
                  f"start signal will never fire for this video; confirmed "
                  f"score-change (60s lookback) is the only signal that can "
                  f"place a start marker")
    if stats['clusters_created'] > 6:
        print(f"[ocr] WARNING: {stats['clusters_created']} distinct score-box "
              f"positions detected - expected ~2 (one per side) for a stable "
              f"broadcast overlay. This many suggests the detector's boxes "
              f"are jittering unusually widely, or the overlay position is "
              f"genuinely changing mid-stream; either way the template bank "
              f"is getting less benefit than it should.")
    print(f"[bouts] {len(bouts)} detected -> {report}")

    jobs = [(b.start, b.end, b.filename) for b in bouts]
    return cut_all(video, jobs, Path(cfg.outdir), cfg.cut_mode)
