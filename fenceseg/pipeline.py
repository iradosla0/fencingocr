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
from .config import Config
from .detect import (FENCER, OVERLAY_LEFT, OVERLAY_RIGHT, SCORE_LEFT,
                     SCORE_RIGHT, Detector, best)
from .ocr import NameReader, ScoreReader, TemporalScoreFilter
from .segment import BoutStateMachine, FrameObs, pair_boundaries, unterminated

# Name plates are static for the length of a bout, so reading one every
# NAME_EVERY sampled frames is plenty for a confident majority vote and keeps
# two extra OCR calls off the hot path.
NAME_EVERY = 5


@dataclass
class Record:
    idx: int
    ts: float
    fencers: int
    sl: Optional[int]
    sr: Optional[int]
    name_l: Optional[str] = None
    name_r: Optional[str] = None


@dataclass
class Bout:
    index: int
    start: float
    end: float
    filename: str
    left_plate: Optional[str]
    right_plate: Optional[str]


def analyse(video: Path, cfg: Config) -> Tuple[List[Record], List, dict]:
    meta = decode.probe(video)
    print(f"[analyse] {video.name}  {meta['width']}x{meta['height']} "
          f"{meta['fps']:.2f}fps  {meta['duration'] / 60:.1f}min  "
          f"codec={meta['codec']}")

    detector = Detector(cfg.weights, cfg.device, cfg.conf_thres,
                        cfg.iou_thres, cfg.half, cfg.imgsz)
    score_reader = ScoreReader(cfg.score_min_conf, cfg.score_target_h,
                               cfg.use_templates, cfg.template_match_thres)
    name_reader = NameReader(cfg.name_min_conf, cfg.name_target_h)
    tfilter = TemporalScoreFilter() if cfg.temporal_vote else None

    sm = BoutStateMachine(waiting_for_start=True)
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

            sl = sr = None
            nl = nr = None

            # Only pay for OCR on frames the guard would let through. Frames
            # failing the guard contribute nothing in the original either.
            if fencers == 2 and bl is not None and br is not None:
                sl = score_reader.read(frame, bl)
                sr = score_reader.read(frame, br)
                if tfilter is not None:
                    sl = tfilter.filter("L", sl, ts)
                    sr = tfilter.filter("R", sr, ts)

                if idx % NAME_EVERY == 0:
                    ol = best(boxes, OVERLAY_LEFT)
                    orr = best(boxes, OVERLAY_RIGHT)
                    if ol is not None:
                        nl = name_reader.read(frame, ol, exclude=bl)
                    if orr is not None:
                        nr = name_reader.read(frame, orr, exclude=br)

            records.append(Record(idx, ts, fencers, sl, sr, nl, nr))

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
    return records, sm.boundaries, stats


def build_bouts(records: List[Record], boundaries: List, cfg: Config) -> List[Bout]:
    by_idx: Dict[int, Record] = {r.idx: r for r in records}
    pairs = pair_boundaries(boundaries)

    dangling = unterminated(boundaries)
    if dangling is not None:
        print(f"[bouts] a start at {dangling.timestamp:.0f}s was never closed "
              f"by an end condition; not exported")

    bouts: List[Bout] = []
    taken: set = set()
    for i, (s, e) in enumerate(pairs, start=1):
        window = [by_idx[k] for k in range(s.sample_index, e.sample_index + 1)
                  if k in by_idx]
        left = naming.vote([r.name_l for r in window])
        right = naming.vote([r.name_r for r in window])
        fname = naming.bout_filename(left, right, i, cfg.space_char)
        fname = naming.dedupe(fname, taken)
        bouts.append(Bout(
            index=i,
            start=max(0.0, s.timestamp - cfg.pad_start),
            end=e.timestamp + cfg.pad_end,
            filename=fname,
            left_plate=left,
            right_plate=right,
        ))
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

    records, boundaries, stats = analyse(video, cfg)
    bouts = build_bouts(records, boundaries, cfg)
    report = write_report(video, records, boundaries, bouts, stats, cfg)

    print(f"\n[analyse] {stats['frames']} frames in {stats['seconds']}s "
          f"({stats['frames'] / max(stats['seconds'], 1e-6):.1f} f/s) "
          f"for {stats['duration'] / 60:.0f} min of video")
    print(f"[ocr] template hits: {stats['template_hits']}  "
          f"tesseract calls: {stats['tesseract_calls']}  "
          f"low-confidence reads dropped: {stats['rejected']}")
    print(f"[bouts] {len(bouts)} detected -> {report}")

    jobs = [(b.start, b.end, b.filename) for b in bouts]
    return cut_all(video, jobs, Path(cfg.outdir), cfg.cut_mode)
