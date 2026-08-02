"""Central configuration for the fencing bout segmenter."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # ---- paths -------------------------------------------------------
    weights: Path = Path("best.pt")
    workdir: Path = Path("work")
    outdir: Path = Path("bouts")

    # ---- sampling ----------------------------------------------------
    # Decision rate. The original notebook evaluated one frame per second
    # (frame_count % int(fps) == 0). Keep 1.0 for behavioural parity.
    sample_fps: float = 1.0
    # Frames pushed through YOLO per forward pass. Pure throughput knob:
    # has no effect on which frames are evaluated or on the decisions made.
    batch_size: int = 32
    # ffmpeg hardware decoder, e.g. "cuda", "videotoolbox", "qsv". None = software.
    hwaccel: Optional[str] = None

    # ---- detection ---------------------------------------------------
    device: str = "cuda:0"
    half: bool = True
    imgsz: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45

    # ---- OCR ---------------------------------------------------------
    # Minimum mean per-character confidence (0-100) for a score read to be
    # accepted. Below this the read returns None, which the state machine
    # treats exactly like the original treated a failed OCR: the frame is
    # skipped and prev_scores is left untouched.
    score_min_conf: float = 55.0
    name_min_conf: float = 45.0
    # Target pixel height that score/name crops are upscaled to before OCR.
    score_target_h: int = 64
    name_target_h: int = 48
    # Learn per-broadcast digit templates on the fly (big speed + accuracy win).
    use_templates: bool = True
    template_match_thres: float = 0.72
    # Optional temporal outlier rejection. OFF by default: it can shift the
    # exact frame a boundary lands on. See README "Temporal voting".
    temporal_vote: bool = False

    # ---- cutting -----------------------------------------------------
    # "copy"      -> stream copy, near-instant, cut snaps to nearest keyframe
    # "reencode"  -> frame-accurate, slower, uses nvenc when available
    cut_mode: str = "copy"
    # Seconds of padding added either side of a detected bout.
    pad_start: float = 0.0
    pad_end: float = 0.0
    # Character placed between name parts in the output filename.
    space_char: str = " "

    # ---- download ----------------------------------------------------
    concurrent_fragments: int = 8
    break_between_downloads: bool = True

    # ---- start-condition mode -----------------------------------------
    # "legacy" (default): the original, protected, fuzz-tested start
    # condition - unchanged, verified byte-for-byte against the source
    # notebook. Nothing below this line has any effect unless start_mode is
    # switched to "clock_or_score".
    #
    # "clock_or_score": an explicitly requested new rule. While waiting for
    # a bout to start, watches for a CONFIRMED clock-decrease or a
    # CONFIRMED score-change, whichever happens first, and places the start
    # marker some lookback before that event. See fenceseg/adaptive_start.py
    # for the full rationale, including why the clock signal needs a
    # heuristic search-and-lock (no trained detector class exists for the
    # timer) and is therefore less certain than the score/name reading
    # elsewhere in this pipeline, which all locate their target via the
    # trained model first.
    start_mode: str = "legacy"
    start_score_lookback_s: float = 60.0
    start_clock_lookback_s: float = 30.0
    start_score_confirm_frames: int = 2
    start_clock_confirm_frames: int = 2
    start_min_gap_after_previous_end_s: float = 0.0
    clock_min_conf: float = 30.0
    clock_search_attempts: int = 10
    clock_lock_threshold: int = 3

    extra: dict = field(default_factory=dict)
