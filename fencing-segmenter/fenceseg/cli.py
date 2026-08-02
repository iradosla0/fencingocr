"""Command line interface.

    # already have the file
    python -m fenceseg.cli run stream.mp4 --weights best.pt

    # download from FencingTV first (URL = the m3u8 from DevTools)
    python -m fenceseg.cli run "https://.../master.m3u8" --weights best.pt

    # a list of URLs and/or local files, one per line
    python -m fenceseg.cli run urls.txt --weights best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fenceseg", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="download if needed, analyse, cut")
    r.add_argument("input", help="video file, URL, or a .txt list of either")
    r.add_argument("--weights", type=Path, default=Path("best.pt"))
    r.add_argument("--workdir", type=Path, default=Path("work"))
    r.add_argument("--outdir", type=Path, default=Path("bouts"))

    r.add_argument("--sample-fps", type=float, default=1.0,
                   help="decision rate; 1.0 matches the original notebook")
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--hwaccel", default=None,
                   help="ffmpeg decoder, e.g. cuda / videotoolbox / qsv")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--imgsz", type=int, default=640)
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--iou", type=float, default=0.45)
    r.add_argument("--no-half", action="store_true")

    r.add_argument("--score-min-conf", type=float, default=55.0)
    r.add_argument("--name-min-conf", type=float, default=45.0)
    r.add_argument("--no-templates", action="store_true",
                   help="disable the learned digit template bank")
    r.add_argument("--temporal-vote", action="store_true",
                   help="enable temporal outlier rejection (see README)")

    r.add_argument("--start-mode", choices=["legacy", "clock_or_score"], default="legacy",
                   help="'legacy' (default): the original, protected start "
                        "condition, unchanged. 'clock_or_score': place the "
                        "start marker 30s before a confirmed clock-decrease, "
                        "or 60s before a confirmed score-change if that "
                        "happens first / the clock is unreadable. See "
                        "fenceseg/adaptive_start.py.")
    r.add_argument("--start-score-lookback", type=float, default=60.0)
    r.add_argument("--start-clock-lookback", type=float, default=30.0)
    r.add_argument("--start-score-confirm-frames", type=int, default=2)
    r.add_argument("--start-clock-confirm-frames", type=int, default=2)

    r.add_argument("--cut-mode", choices=["copy", "reencode"], default="copy")
    r.add_argument("--pad-start", type=float, default=0.0)
    r.add_argument("--pad-end", type=float, default=0.0)
    r.add_argument("--underscores", action="store_true",
                   help="use underscores instead of spaces inside names")

    r.add_argument("--fragments", type=int, default=8,
                   help="concurrent HLS fragment downloads")
    r.add_argument("--no-pause", action="store_true",
                   help="skip the randomised break between downloads")
    r.add_argument("--analyse-only", action="store_true",
                   help="detect boundaries and write the report, do not cut")
    r.add_argument("--workers", type=int, default=1,
                   help="process this many videos concurrently, sharing one "
                        "loaded detector. See fenceseg/batch.py's docstring "
                        "before setting this above 1 - the right number "
                        "depends on your GPU's decode-engine count and CPU "
                        "cores, not a guess. Ignored for a single video.")
    return p


def cfg_from_args(a) -> Config:
    return Config(
        weights=a.weights, workdir=a.workdir, outdir=a.outdir,
        sample_fps=a.sample_fps, batch_size=a.batch_size, hwaccel=a.hwaccel,
        device=a.device, half=not a.no_half, imgsz=a.imgsz,
        conf_thres=a.conf, iou_thres=a.iou,
        score_min_conf=a.score_min_conf, name_min_conf=a.name_min_conf,
        use_templates=not a.no_templates, temporal_vote=a.temporal_vote,
        start_mode=a.start_mode,
        start_score_lookback_s=a.start_score_lookback,
        start_clock_lookback_s=a.start_clock_lookback,
        start_score_confirm_frames=a.start_score_confirm_frames,
        start_clock_confirm_frames=a.start_clock_confirm_frames,
        cut_mode=a.cut_mode, pad_start=a.pad_start, pad_end=a.pad_end,
        space_char="_" if a.underscores else " ",
        concurrent_fragments=a.fragments,
        break_between_downloads=not a.no_pause,
    )


def resolve_inputs(spec: str, cfg: Config):
    from . import download

    items = []
    p = Path(spec)
    if p.suffix.lower() == ".txt" and p.exists():
        items = download.read_url_list(p)
    else:
        items = [spec]

    videos = []
    urls = [i for i in items if i.startswith("http")]
    files = [Path(i) for i in items if not i.startswith("http")]

    for f in files:
        if not f.exists():
            raise FileNotFoundError(f)
        videos.append(f)

    if urls:
        videos += download.download_many(
            urls, cfg.workdir, cfg.concurrent_fragments,
            cfg.break_between_downloads
        )
    return videos


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = cfg_from_args(args)

    if not cfg.weights.exists():
        print(f"weights not found: {cfg.weights}", file=sys.stderr)
        return 2

    cfg.workdir.mkdir(parents=True, exist_ok=True)
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    videos = resolve_inputs(args.input, cfg)
    if not videos:
        print("nothing to process", file=sys.stderr)
        return 2

    from .pipeline import analyse, build_bouts, process, write_report

    if args.workers > 1 and len(videos) > 1 and not args.analyse_only:
        from .batch import process_many
        results = process_many(videos, cfg, max_workers=args.workers)
        total = sum(len(v) for v in results.values())
        print(f"[done] {total} bouts written across {len(results)} video(s) "
              f"under {cfg.outdir}/<video-stem>/")
        return 0 if len(results) == len(videos) else 1

    for v in videos:
        if args.analyse_only:
            from .pipeline import resolve_bout_names
            from . import decode as _decode
            records, boundaries, stats, overlay_clusterer = analyse(v, cfg)
            bouts = build_bouts(records, boundaries, cfg)
            meta = _decode.probe(v)
            bouts = resolve_bout_names(v, bouts, overlay_clusterer, cfg, meta)
            report = write_report(v, records, boundaries, bouts, stats, cfg)
            print(f"[report] {report}")
            for b in bouts:
                print(f"  {b.start:8.1f}-{b.end:8.1f}  {b.filename}")
        else:
            written = process(v, cfg)
            print(f"[done] {len(written)} bouts written to {cfg.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
