"""Cutting bouts out of the source file.

Two things the notebook did that cost a great deal of time and bought nothing:

  * It re-downloaded the entire livestream in cell 5 in order to cut it, after
    having already downloaded it in cell 3. The file is already on disk.

  * It split the source into 3-hour chunks and, if the video was <=1280 wide or
    AV1, re-encoded every chunk through nvenc before analysis. Analysis reads
    frames through ffmpeg either way, which decodes AV1 and any resolution
    natively, so the re-encode was pure overhead -- typically hours of GPU time
    per stream.

Cut modes:
  copy      stream copy. Effectively instant. The cut snaps to the nearest
            preceding keyframe, so the start can land up to one GOP (commonly
            2-10s on an HLS stream) early. Given boundaries are detected on a
            1 Hz grid anyway, this is usually the right default.
  reencode  frame-accurate. Re-encodes only the bout, not the stream. Uses
            h264_nvenc when present, falls back to libx264.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


def _has_encoder(name: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=20).stdout
        return name in out
    except Exception:
        return False


_NVENC: Optional[bool] = None


def has_nvenc() -> bool:
    global _NVENC
    if _NVENC is None:
        _NVENC = _has_encoder("h264_nvenc")
    return _NVENC


def cut(src: Path, start: float, end: float, dest: Path,
        mode: str = "copy") -> bool:
    """Extract [start, end) from src into dest. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)
    if duration <= 0:
        return False

    if mode == "copy":
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", f"{start:.3f}", "-i", str(src),
            "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(dest),
        ]
    else:
        vcodec = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"] if has_nvenc() \
            else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            # Coarse input seek to a keyframe before the target, then an
            # accurate output seek for the remainder. Fast and frame-exact.
            "-ss", f"{max(0.0, start - 10):.3f}", "-i", str(src),
            "-ss", f"{min(10.0, start):.3f}",
            "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            *vcodec,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(dest),
        ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[cut] ffmpeg failed for {dest.name}: {res.stderr.strip()[:400]}")
        if dest.exists():
            dest.unlink()
        return False
    return dest.exists() and dest.stat().st_size > 0


def cut_all(src: Path, jobs: List[Tuple[float, float, str]], outdir: Path,
            mode: str = "copy", ext: str = ".mp4") -> List[Path]:
    """jobs = [(start_s, end_s, filename_without_extension), ...]"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (start, end, name) in enumerate(jobs, start=1):
        dest = outdir / f"{name}{ext}"
        print(f"[cut] {i}/{len(jobs)} {start:8.1f}-{end:8.1f}s  {dest.name}")
        if cut(src, start, end, dest, mode):
            written.append(dest)
    return written
