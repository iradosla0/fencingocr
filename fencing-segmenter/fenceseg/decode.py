"""Frame sampling.

The original notebook did this:

    while cap.isOpened():
        ret, frame = cap.read()          # decodes EVERY frame
        if frame_count % int(fps) == 0:  # ...then throws 96% of them away
            process(frame)

`cap.read()` fully decodes and colour-converts every frame in the file. For a
3 hour 30fps stream that is ~324,000 full decodes to obtain ~10,800 usable
frames. Decoding, not inference, was the dominant cost of the whole pipeline.

Here ffmpeg applies the `fps` filter inside the decoder graph, so frames are
dropped before colour conversion and only the sampled frames ever cross into
Python. Optional `-hwaccel` moves the decode onto the GPU as well.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np


class ProbeError(RuntimeError):
    pass


def probe(path: Path) -> dict:
    """Return {width, height, duration, fps, codec} for the first video stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,avg_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise ProbeError(f"ffprobe failed on {path}: {res.stderr.strip()}")
    data = json.loads(res.stdout)
    if not data.get("streams"):
        raise ProbeError(f"no video stream in {path}")
    st = data["streams"][0]

    num, _, den = st.get("avg_frame_rate", "0/1").partition("/")
    try:
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "codec": st.get("codec_name", ""),
        "fps": fps,
        "duration": float(data.get("format", {}).get("duration", 0.0)),
    }


def iter_sampled_frames(
    path: Path,
    sample_fps: float = 1.0,
    hwaccel: Optional[str] = None,
    meta: Optional[dict] = None,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (sample_index, timestamp_seconds, BGR frame) at `sample_fps`.

    Frames are delivered at full source resolution. That matters: the score
    digits are only ~15-25px tall in a 1080p broadcast and downscaling before
    OCR was never going to be recoverable.
    """
    meta = meta or probe(path)
    w, h = meta["width"], meta["height"]
    frame_bytes = w * h * 3

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += [
        "-i", str(path),
        "-map", "0:v:0",
        "-vf", f"fps={sample_fps}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=frame_bytes * 4)
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            yield idx, idx / sample_fps, frame
            idx += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.stdout.close()
        except Exception:
            pass
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.stderr.close()
        proc.wait()
        if proc.returncode not in (0, None) and idx == 0:
            raise RuntimeError(f"ffmpeg decode failed for {path}:\n{err}")


def iter_batches(
    path: Path,
    sample_fps: float = 1.0,
    batch_size: int = 32,
    hwaccel: Optional[str] = None,
    meta: Optional[dict] = None,
):
    """Group `iter_sampled_frames` into batches for the detector."""
    batch = []
    for idx, ts, frame in iter_sampled_frames(path, sample_fps, hwaccel, meta):
        batch.append((idx, ts, frame))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_frame(path: Path, timestamp: float, meta: Optional[dict] = None) -> Optional[np.ndarray]:
    """Extract exactly one frame at `timestamp` via a targeted ffmpeg seek.

    Used for sparse, targeted sampling (e.g. reading a name plate at a
    handful of specific instants in a bout) rather than decoding the whole
    stream. A seek-then-decode-one-frame call like this is cheap - ffmpeg
    seeks to the nearest preceding keyframe and decodes forward only as far
    as needed, not the whole file - so calling this a few times per bout
    costs nothing close to a second full decode pass.

    Returns None if the timestamp is out of range or ffmpeg produced no
    frame (e.g. right at the very end of the file).
    """
    meta = meta or probe(path)
    w, h = meta["width"], meta["height"]
    frame_bytes = w * h * 3

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{max(0.0, timestamp):.3f}",
        "-i", str(path),
        "-frames:v", "1",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or len(res.stdout) < frame_bytes:
        return None
    return np.frombuffer(res.stdout[:frame_bytes], dtype=np.uint8).reshape(h, w, 3)
