"""Measure how many concurrent hardware decodes your GPU actually sustains.

This does not look up your card's spec sheet - it launches N ffmpeg
`-hwaccel cuda` decodes of the same file at once, for increasing N, and times
each round. Real physical decode-engine contention shows up as aggregate
throughput flattening out (or per-stream time climbing) once N exceeds the
engine count; overlap that's still winning shows up as aggregate throughput
still climbing.

Usage:
    python tools/probe_concurrency.py your_video.mp4
    python tools/probe_concurrency.py your_video.mp4 --max-n 8 --seconds 20

Read the output: the row where "total throughput" stops increasing (or drops)
is roughly your hardware decode-engine count. Set fenceseg batch max_workers
at or slightly below that row, then account for CPU/OCR budget on top (see
batch.py's module docstring) - the decode-engine number is a ceiling on GPU
decode parallelism, not the final answer for max_workers.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_one(video: Path, seconds: float, hwaccel: str) -> float:
    """Decode `seconds` of the file once. Returns wall-clock seconds taken."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(video), "-t", str(seconds), "-f", "null", "-"]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True)
    dt = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode("utf-8", "replace")[-500:])
    return dt


def probe_n(video: Path, n: int, seconds: float, hwaccel: str) -> dict:
    with ThreadPoolExecutor(max_workers=n) as ex:
        t0 = time.time()
        futs = [ex.submit(run_one, video, seconds, hwaccel) for _ in range(n)]
        per_stream = [f.result() for f in futs]
    wall = time.time() - t0
    # Throughput: total seconds of source video decoded per wall-clock second.
    throughput = (n * seconds) / wall
    return {
        "n": n,
        "wall_s": wall,
        "per_stream_avg_s": sum(per_stream) / len(per_stream),
        "per_stream_max_s": max(per_stream),
        "throughput_x": throughput,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--max-n", type=int, default=6,
                    help="highest concurrency level to test")
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="seconds of video each concurrent decode processes")
    ap.add_argument("--hwaccel", default="cuda",
                    help="pass '' to probe software-decode concurrency instead")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"not found: {args.video}")

    print(f"probing {args.video.name}  hwaccel={args.hwaccel or '(software)'}  "
          f"{args.seconds}s per stream\n")
    print(f"{'N':>3}  {'wall(s)':>8}  {'avg/stream(s)':>14}  "
          f"{'max/stream(s)':>14}  {'throughput':>11}")

    rows = []
    for n in range(1, args.max_n + 1):
        r = probe_n(args.video, n, args.seconds, args.hwaccel)
        rows.append(r)
        print(f"{r['n']:>3}  {r['wall_s']:>8.2f}  {r['per_stream_avg_s']:>14.2f}  "
              f"{r['per_stream_max_s']:>14.2f}  {r['throughput_x']:>9.2f}x")

    print("\nLook for where throughput stops climbing (or per-stream time")
    print("starts climbing) as N increases - that row is roughly your")
    print("hardware's concurrent decode ceiling for this codec/resolution.")

    best = max(rows, key=lambda r: r["throughput_x"])
    print(f"\nBest observed: N={best['n']}  ({best['throughput_x']:.2f}x "
          f"realtime aggregate)")


if __name__ == "__main__":
    raise SystemExit(main())
