"""Downloading, adapted for FencingTV.

Replaces the notebook's YouTube-only `download_video` and follows the approach
in andrefisch/VideoDownloadScript, which is what FencingTV requires.

Two things about FencingTV drive the design:

1. The URL is an HLS master playlist (`.m3u8`) pulled by hand out of Chrome
   DevTools -> Network -> filter `.m3u8` -> the entry with the random-looking
   name (not the "rendition" ones) -> Request URL. There is no page-level API
   to discover it, so the URL still comes from you.

2. An m3u8 carries no title metadata. yt-dlp's `%(title)s` will produce
   something meaningless, so an explicit output name is required. This does
   not matter for the final bout files -- those are named from the overlay --
   but it keeps the working directory legible.

The one real download speedup available here is fragment concurrency. HLS
streams are thousands of small segments fetched serially by default;
`concurrent_fragment_downloads` fetches N at once, which for a multi-hour
stream is usually the difference between an hour and a few minutes.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Iterable, List, Optional


def download(url: str, out_path: Path, concurrent_fragments: int = 8,
             fmt: str = "bestvideo+bestaudio/best",
             quiet: bool = False) -> Path:
    """Download one URL (m3u8, YouTube, anything yt-dlp handles) to out_path."""
    import yt_dlp

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    opts = {
        "format": fmt,
        "outtmpl": str(out_path.with_suffix("")) + ".%(ext)s",
        "merge_output_format": "mp4",
        "concurrent_fragment_downloads": max(1, int(concurrent_fragments)),
        "retries": 10,
        "fragment_retries": 20,
        "skip_unavailable_fragments": False,
        "continuedl": True,
        "quiet": quiet,
        "noprogress": quiet,
        # Keep the container seekable so the cutting stage can seek cheaply.
        "postprocessor_args": {"merger": ["-movflags", "+faststart"]},
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    stem = out_path.with_suffix("")
    for cand in sorted(stem.parent.glob(stem.name + ".*")):
        if cand.suffix.lower() in {".mp4", ".mkv", ".ts", ".webm", ".mov"}:
            return cand
    raise FileNotFoundError(f"yt-dlp produced no output for {url}")


def read_url_list(path: Path) -> List[str]:
    """One URL per line; blank lines and # comments ignored."""
    urls = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def polite_break(enabled: bool = True, lo: int = 180, hi: int = 360) -> None:
    """The randomised pause between downloads from download.sh."""
    if not enabled:
        return
    t = random.randint(lo, hi)
    print(f"[download] pausing {t}s before the next stream")
    time.sleep(t)


def download_many(urls: Iterable[str], workdir: Path,
                  concurrent_fragments: int = 8,
                  pause_between: bool = True) -> List[Path]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    urls = list(urls)
    for i, url in enumerate(urls, start=1):
        target = workdir / f"stream_{i:02d}.mp4"
        print(f"[download] {i}/{len(urls)}: {url}")
        paths.append(download(url, target, concurrent_fragments))
        if i < len(urls):
            polite_break(pause_between)
    return paths
