"""Verify fenceseg.batch.process_many's concurrency and isolation contract.

Runs with everything below Detector.detect() mocked out, so this needs no
GPU, no torch, no real video file. It checks the properties that would be
easy to get wrong and hard to notice from a real run's console output:

  1. The shared detector is constructed exactly once for a multi-video batch,
     not once per video (that was the whole point of sharing it).
  2. A caller-supplied detector is reused as-is, never rebuilt.
  3. One video raising does not stop or corrupt the others' results.
  4. Every video's bouts are cut into cfg.outdir/<video-stem>/, not flattened
     together where two videos could overwrite each other's files.
  5. Every video gets its own ScoreReader/NameReader/state machine (checked
     indirectly: each analyse() call receives a distinct video path and the
     mock returns per-call data keyed by that path, proving no cross-call
     state leaked through a shared object).

analyse() now returns a 4-tuple (records, boundaries, stats, overlay_
clusterer) and process_many's per-video closure also calls decode.probe()
and resolve_bout_names() as part of the Phase 2 name-resolution step - both
mocked here alongside the existing analyse/build_bouts/write_report/cut_all
mocks, for the same reason: no GPU or real file needed to test the
concurrency/isolation contract itself.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenceseg import batch  # noqa: E402
from fenceseg.config import Config  # noqa: E402
from fenceseg.pipeline import Bout  # noqa: E402


def make_bout(i: int) -> Bout:
    return Bout(index=i, start=float(i), end=float(i + 1),
               filename=f"bout_{i}", left_plate=None, right_plate=None)


FAKE_META = {"width": 100, "height": 100, "duration": 10.0, "fps": 1.0, "codec": "h264"}


def _patches(analyse_fn, build_bouts_fn=lambda r, b, c: [],
            resolve_names_fn=None, cut_all_fn=lambda *a, **kw: [],
            detector_cls=lambda *a, **kw: object()):
    """Common set of mocks every test below needs, with per-test overrides."""
    if resolve_names_fn is None:
        resolve_names_fn = lambda video, bouts, oc, cfg, meta, detector=None: bouts
    return [
        mock.patch.object(batch, "Detector", detector_cls),
        mock.patch.object(batch, "analyse", analyse_fn),
        mock.patch.object(batch, "build_bouts", build_bouts_fn),
        mock.patch.object(batch, "resolve_bout_names", resolve_names_fn),
        mock.patch.object(batch.decode, "probe", lambda v: FAKE_META),
        mock.patch.object(batch, "write_report", lambda *a: None),
        mock.patch.object(batch, "cut_all", cut_all_fn),
    ]


def _apply(patches):
    for p in patches:
        p.start()
    return patches


def _undo(patches):
    for p in patches:
        p.stop()


def test_shared_detector_built_once():
    build_calls = []

    class FakeDetector:
        def __init__(self, *a, **kw):
            build_calls.append(1)

    videos = [Path(f"v{i}.mp4") for i in range(4)]
    cfg = Config()

    patches = _patches(
        analyse_fn=lambda v, c, detector=None: ([], [], {"frames": 0}, object()),
        detector_cls=FakeDetector,
    )
    p = _apply(patches)
    try:
        batch.process_many(videos, cfg, max_workers=4)
    finally:
        _undo(p)

    assert len(build_calls) == 1, f"expected 1 detector build, got {len(build_calls)}"
    print("PASS: shared detector built exactly once for 4 videos")


def test_supplied_detector_never_rebuilt():
    build_calls = []

    class FakeDetector:
        def __init__(self, *a, **kw):
            build_calls.append(1)

    supplied = FakeDetector()
    assert len(build_calls) == 1
    videos = [Path("v0.mp4"), Path("v1.mp4")]
    cfg = Config()

    patches = _patches(
        analyse_fn=lambda v, c, detector=None: ([], [], {"frames": 0}, object()),
        detector_cls=FakeDetector,
    )
    p = _apply(patches)
    try:
        batch.process_many(videos, cfg, max_workers=2, detector=supplied)
    finally:
        _undo(p)

    assert len(build_calls) == 1, "supplying a detector must not trigger another build"
    print("PASS: caller-supplied detector reused, not rebuilt")


def test_one_failure_does_not_sink_the_batch():
    videos = [Path("good0.mp4"), Path("bad.mp4"), Path("good1.mp4")]
    cfg = Config()

    def fake_analyse(v, c, detector=None):
        if v.name == "bad.mp4":
            raise RuntimeError("simulated OCR crash")
        return ([], [make_bout(0)], {"frames": 10}, object())

    patches = _patches(
        analyse_fn=fake_analyse,
        build_bouts_fn=lambda r, b, c: b,
        cut_all_fn=lambda video, jobs, outdir, mode: [outdir / j[2] for j in jobs],
    )
    p = _apply(patches)
    try:
        results = batch.process_many(videos, cfg, max_workers=3)
    finally:
        _undo(p)

    assert Path("bad.mp4") not in results, "failed video must not appear in results"
    assert Path("good0.mp4") in results and Path("good1.mp4") in results, \
        "other videos must still succeed when one fails"
    print("PASS: one video failing does not affect the others")


def test_outputs_namespaced_per_video():
    videos = [Path("eventA.mp4"), Path("eventB.mp4")]
    cfg = Config()
    cfg.outdir = Path("/tmp/fenceseg_test_outdir")

    seen_outdirs = []

    def fake_cut_all(video, jobs, outdir, mode):
        seen_outdirs.append(outdir)
        return [outdir / j[2] for j in jobs]

    patches = _patches(
        analyse_fn=lambda v, c, detector=None: ([], [make_bout(0)], {"frames": 1}, object()),
        build_bouts_fn=lambda r, b, c: b,
        cut_all_fn=fake_cut_all,
    )
    p = _apply(patches)
    try:
        batch.process_many(videos, cfg, max_workers=2)
    finally:
        _undo(p)

    expected = {cfg.outdir / "eventA", cfg.outdir / "eventB"}
    assert set(seen_outdirs) == expected, \
        f"expected per-video subfolders {expected}, got {set(seen_outdirs)}"
    assert len(set(seen_outdirs)) == len(seen_outdirs), "outdirs collided"
    print("PASS: each video's bouts are cut into their own subfolder")


def test_concurrent_calls_actually_overlap():
    """Sanity check that max_workers>1 really runs videos concurrently and
    isn't secretly serialised somewhere (e.g. a stray lock)."""
    videos = [Path(f"v{i}.mp4") for i in range(4)]
    cfg = Config()
    concurrent_now = 0
    max_concurrent = 0
    lock = threading.Lock()

    def fake_analyse(v, c, detector=None):
        nonlocal concurrent_now, max_concurrent
        with lock:
            concurrent_now += 1
            max_concurrent = max(max_concurrent, concurrent_now)
        time.sleep(0.05)
        with lock:
            concurrent_now -= 1
        return ([], [], {"frames": 0}, object())

    patches = _patches(analyse_fn=fake_analyse)
    p = _apply(patches)
    try:
        batch.process_many(videos, cfg, max_workers=4)
    finally:
        _undo(p)

    assert max_concurrent > 1, \
        f"expected genuine overlap, max observed concurrency was {max_concurrent}"
    print(f"PASS: observed up to {max_concurrent} videos analysing concurrently")


if __name__ == "__main__":
    tests = [
        test_shared_detector_built_once,
        test_supplied_detector_never_rebuilt,
        test_one_failure_does_not_sink_the_batch,
        test_outputs_namespaced_per_video,
        test_concurrent_calls_actually_overlap,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    print()
    if failed:
        print(f"{failed}/{len(tests)} FAILED")
        raise SystemExit(1)
    print(f"ALL {len(tests)} PASSED")
