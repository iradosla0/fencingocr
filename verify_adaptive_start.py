"""Verify fenceseg.adaptive_start.AdaptiveBoutStateMachine.

Two independent things get checked:

1. END-condition parity against the protected fenceseg.segment.BoutStateMachine.
   adaptive_start.py does not import or modify segment.py - it reproduces the
   end condition independently, by design, so this class has no runtime
   dependency on the protected module's internals. That means nothing
   structurally guarantees the two stay in sync; this test is what does.
   Same random-sequence generator as tools/verify_state_machine.py, run
   through both classes, requiring identical END-firing behaviour (label and
   timestamp) at every step.

2. The new START behaviour: clock-confirmed-first wins with a 30s lookback,
   score-confirmed-first wins with a 60s lookback (including when the clock
   is never locked at all), a single-frame misread on EITHER signal must not
   fire prematurely, and the computed marker never goes negative or before
   the previous bout's end.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenceseg.adaptive_start import AdaptiveBoutStateMachine  # noqa: E402
from fenceseg.segment import BoutStateMachine, FrameObs  # noqa: E402


# ---------------------------------------------------------------------------
# 1. END-condition parity
# ---------------------------------------------------------------------------

def realistic_seq(rng, n_bouts=6):
    """Same generator as tools/verify_state_machine.py's realistic_seq."""
    out = []
    for _ in range(n_bouts):
        for _ in range(rng.randint(1, 3)):
            out.append((2, True, True, 0, 0))
        l = r = 0
        while l < 15 and r < 15:
            if rng.random() < 0.5:
                l += 1
            else:
                r += 1
            for _ in range(rng.randint(1, 4)):
                if rng.random() < 0.12:
                    out.append((rng.choice([0, 1, 3]), True, True, l, r))
                elif rng.random() < 0.08:
                    out.append((2, True, True, None, None))
                else:
                    out.append((2, True, True, l, r))
        for _ in range(rng.randint(1, 3)):
            out.append((2, True, True, 0, 0))
    return out


def run_legacy_ends(seq):
    sm = BoutStateMachine(waiting_for_start=True)
    ends = []
    for i, (fc, hl, hr, sl, sr) in enumerate(seq):
        label = sm.step(FrameObs(i, float(i), fc, hl, hr, sl, sr))
        if label == "end":
            ends.append((i, float(i)))
    return ends


def run_adaptive_ends(seq):
    sm = AdaptiveBoutStateMachine(waiting_for_start=True)
    ends = []
    for i, (fc, hl, hr, sl, sr) in enumerate(seq):
        label = sm.step(FrameObs(i, float(i), fc, hl, hr, sl, sr))
        if label == "end":
            ends.append((i, float(i)))
    return ends


def test_end_condition_parity():
    rng = random.Random(20260802)
    failures = 0
    checked = 0
    for _ in range(3000):
        seq = realistic_seq(rng)
        checked += 1
        a, b = run_legacy_ends(seq), run_adaptive_ends(seq)
        if a != b:
            failures += 1
            if failures <= 5:
                print(f"  MISMATCH: legacy={a} adaptive={b}")
    print(f"  {checked} sequences checked, {failures} mismatches")
    assert failures == 0, f"{failures} END-condition mismatches vs the protected class"
    print("PASS: END condition is identical to the protected BoutStateMachine")


# ---------------------------------------------------------------------------
# 2. New START behaviour
# ---------------------------------------------------------------------------

def obs(i, ts, sl, sr, fc=2, hl=True, hr=True):
    return FrameObs(i, ts, fc, hl, hr, sl, sr)


def test_clock_wins_with_30s_lookback():
    sm = AdaptiveBoutStateMachine(clock_confirm_frames=2, score_confirm_frames=2)
    # Clock ticks down every frame from t=0; confirmed after 2 consecutive
    # decreases (frames at t=1,2). No score changes at all yet.
    clocks = [180.0, 179.0, 178.0, 177.0]
    label = None
    for i, c in enumerate(clocks):
        label = sm.step(obs(i, float(i), 0, 0), clock_seconds=c)
        if label:
            break
    assert label == "start", f"expected start, got {label}"
    b = sm.boundaries[-1]
    # Confirmed at i=2 (t=2.0, second consecutive decrease) -> marker = 2 - 30 (clamped to 0)
    assert b.timestamp == 0.0, f"expected clamped-to-0 marker, got {b.timestamp}"
    print("PASS: clock-confirmed start fires with a 30s lookback (clamped correctly near t=0)")


def test_clock_wins_with_30s_lookback_no_clamp():
    sm = AdaptiveBoutStateMachine(clock_confirm_frames=2)
    clocks = [200.0] + [200.0 - i for i in range(1, 10)]
    label = None
    for i, c in enumerate(clocks):
        label = sm.step(obs(i, float(i) + 100.0, 0, 0), clock_seconds=c)
        if label:
            break
    assert label == "start"
    b = sm.boundaries[-1]
    # Confirmed at frame index 2 -> ts=102.0; marker = 102 - 30 = 72
    assert abs(b.timestamp - 72.0) < 1e-9, f"expected 72.0, got {b.timestamp}"
    print("PASS: clock lookback arithmetic correct away from the clamp boundary")


def test_score_wins_when_clock_never_locks():
    sm = AdaptiveBoutStateMachine(score_confirm_frames=2)
    # No clock_seconds ever passed (simulates ClockReader never locking).
    # Baseline 0-0, then a confirmed change to 1-0 at frames 3,4.
    seq = [(0, 0), (0, 0), (1, 0), (1, 0)]
    label = None
    for i, (l, r) in enumerate(seq):
        label = sm.step(obs(i, float(i) + 200.0, l, r), clock_seconds=None)
        if label:
            break
    assert label == "start"
    b = sm.boundaries[-1]
    # Confirmed at i=3 -> ts=203; marker = 203 - 60 = 143
    assert abs(b.timestamp - 143.0) < 1e-9, f"expected 143.0, got {b.timestamp}"
    print("PASS: score-confirmed start correctly used when clock signal is absent, 60s lookback")


def test_score_wins_when_confirmed_before_clock():
    sm = AdaptiveBoutStateMachine(score_confirm_frames=2, clock_confirm_frames=5)
    # Clock decreases too, but needs 5 confirms; score confirms after 2 -
    # score should win since it's confirmed earlier in wall-clock terms.
    label = None
    for i in range(6):
        c = 180.0 - i
        sl, sr = (1, 0) if i >= 2 else (0, 0)
        label = sm.step(obs(i, float(i) + 500.0, sl, sr), clock_seconds=c)
        if label:
            break
    assert label == "start"
    b = sm.boundaries[-1]
    # Score confirms at i=3 (second consecutive 1-0) -> ts=503; marker=503-60=443
    assert abs(b.timestamp - 443.0) < 1e-9, f"expected 443.0 (score, -60s), got {b.timestamp}"
    print("PASS: score signal wins and clock is disregarded when score confirms first")


def test_single_frame_score_blip_does_not_fire():
    sm = AdaptiveBoutStateMachine(score_confirm_frames=2)
    # 0-0 baseline, one frame reads 1-0 (a misread), then back to 0-0.
    # Must NOT trigger start - this is the explicit anti-preemptive requirement.
    seq = [(0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0)]
    fired = False
    for i, (l, r) in enumerate(seq):
        label = sm.step(obs(i, float(i), l, r))
        if label:
            fired = True
    assert not fired, "a single-frame score misread must not trigger a start"
    print("PASS: single-frame score misread correctly ignored (anti-preemptive-start)")


def test_single_frame_clock_blip_does_not_fire():
    sm = AdaptiveBoutStateMachine(clock_confirm_frames=2)
    # Steady at 180, one frame dips to 179 (misread), then back to 180.
    clocks = [180.0, 180.0, 179.0, 180.0, 180.0, 180.0]
    fired = False
    for i, c in enumerate(clocks):
        label = sm.step(obs(i, float(i), 0, 0), clock_seconds=c)
        if label:
            fired = True
    assert not fired, "a single-frame clock misread must not trigger a start"
    print("PASS: single-frame clock misread correctly ignored (same discipline applied to clock)")


def test_marker_never_precedes_previous_end():
    sm = AdaptiveBoutStateMachine(score_confirm_frames=2, min_gap_after_previous_end=1.0)
    # Drive an end first (needs a completed bout: start then win-by-15).
    # Simplify: force waiting_for_start=False, prev_scores set, then send a
    # 15 to fire END at t=10.
    sm.waiting_for_start = False
    sm.prev_scores = (14, 3)
    label = sm.step(obs(0, 10.0, 15, 3))
    assert label == "end"
    assert sm._last_end_ts == 10.0

    # Now a score change confirms almost immediately after (t=11, 12) - a
    # naive 60s lookback would compute 12-60 = -48, which must be clamped
    # to no earlier than end_ts + min_gap = 11.0.
    sm.step(obs(1, 11.0, 0, 0))
    label = sm.step(obs(2, 11.5, 1, 0))
    label = sm.step(obs(3, 12.0, 1, 0))
    assert label == "start", f"expected start, got {label}"
    b = sm.boundaries[-1]
    assert b.timestamp >= 11.0, f"marker {b.timestamp} precedes previous end + gap"
    print(f"PASS: marker correctly clamped to {b.timestamp} (not the naive negative value)")


def test_clamp_accounts_for_pad_end():
    """Regression test for a real bug: pipeline.py passes
    min_gap_after_previous_end = cfg.pad_end + start_min_gap_after_previous_end_s,
    so the clamp floor reflects the ACTUAL exported (padded) end, not just
    the raw detected end timestamp. Without folding pad_end in here, a
    padded previous-bout end plus a fast-confirming next-bout start could
    produce overlapping cut files - this reproduces that exact scenario.
    """
    PAD_END = 10.0
    sm = AdaptiveBoutStateMachine(
        clock_confirm_frames=2,
        clock_lookback_s=30.0,
        min_gap_after_previous_end=PAD_END,  # what pipeline.py now passes
    )
    sm.waiting_for_start = False
    sm.prev_scores = (14, 3)
    label = sm.step(obs(0, 100.0, 15, 3))
    assert label == "end"
    padded_export_end = sm._last_end_ts + PAD_END  # what build_bouts() will actually export

    sm.step(obs(1, 101.0, 0, 0), clock_seconds=125.0)
    sm.step(obs(2, 102.0, 0, 0), clock_seconds=124.0)
    label = sm.step(obs(3, 103.0, 0, 0), clock_seconds=123.0)
    assert label == "start", f"expected start, got {label}"
    marker = sm.boundaries[-1].timestamp
    assert marker >= padded_export_end, (
        f"OVERLAP BUG: start marker {marker} precedes the padded previous "
        f"end {padded_export_end} - the two cut clips would overlap"
    )
    print(f"PASS: marker ({marker}) correctly stays at/after the padded "
          f"previous end ({padded_export_end}), no overlap")


if __name__ == "__main__":
    tests = [
        test_end_condition_parity,
        test_clock_wins_with_30s_lookback,
        test_clock_wins_with_30s_lookback_no_clamp,
        test_score_wins_when_clock_never_locks,
        test_score_wins_when_confirmed_before_clock,
        test_single_frame_score_blip_does_not_fire,
        test_single_frame_clock_blip_does_not_fire,
        test_marker_never_precedes_previous_end,
        test_clamp_accounts_for_pad_end,
    ]
    failed = 0
    for t in tests:
        print(f"--- {t.__name__} ---")
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
