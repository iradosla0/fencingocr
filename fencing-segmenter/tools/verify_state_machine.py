"""Prove the refactored state machine is behaviourally identical to the original.

`reference_process_frame` below is a verbatim transcription of `process_frame`
from cell 4 of streamlined-version1.ipynb, with only the YOLO/OCR calls
replaced by the values they would have returned. Every branch, comparison and
assignment is unchanged, including the `prev_scores` truth test and the
`fencer_count == 2` re-check inside the start condition.

Run:  python tools/verify_state_machine.py
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fenceseg.segment import BoutStateMachine, FrameObs  # noqa: E402


# ---------------------------------------------------------------------------
# verbatim original
# ---------------------------------------------------------------------------

def reference_process_frame(fencer_count, has_left, has_right,
                            score_left_int, score_right_int,
                            frame_number, log, prev_scores, waiting_for_start):
    score_left = object() if has_left else None
    score_right = object() if has_right else None

    if fencer_count != 2 or score_left is None or score_right is None:
        return prev_scores, waiting_for_start

    label = None
    if fencer_count == 2 and waiting_for_start and score_left_int == 0 and score_right_int == 0:
        label = 'start'
        waiting_for_start = False
    elif not waiting_for_start and prev_scores and (
            (prev_scores[0] not in [None, 0] and prev_scores[1] not in [None, 0]
             and score_left_int == 0 and score_right_int == 0)
            or score_left_int == 15 or score_right_int == 15):
        label = 'end'
        waiting_for_start = True

    if label:
        log.append((label, frame_number))

    return (score_left_int, score_right_int), waiting_for_start


def run_reference(seq):
    prev_scores = (None, None)
    waiting_for_start = True
    log = []
    for i, (fc, hl, hr, sl, sr) in enumerate(seq):
        prev_scores, waiting_for_start = reference_process_frame(
            fc, hl, hr, sl, sr, i, log, prev_scores, waiting_for_start
        )
    return log, prev_scores, waiting_for_start


def run_new(seq):
    sm = BoutStateMachine(waiting_for_start=True)
    for i, (fc, hl, hr, sl, sr) in enumerate(seq):
        sm.step(FrameObs(
            sample_index=i, timestamp=float(i),
            fencer_count=fc, has_score_left=hl, has_score_right=hr,
            score_left=sl, score_right=sr,
        ))
    log = [(b.label, b.sample_index) for b in sm.boundaries]
    return log, sm.prev_scores, sm.waiting_for_start


# ---------------------------------------------------------------------------
# fuzzing
# ---------------------------------------------------------------------------

SCORES = [None, 0, 1, 2, 5, 13, 14, 15]
FENCERS = [0, 1, 2, 3]


def random_seq(rng, n):
    out = []
    for _ in range(n):
        out.append((
            rng.choice(FENCERS),
            rng.random() > 0.15,
            rng.random() > 0.15,
            rng.choice(SCORES),
            rng.choice(SCORES),
        ))
    return out


def realistic_seq(rng, n_bouts=6):
    """Sequences shaped like real bouts: 0-0, monotone climb, 15 or reset."""
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
            reps = rng.randint(1, 4)
            for _ in range(reps):
                # occasional dropout / occlusion / OCR failure
                if rng.random() < 0.12:
                    out.append((rng.choice([0, 1, 3]), True, True, l, r))
                elif rng.random() < 0.08:
                    out.append((2, True, True, None, None))
                else:
                    out.append((2, True, True, l, r))
        for _ in range(rng.randint(1, 3)):
            out.append((2, True, True, 0, 0))
    return out


def main():
    rng = random.Random(20260802)
    failures = 0
    checked = 0

    # 1. exhaustive over all short sequences from a reduced alphabet
    small_scores = [None, 0, 3, 15]
    alphabet = [(fc, hl, hr, sl, sr)
                for fc in (1, 2)
                for hl in (True, False)
                for hr in (True,)
                for sl in small_scores
                for sr in small_scores]
    for seq in itertools.product(alphabet, repeat=2):
        checked += 1
        if run_reference(list(seq)) != run_new(list(seq)):
            failures += 1
            print("MISMATCH (exhaustive):", seq)
            if failures > 5:
                break

    # 2. uniform random
    for _ in range(20000):
        seq = random_seq(rng, rng.randint(1, 60))
        checked += 1
        a, b = run_reference(seq), run_new(seq)
        if a != b:
            failures += 1
            print("MISMATCH (random):", seq, a, b)
            if failures > 5:
                break

    # 3. realistic bout-shaped
    for _ in range(2000):
        seq = realistic_seq(rng)
        checked += 1
        a, b = run_reference(seq), run_new(seq)
        if a != b:
            failures += 1
            print("MISMATCH (realistic):", a, b)
            if failures > 5:
                break

    print(f"sequences checked: {checked}")
    print(f"mismatches:        {failures}")
    if failures == 0:
        print("PASS - refactored state machine is identical to the original.")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
