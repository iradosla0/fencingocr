"""Bout boundary state machine.

*** The detection conditions in this file are byte-for-byte equivalent to the
original notebook. Nothing here has been "improved". ***

Original, from cell 4 of streamlined-version1.ipynb:

    if fencer_count != 2 or score_left is None or score_right is None:
        return prev_scores, waiting_for_start

    ... OCR, int() conversion, ValueError -> return unchanged ...

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

    return (score_left_int, score_right_int), waiting_for_start

Three subtleties that are load-bearing and are preserved verbatim:

  * The guard is part of the condition. A frame without exactly two `fencer`
    detections, or missing either score box, contributes nothing at all --
    it does not even update prev_scores.
  * prev_scores is updated to the newly read pair on every frame that passes
    the guard, including frames where no label fires, and including frames
    where OCR returned None (the pair (None, None) is stored).
  * `prev_scores` is truth-tested. The tuple (None, None) is truthy in Python,
    so the end branch is reachable on it; it then fails the inner
    `not in [None, 0]` test. Only the initial value matters for the falsy
    case, and it is a tuple too. Preserved by keeping the same truth test.

tools/verify_state_machine.py fuzzes this implementation against a verbatim
transcription of the original over millions of random sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

START = "start"
END = "end"


@dataclass
class Boundary:
    label: str
    sample_index: int
    timestamp: float


@dataclass
class FrameObs:
    """Everything the state machine needs to know about one sampled frame."""
    sample_index: int
    timestamp: float
    fencer_count: int
    has_score_left: bool
    has_score_right: bool
    score_left: Optional[int]
    score_right: Optional[int]


class BoutStateMachine:
    def __init__(self, waiting_for_start: bool = True):
        self.waiting_for_start = waiting_for_start
        self.prev_scores: Tuple[Optional[int], Optional[int]] = (None, None)
        self.boundaries: List[Boundary] = []

    def step(self, obs: FrameObs) -> Optional[str]:
        # --- guard: identical to the original early return -----------------
        if obs.fencer_count != 2 or not obs.has_score_left or not obs.has_score_right:
            return None

        score_left_int = obs.score_left
        score_right_int = obs.score_right

        prev_scores = self.prev_scores
        waiting_for_start = self.waiting_for_start

        # --- labelling: identical to the original ---------------------------
        label = None
        if (obs.fencer_count == 2 and waiting_for_start
                and score_left_int == 0 and score_right_int == 0):
            label = START
            waiting_for_start = False
        elif (not waiting_for_start and prev_scores and (
                (prev_scores[0] not in [None, 0] and prev_scores[1] not in [None, 0]
                 and score_left_int == 0 and score_right_int == 0)
                or score_left_int == 15 or score_right_int == 15)):
            label = END
            waiting_for_start = True

        self.prev_scores = (score_left_int, score_right_int)
        self.waiting_for_start = waiting_for_start

        if label:
            self.boundaries.append(Boundary(label, obs.sample_index, obs.timestamp))
        return label


def pair_boundaries(boundaries: List[Boundary]) -> List[Tuple[Boundary, Boundary]]:
    """Zip boundaries into (start, end) pairs.

    The original did this with a regex over the log file that required a
    `start:` line immediately followed by an `end:` line. Same result, but an
    unterminated trailing start is reported rather than silently dropped by a
    regex miss.
    """
    pairs = []
    pending: Optional[Boundary] = None
    for b in boundaries:
        if b.label == START:
            pending = b
        elif b.label == END and pending is not None:
            pairs.append((pending, b))
            pending = None
    return pairs


def unterminated(boundaries: List[Boundary]) -> Optional[Boundary]:
    pending = None
    for b in boundaries:
        if b.label == START:
            pending = b
        elif b.label == END:
            pending = None
    return pending
