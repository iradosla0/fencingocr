"""Adaptive start-marker detection: clock-decrease OR confirmed score-change.

*** This file does NOT modify segment.py. BoutStateMachine there remains
byte-for-byte identical to the original notebook, verified by
tools/verify_state_machine.py. Everything here is new, additional,
explicitly requested behaviour - not a change to the protected logic. ***

Opt in via Config.start_mode = "clock_or_score" (default remains "legacy",
i.e. nothing changes unless you turn this on).

--- The rule ---

While waiting for a bout to start:
  * If the match clock is observed counting DOWN (confirmed - see below,
    not a single-frame read), place the start marker 30 seconds before that
    instant.
  * If a score change is CONFIRMED before the clock-decrease is confirmed
    (or the clock was never locked/readable for this stream), disregard the
    clock signal for this bout and place the start marker 60 seconds before
    the confirmed score-change instant instead.

--- Confirmation, applied to both signals, not just the one it was asked for ---

A single misread digit can produce a false score jump OR a false clock
decrease (e.g. "3:00" misread as "2:00" from one smudged character reads as
a legitimate-looking decrease). The person explicitly required the
score-change signal not to fire on an unconfirmed read; the same discipline
is applied symmetrically to the clock signal for the same reason:
  * A score change only counts once the SAME new (left, right) pair has
    been read on `score_confirm_frames` consecutive qualifying frames.
  * A clock decrease only counts once `clock_confirm_frames` consecutive
    reads are each strictly lower than the value immediately before the
    decreasing streak began.

--- The end condition is untouched ---

Bout end detection - score reaching 15, or resetting to 0-0 after being
non-zero - is segment.py's existing logic, reproduced here rather than
imported so this class has no runtime dependency on internal details of the
protected class, but checked for exact behavioural parity by
tools/verify_adaptive_start.py, which runs identical input sequences
through both BoutStateMachine and AdaptiveBoutStateMachine and requires
identical END-firing behaviour on every sequence. If you ever touch the end
condition in either file, re-run that check.

--- What "sample_index" means on a START boundary here ---

Unlike segment.BoutStateMachine, a START boundary's timestamp here is NOT
the timestamp of the frame that triggered it - it's that frame's timestamp
minus the 30s/60s lookback, which is the actual export marker. sample_index
is kept as the triggering frame's index for traceability/logging, so it will
not satisfy sample_index / sample_fps == timestamp the way it does in the
legacy class. Only .timestamp is used downstream for cutting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .segment import END, START, Boundary, FrameObs


class AdaptiveBoutStateMachine:
    def __init__(
        self,
        waiting_for_start: bool = True,
        score_confirm_frames: int = 2,
        clock_confirm_frames: int = 2,
        score_lookback_s: float = 60.0,
        clock_lookback_s: float = 30.0,
        min_gap_after_previous_end: float = 0.0,
    ):
        self.waiting_for_start = waiting_for_start
        self.prev_scores: Tuple[Optional[int], Optional[int]] = (None, None)
        self.boundaries: List[Boundary] = []

        self.score_confirm_frames = score_confirm_frames
        self.clock_confirm_frames = clock_confirm_frames
        self.score_lookback_s = score_lookback_s
        self.clock_lookback_s = clock_lookback_s
        self.min_gap_after_previous_end = min_gap_after_previous_end

        self._last_end_ts: Optional[float] = None
        self._reset_watch()

    def _reset_watch(self) -> None:
        # Score-change confirmation state.
        self._score_baseline: Tuple[Optional[int], Optional[int]] = (None, None)
        self._score_candidate: Optional[Tuple[int, int]] = None
        self._score_candidate_streak: int = 0
        self._score_confirmed_event: Optional[Tuple[float, int]] = None  # (ts, sample_idx)

        # Clock-decrease confirmation state.
        self._clock_prev_value: Optional[float] = None
        self._clock_streak_len: int = 0
        self._clock_confirmed_event: Optional[Tuple[float, int]] = None

    def step(self, obs: FrameObs, clock_seconds: Optional[float] = None) -> Optional[str]:
        # --- END condition: behaviourally identical to segment.BoutStateMachine ---
        if obs.fencer_count == 2 and obs.has_score_left and obs.has_score_right:
            score_left_int = obs.score_left
            score_right_int = obs.score_right
            prev_scores = self.prev_scores
            waiting_for_start = self.waiting_for_start

            end_fired = bool(
                not waiting_for_start and prev_scores and (
                    (prev_scores[0] not in [None, 0] and prev_scores[1] not in [None, 0]
                     and score_left_int == 0 and score_right_int == 0)
                    or score_left_int == 15 or score_right_int == 15
                )
            )
            self.prev_scores = (score_left_int, score_right_int)

            if end_fired:
                self.waiting_for_start = True
                self._last_end_ts = obs.timestamp
                self._reset_watch()
                b = Boundary(END, obs.sample_index, obs.timestamp)
                self.boundaries.append(b)
                return END

        # --- START condition: new rule, tracked only while waiting ---
        if not self.waiting_for_start:
            return None

        self._track_score(obs)
        self._track_clock(obs, clock_seconds)

        chosen = self._earliest_confirmed_event()
        if chosen is None:
            return None

        kind, (event_ts, event_idx) = chosen
        lookback = self.score_lookback_s if kind == "score" else self.clock_lookback_s
        marker_ts = event_ts - lookback

        floor = 0.0
        if self._last_end_ts is not None:
            floor = max(floor, self._last_end_ts + self.min_gap_after_previous_end)
        marker_ts = max(marker_ts, floor)

        self.waiting_for_start = False
        self._reset_watch()
        b = Boundary(START, event_idx, marker_ts)
        self.boundaries.append(b)
        return START

    def _track_score(self, obs: FrameObs) -> None:
        if not (obs.fencer_count == 2 and obs.has_score_left and obs.has_score_right
                and obs.score_left is not None and obs.score_right is not None):
            return
        cur = (obs.score_left, obs.score_right)
        if self._score_baseline == (None, None):
            self._score_baseline = cur
            return
        if cur == self._score_baseline:
            self._score_candidate = None
            self._score_candidate_streak = 0
            return
        if self._score_candidate == cur:
            self._score_candidate_streak += 1
        else:
            self._score_candidate = cur
            self._score_candidate_streak = 1
        if (self._score_candidate_streak >= self.score_confirm_frames
                and self._score_confirmed_event is None):
            self._score_confirmed_event = (obs.timestamp, obs.sample_index)

    def _track_clock(self, obs: FrameObs, clock_seconds: Optional[float]) -> None:
        if clock_seconds is None:
            return
        if self._clock_prev_value is None:
            self._clock_prev_value = clock_seconds
            return
        if clock_seconds < self._clock_prev_value:
            self._clock_streak_len += 1
            if (self._clock_streak_len >= self.clock_confirm_frames
                    and self._clock_confirmed_event is None):
                self._clock_confirmed_event = (obs.timestamp, obs.sample_index)
        else:
            # Equal or increasing (e.g. reset for the next period) breaks
            # the decreasing streak; a genuine countdown never goes up.
            self._clock_streak_len = 0
        self._clock_prev_value = clock_seconds

    def _earliest_confirmed_event(self) -> Optional[Tuple[str, Tuple[float, int]]]:
        s, c = self._score_confirmed_event, self._clock_confirmed_event
        if s is not None and c is not None:
            return ("score", s) if s[0] <= c[0] else ("clock", c)
        if s is not None:
            return ("score", s)
        if c is not None:
            return ("clock", c)
        return None
