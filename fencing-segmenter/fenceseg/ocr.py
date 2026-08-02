"""OCR for score boxes and name plates.

What the original did, and why each step was hurting:

    gray      = cvtColor(roi, BGR2GRAY)
    denoised  = fastNlMeansDenoising(gray, None, 30, 7, 21)
    _, binar  = threshold(denoised, 0, 255, BINARY + OTSU)
    sharpened = filter2D(binar, -1, [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    text      = image_to_string(sharpened, '--oem 1 --psm 10 -c whitelist=0-9')

1. `--psm 10` means "treat the image as a single character". Fencing scores
   run to 15. Every two-digit score was being fed to a mode that structurally
   cannot return two digits. This is the single largest accuracy bug in the
   notebook, and it bites exactly at the score that triggers the end
   condition. Fixed by using `--psm 7` (single text line).

2. `fastNlMeansDenoising(h=30)` is non-local means. It is one of the most
   expensive filters in OpenCV, it is designed for sensor noise in photographs,
   and a broadcast score overlay is synthetic graphics with no sensor noise.
   It was costing tens of ms per call, twice per frame, to blur away the thin
   strokes it was supposed to preserve. Removed.

3. Otsu produces "text bright on dark" or "text dark on bright" depending on
   the overlay skin. Tesseract expects dark text on a light background. Half
   your overlay formats were being handed inverted images. Fixed by measuring
   foreground polarity and normalising.

4. No upscaling. Score digits are ~15-25px tall at 1080p; Tesseract's LSTM
   engine wants roughly 30-40px of x-height. Fixed by upscaling crops to a
   fixed target height with Lanczos before thresholding.

5. The sharpening kernel ran *after* binarisation, on an image that was already
   pure black and white. It could only add speckle. Removed.

6. No confidence gate: a garbage read was returned as a confident integer.
   A misread of 13 as 15 ends a bout early. Now sub-threshold reads return
   None, which the state machine already handles as "skip this frame".

On top of the fixes, `DigitTemplateBank` learns the broadcast's own digit
glyphs from high-confidence Tesseract reads and thereafter matches by
normalised cross-correlation. After a few hundred frames Tesseract is barely
called: matching a 40x64 template is microseconds, whereas every
`pytesseract` call forks a subprocess (~40-80ms). This is where most of the
OCR speedup comes from, and it also raises accuracy, because the templates
come from the exact font and rendering of the stream being processed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detect import OVERLAY_LEFT
from .naming import COUNTRY_CODES

try:
    import pytesseract
    from pytesseract import Output
    _HAVE_TESS = True
except Exception:  # pragma: no cover
    _HAVE_TESS = False

SCORE_CONFIG = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789"
NAME_CONFIG = (
    '--oem 1 --psm 7 -c '
    '"tessedit_char_whitelist='
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .-'"
    '"'
)
# --psm 8 = single word. Used for the small country badge some broadcasts
# render as a separate, much smaller text element next to the flag - see
# NameReader._read_corner.
CORNER_CONFIG = "--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Match clock: digits, colon, and period (some broadcasts show hundredths,
# e.g. "3:00.00" - see image 2 from the format-variation discussion).
CLOCK_CONFIG = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789:."

_CLOCK_PATTERN = re.compile(r"^(\d{1,2}):(\d{2})(?:[.:](\d{1,2}))?$")


def parse_clock_text(text: str) -> Optional[float]:
    """Parse a clock OCR read like '3:00', '03:00', '3:00.00' into seconds.

    Returns None for anything that doesn't match M:SS[.hh], and also for
    values outside what a fencing period can plausibly show (0 to 4
    minutes) - that range check turns an OCR hallucination into a rejected
    read rather than a false clock signal, the same principle already
    applied to score validation.
    """
    if not text:
        return None
    m = _CLOCK_PATTERN.match(text.strip())
    if not m:
        return None
    minutes = int(m.group(1))
    seconds = int(m.group(2))
    if seconds > 59:
        return None
    total = minutes * 60 + seconds
    frac = m.group(3)
    if frac:
        total += int(frac.ljust(2, "0")[:2]) / 100.0
    if not (0.0 <= total <= 4 * 60):
        return None
    return total

MAX_SCORE = 15


def _validate_tesseract_configs() -> None:
    """Fail loudly at import time if any config string is malformed.

    pytesseract parses config strings with shlex.split() before invoking the
    tesseract binary. An unescaped quote character anywhere in a whitelist
    (e.g. the apostrophe needed for names like O'Brien) raises ValueError
    from inside pytesseract. _tess() below wraps that call in a blanket
    except Exception - correctly, so that a genuinely bad single frame
    doesn't crash a 3-hour run - but that same broad except silently
    swallowed a config string that is malformed on EVERY call, not just a
    bad frame. The result: NAME_CONFIG had exactly this bug, and name
    reading returned empty results 100% of the time, for the entire time
    this module existed, with no error anywhere to point at it.

    Checking every config string against shlex.split() once at import time
    turns that class of bug into an immediate, loud failure instead of a
    multi-hour silent-zero-accuracy run.
    """
    import shlex
    for name, value in [("SCORE_CONFIG", SCORE_CONFIG),
                        ("NAME_CONFIG", NAME_CONFIG),
                        ("CORNER_CONFIG", CORNER_CONFIG),
                        ("CLOCK_CONFIG", CLOCK_CONFIG)]:
        try:
            shlex.split(value)
        except ValueError as e:
            raise ValueError(
                f"fenceseg.ocr.{name} is not a valid shell-style config "
                f"string ({e}). pytesseract parses this with shlex.split() "
                f"before calling tesseract - every OCR call using it will "
                f"silently return empty text. Fix the quoting in {name}."
            ) from e


_validate_tesseract_configs()


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def crop(frame: np.ndarray, box, pad: int = 2) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.as_int()
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), np.uint8)
    return frame[y1:y2, x1:x2]


def _upscale(gray: np.ndarray, target_h: int) -> np.ndarray:
    h = gray.shape[0]
    if h == 0:
        return gray
    if h >= target_h:
        return gray
    scale = target_h / float(h)
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def binarize(roi_bgr: np.ndarray, target_h: int) -> np.ndarray:
    """Return a clean binary image: dark text on a light background."""
    if roi_bgr.size == 0:
        return np.zeros((1, 1), np.uint8)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr
    gray = _upscale(gray, target_h)
    # Mild contrast stretch; overlays are often low-contrast against video.
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Normalise polarity: glyph strokes are the minority of pixels, so the
    # background should dominate. If the image came out mostly black, the
    # text was light-on-dark and needs inverting for Tesseract.
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def segment_glyphs(binary: np.ndarray, min_h_frac: float = 0.45) -> List[np.ndarray]:
    """Split a binary line image into individual glyph crops, left to right."""
    inv = cv2.bitwise_not(binary)  # components = strokes
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h_img = binary.shape[0]
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < min_h_frac * h_img:
            continue
        if area < 8:
            continue
        comps.append((x, y, w, h))
    comps.sort(key=lambda c: c[0])

    # Merge components that overlap heavily in x (e.g. a broken '5').
    merged: List[List[int]] = []
    for x, y, w, h in comps:
        if merged and x < merged[-1][0] + merged[-1][2] * 0.5:
            mx, my, mw, mh = merged[-1]
            nx1, ny1 = min(mx, x), min(my, y)
            nx2, ny2 = max(mx + mw, x + w), max(my + mh, y + h)
            merged[-1] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
        else:
            merged.append([x, y, w, h])

    return [binary[y:y + h, x:x + w] for x, y, w, h in merged]


def _normalize_glyph(g: np.ndarray, size: Tuple[int, int] = (24, 32)) -> np.ndarray:
    if g.size == 0:
        return np.zeros(size[::-1], np.float32)
    r = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    r -= r.mean()
    n = np.linalg.norm(r)
    return r / n if n > 1e-6 else r


# ---------------------------------------------------------------------------
# learned digit templates
# ---------------------------------------------------------------------------

@dataclass
class DigitTemplateBank:
    """Per-broadcast digit glyph bank, learned from confident Tesseract reads.

    Handles the "different overlay formats" problem without retraining: a bank
    is keyed by format signature, so a stream that switches overlay skins
    simply accumulates a second set of templates.
    """
    match_thres: float = 0.72
    max_per_digit: int = 12
    banks: Dict[str, Dict[str, List[np.ndarray]]] = field(default_factory=dict)

    def _bank(self, sig: str) -> Dict[str, List[np.ndarray]]:
        return self.banks.setdefault(sig, {})

    def add(self, sig: str, glyphs: List[np.ndarray], text: str) -> None:
        if len(glyphs) != len(text):
            return
        b = self._bank(sig)
        for g, ch in zip(glyphs, text):
            lst = b.setdefault(ch, [])
            if len(lst) < self.max_per_digit:
                lst.append(_normalize_glyph(g))

    def ready(self, sig: str) -> bool:
        b = self.banks.get(sig, {})
        return len(b) >= 2 and sum(len(v) for v in b.values()) >= 6

    def match(self, sig: str, glyphs: List[np.ndarray]) -> Tuple[Optional[str], float]:
        b = self.banks.get(sig)
        if not b or not glyphs:
            return None, 0.0
        chars, scores = [], []
        for g in glyphs:
            ng = _normalize_glyph(g)
            best_ch, best_s = None, -1.0
            for ch, tmpls in b.items():
                for t in tmpls:
                    s = float(np.dot(ng.ravel(), t.ravel()))
                    if s > best_s:
                        best_s, best_ch = s, ch
            if best_ch is None or best_s < self.match_thres:
                return None, best_s if best_s > 0 else 0.0
            chars.append(best_ch)
            scores.append(best_s)
        return "".join(chars), float(np.mean(scores))


class PositionClusterer:
    """Groups detector boxes into stable position clusters.

    Replaces an earlier version that quantised (x, y, height) onto a fixed
    grid. That was fragile in exactly the dimension that mattered: bucketing
    box height into 40 slices of the FULL FRAME height gives ~27px buckets
    on a 1080p frame, but a score box is only ~15-30px tall to begin with.
    Ordinary YOLO box jitter of a few pixels between frames - expected on a
    detector trained on a small dataset - was enough to flip buckets, which
    fragmented one stable broadcast overlay into dozens of tiny signatures,
    none of which ever accumulated the samples needed for DigitTemplateBank
    to become ready. The template path silently never engaged and every
    frame paid full Tesseract cost.

    This clusters by proximity instead: a box joins the nearest existing
    cluster if it is within `tolerance` (as a fraction of frame diagonal),
    otherwise it starts a new one. A cluster's centroid drifts slowly toward
    new members (exponential moving average) so gradual detector jitter
    never accumulates into a boundary crossing the way a fixed grid line
    could. Genuinely different positions - a different broadcast, or a
    mid-stream graphics change - still get their own cluster; they are
    outside the tolerance radius of anything seen so far.

    Also tracks width, not just center+height. Score boxes legitimately
    resize between a one-digit and two-digit score, so ScoreReader only ever
    used center+height for signature-keying and still crops from each
    frame's freshly detected box. The overlay/name-plate box is different:
    it's a fixed-size graphic for the whole broadcast, so tracking width too
    lets a caller reconstitute a full (x1,y1,x2,y2) box from the cluster
    alone and skip needing a fresh per-frame detection entirely once locked.
    """

    def __init__(self, tolerance: float = 0.04, drift: float = 0.1):
        self.tolerance = tolerance
        self.drift = drift
        self._centroids: Dict[str, Tuple[float, float, float, float]] = {}  # key -> (cx, cy, bw, bh)
        self._hits: Dict[str, int] = {}
        self._next_id: Dict[str, int] = {}

    def key_for(self, box, frame_shape) -> Tuple[str, bool]:
        """Returns (signature_key, is_new_cluster)."""
        h, w = frame_shape[:2]
        cx = (box.xmin + box.xmax) / 2.0 / max(w, 1)
        cy = (box.ymin + box.ymax) / 2.0 / max(h, 1)
        bw = (box.xmax - box.xmin) / max(w, 1)
        bh = (box.ymax - box.ymin) / max(h, 1)

        best_key, best_d = None, None
        prefix = box.name + ":"
        for key, (kcx, kcy, kbw, kbh) in self._centroids.items():
            if not key.startswith(prefix):
                continue
            d = ((cx - kcx) ** 2 + (cy - kcy) ** 2 + (bh - kbh) ** 2) ** 0.5
            if best_d is None or d < best_d:
                best_d, best_key = d, key

        if best_key is not None and best_d <= self.tolerance:
            okcx, okcy, okbw, okbh = self._centroids[best_key]
            a = self.drift
            self._centroids[best_key] = (
                (1 - a) * okcx + a * cx, (1 - a) * okcy + a * cy,
                (1 - a) * okbw + a * bw, (1 - a) * okbh + a * bh,
            )
            self._hits[best_key] = self._hits.get(best_key, 0) + 1
            return best_key, False

        n = self._next_id.get(box.name, 0)
        self._next_id[box.name] = n + 1
        new_key = f"{box.name}:{n}"
        self._centroids[new_key] = (cx, cy, bw, bh)
        self._hits[new_key] = 1
        return new_key, True

    def dominant_key(self, class_name: str) -> Optional[str]:
        """The cluster with the most hits for a given class, or None."""
        prefix = class_name + ":"
        candidates = {k: v for k, v in self._hits.items() if k.startswith(prefix)}
        if not candidates:
            return None
        return max(candidates, key=candidates.get)

    def locked_box(self, class_name: str, frame_shape,
                  min_hits: int = 20) -> Optional[Tuple[float, float, float, float]]:
        """Absolute-pixel (x1,y1,x2,y2) for the dominant cluster of a class,
        or None if no cluster of that class has reached min_hits yet.

        min_hits defaults high (20) relative to the position-jitter tolerance
        used for score signature-keying, deliberately: this is used to skip
        real detector work later, so it should only fire once a position is
        genuinely well-established, not on the first couple of lucky matches.
        """
        key = self.dominant_key(class_name)
        if key is None or self._hits.get(key, 0) < min_hits:
            return None
        cx, cy, bw, bh = self._centroids[key]
        h, w = frame_shape[:2]
        return (
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h,
        )

    @property
    def n_clusters(self) -> int:
        return len(self._centroids)


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def _tess(img: np.ndarray, config: str) -> Tuple[str, float]:
    """Run Tesseract, returning (text, mean per-word confidence 0-100)."""
    if not _HAVE_TESS:
        return "", 0.0
    try:
        data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    except Exception:
        return "", 0.0
    words, confs = [], []
    for txt, c in zip(data.get("text", []), data.get("conf", [])):
        txt = (txt or "").strip()
        try:
            c = float(c)
        except (TypeError, ValueError):
            continue
        if txt and c >= 0:
            words.append(txt)
            confs.append(c)
    if not words:
        return "", 0.0
    return " ".join(words), float(np.mean(confs))


class ScoreReader:
    """Reads an integer score from a scoreLeft/scoreRight box.

    Returns None whenever the read is not trustworthy. None is the same signal
    the original produced on OCR failure, and the state machine treats it
    identically: the frame contributes nothing and prev_scores is preserved.
    """

    def __init__(self, min_conf: float = 55.0, target_h: int = 64,
                 use_templates: bool = True, match_thres: float = 0.72):
        self.min_conf = min_conf
        self.target_h = target_h
        self.use_templates = use_templates
        self.bank = DigitTemplateBank(match_thres=match_thres)
        self.clusterer = PositionClusterer()
        self.stats = {"template_hits": 0, "tesseract_calls": 0, "rejected": 0,
                      "clusters_created": 0}

    def read(self, frame: np.ndarray, box) -> Optional[int]:
        roi = crop(frame, box)
        binary = binarize(roi, self.target_h)
        glyphs = segment_glyphs(binary)

        # Fencing scores are one or two digits. Anything else is a bad crop.
        if not (1 <= len(glyphs) <= 2):
            glyphs = []

        sig, is_new = self.clusterer.key_for(box, frame.shape)
        if is_new:
            self.stats["clusters_created"] += 1

        # Fast path: learned templates.
        if self.use_templates and glyphs and self.bank.ready(sig):
            text, score = self.bank.match(sig, glyphs)
            if text is not None:
                val = self._validate(text)
                if val is not None:
                    self.stats["template_hits"] += 1
                    return val

        # Slow path: Tesseract.
        self.stats["tesseract_calls"] += 1
        text, conf = _tess(binary, SCORE_CONFIG)
        text = re.sub(r"\D", "", text)
        val = self._validate(text)
        if val is None or conf < self.min_conf:
            self.stats["rejected"] += 1
            return None

        # Teach the bank from this confident read.
        if self.use_templates and glyphs and len(glyphs) == len(text):
            self.bank.add(sig, glyphs, text)
        return val

    @staticmethod
    def _validate(text: str) -> Optional[int]:
        if not text or not text.isdigit():
            return None
        try:
            v = int(text)
        except ValueError:
            return None
        # A fencing score cannot exceed 15. Range-checking here converts an
        # OCR hallucination into a skipped frame instead of a false boundary.
        if 0 <= v <= MAX_SCORE:
            return v
        return None


class NameReader:
    """Reads the fencer name plate from an overlayLeft/overlayRight box.

    Some broadcasts render the country code as a small badge tucked next to
    the flag - visually much smaller than, and spatially separate from, the
    name text (e.g. a tiny "CHN" sitting above the flag icon while "HUANG
    Qianqian" runs at 2-3x that text height alongside it). The main read
    below uses one uniform upscale target tuned for name-sized text; a badge
    that much smaller relative to the rest of the crop will often fall below
    Tesseract's readable threshold in that single pass even though it is
    technically inside the crop.

    Where that badge sits is NOT assumed to be the same across different
    broadcasts. One observed layout puts it in the top corner nearest the
    flag; nothing guarantees every future stream agrees, and a livestream
    that puts it somewhere else should not be silently unreadable just
    because an earlier stream's layout got hardcoded here. What IS true
    (confirmed against real footage, not assumed) is that the layout is
    consistent for the length of one broadcast - a single livestream doesn't
    switch overlay skins mid-stream. So instead of one fixed geometry, this
    tries a small set of candidate regions during the first several reads on
    each side of THIS video, locks onto whichever one actually produces a
    valid code, and stops searching (saving Tesseract calls) if none do -
    the same per-video-adaptive shape already used by PositionClusterer for
    box jitter and DigitTemplateBank for digit glyphs.

    A stream with no corner badge at all (country embedded in the main text,
    or no country text anywhere) correctly settles into "give up searching"
    rather than paying the cost of a corner scan on every remaining frame.
    """

    # Candidate regions to try, as (y0, y1) fractions of the plate box height,
    # searched at the horizontal edge nearest that side's flag. Extend this
    # if a future stream's badge position doesn't match either candidate -
    # the search/lock mechanism below needs no other change to support it.
    CORNER_CANDIDATES = {
        "top": (0.0, 0.42),
        "bottom": (0.58, 1.0),
    }

    def __init__(self, min_conf: float = 45.0, target_h: int = 48,
                 corner_min_conf: float = 35.0, corner_search_attempts: int = 6,
                 corner_lock_threshold: int = 2):
        self.min_conf = min_conf
        self.target_h = target_h
        self.corner_min_conf = corner_min_conf
        self.corner_search_attempts = corner_search_attempts
        self.corner_lock_threshold = corner_lock_threshold
        # Per-video, per-side search state - never shared across videos,
        # same rule as every other piece of mutable OCR state in this file.
        self._corner_state = {
            side: {"mode": "searching", "attempts": 0,
                  "hits": {c: 0 for c in self.CORNER_CANDIDATES}}
            for side in ("left", "right")
        }
        self.stats = {"corner_reads": 0, "corner_hits": 0,
                      "corner_mode_left": "searching", "corner_mode_right": "searching"}

    def read(self, frame: np.ndarray, box, exclude=None) -> Optional[str]:
        roi_bgr = crop(frame, box, pad=1)
        if exclude is not None:
            roi_bgr = self._mask_out(roi_bgr, box, exclude)
        binary = binarize(roi_bgr, self.target_h)
        text, conf = _tess(binary, NAME_CONFIG)
        text = re.sub(r"\s+", " ", text).strip(" .-'")
        if len(text) < 2 or conf < self.min_conf:
            text = None

        if text is None or not self._has_country_token(text):
            side = "left" if box.name == OVERLAY_LEFT else "right"
            corner = self._read_corner_adaptive(roi_bgr, side)
            if corner:
                text = f"{text} {corner}" if text else corner

        if not text or len(text) < 2:
            return None
        return text

    @staticmethod
    def _has_country_token(text: str) -> bool:
        tokens = text.split()
        if not tokens:
            return False
        for i in (0, -1):
            t = re.sub(r"[^A-Za-z]", "", tokens[i]).upper()
            if len(t) == 3 and t in COUNTRY_CODES:
                return True
        return False

    def _read_corner_adaptive(self, roi_bgr: np.ndarray, side: str) -> Optional[str]:
        state = self._corner_state[side]

        if state["mode"] == "none":
            return None  # gave up: no corner badge found on this side, don't keep paying for it
        if state["mode"] == "locked":
            return self._read_corner_region(roi_bgr, side, state["candidate"])

        # Still searching: cycle through candidates round-robin.
        state["attempts"] += 1
        self.stats["corner_reads"] += 1
        names = list(self.CORNER_CANDIDATES)
        candidate = names[(state["attempts"] - 1) % len(names)]
        result = self._read_corner_region(roi_bgr, side, candidate)

        if result:
            self.stats["corner_hits"] += 1
            state["hits"][candidate] += 1
            if state["hits"][candidate] >= self.corner_lock_threshold:
                state["mode"] = "locked"
                state["candidate"] = candidate
                self.stats[f"corner_mode_{side}"] = f"locked:{candidate}"

        if state["mode"] == "searching" and state["attempts"] >= self.corner_search_attempts:
            best = max(state["hits"], key=lambda c: state["hits"][c])
            if state["hits"][best] > 0:
                state["mode"] = "locked"
                state["candidate"] = best
                self.stats[f"corner_mode_{side}"] = f"locked:{best}"
            else:
                state["mode"] = "none"
                self.stats[f"corner_mode_{side}"] = "none"

        return result

    def _read_corner_region(self, roi_bgr: np.ndarray, side: str,
                            candidate: str) -> Optional[str]:
        """OCR one candidate corner region, tight-cropped and upscaled.

        The badge glyphs are small even within this already-small corner
        crop. Upscaling the whole crop (mostly empty background) to a modest
        target height barely enlarges the letters themselves. Instead: rough
        threshold first, find the actual foreground pixel bounding box, crop
        tight to that, then upscale aggressively - the same lesson that
        applies to the score digits, just at a smaller scale here.
        """
        h, w = roi_bgr.shape[:2]
        y0f, y1f = self.CORNER_CANDIDATES[candidate]
        y0, y1 = int(h * y0f), int(h * y1f)
        cw = max(1, int(w * 0.28))
        patch = roi_bgr[y0:y1, 0:cw] if side == "left" else roi_bgr[y0:y1, w - cw:w]
        if patch.size == 0:
            return None

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        _, rough = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if rough.mean() < 127:
            rough = cv2.bitwise_not(rough)
        ys, xs = np.where(rough < 128)
        if len(xs) < 4:
            return None
        x1, x2 = max(0, xs.min() - 1), min(rough.shape[1], xs.max() + 2)
        y1b, y2b = max(0, ys.min() - 1), min(rough.shape[0], ys.max() + 2)

        # A real 3-letter badge is compact. If the foreground bounding box
        # spans most of the candidate patch's width, this is very likely
        # bleed-through from the adjacent name text rather than an actual
        # badge - reject rather than guess.
        if (x2 - x1) > 0.65 * patch.shape[1]:
            return None

        tight = patch[y1b:y2b, x1:x2]
        if tight.size == 0:
            return None

        binary = binarize(tight, target_h=64)
        text, conf = _tess(binary, CORNER_CONFIG)
        text = re.sub(r"[^A-Za-z]", "", text).upper()
        if len(text) == 3 and conf >= self.corner_min_conf:
            return text
        return None

    @staticmethod
    def _mask_out(roi_bgr: np.ndarray, box, exclude) -> np.ndarray:
        """Blank the score box out of the name plate crop if they overlap."""
        out = roi_bgr.copy()
        ox1 = int(exclude.xmin - box.xmin) + 1
        oy1 = int(exclude.ymin - box.ymin) + 1
        ox2 = int(exclude.xmax - box.xmin) + 1
        oy2 = int(exclude.ymax - box.ymin) + 1
        h, w = out.shape[:2]
        ox1, oy1 = max(0, ox1), max(0, oy1)
        ox2, oy2 = min(w, ox2), min(h, oy2)
        if ox2 > ox1 and oy2 > oy1:
            # Fill with the plate's own background colour rather than black,
            # so Otsu is not skewed by an artificial dark block.
            bg = np.median(out.reshape(-1, out.shape[-1]), axis=0)
            out[oy1:oy2, ox1:ox2] = bg
        return out


class TemporalScoreFilter:
    """Optional single-frame outlier rejection. OFF by default.

    Rationale: at a 1 Hz decision rate a real fencing score moves by one touch
    at a time, with seconds of halted action either side. An OCR read that
    jumps implausibly far in one second is almost always a misread (3 read as
    13, 1 read as 4). Rejecting it returns None, which is the "skip this frame"
    path the state machine already had.

    Deliberately narrow, because the end condition depends on two events this
    filter must never suppress:
      * the reset to 0-0 at the end of a bout, which is a *decrease*
      * a score reaching 15
    Both are whitelisted unconditionally. The allowed jump also widens with the
    time since the last accepted read, so a long occlusion does not cause the
    filter to reject the (correct, much higher) score that follows.

    This is off by default because it changes which frames yield a value, and
    therefore can shift a boundary by a sampling interval. Turn it on with
    --temporal-vote once you have compared the two on a stream you know.
    """

    def __init__(self, base_jump: int = 3, seconds_per_extra: float = 5.0):
        self.base_jump = base_jump
        self.seconds_per_extra = seconds_per_extra
        self._last: Dict[str, Tuple[int, float]] = {}
        self.rejected = 0

    def filter(self, side: str, value: Optional[int], ts: float) -> Optional[int]:
        if value is None:
            return None
        # Never suppress a reset or a bout-winning score.
        if value == 0 or value == MAX_SCORE:
            self._last[side] = (value, ts)
            return value

        prev = self._last.get(side)
        if prev is None:
            self._last[side] = (value, ts)
            return value

        prev_val, prev_ts = prev
        elapsed = max(0.0, ts - prev_ts)
        allowed = self.base_jump + int(elapsed / self.seconds_per_extra)

        if value < prev_val or (value - prev_val) > allowed:
            self.rejected += 1
            return None

        self._last[side] = (value, ts)
        return value

    def reset(self) -> None:
        self._last.clear()


class ClockReader:
    """Locates and reads the match clock, without a trained detector class.

    The current model has no 'timer' class - fencer, overlayLeft,
    overlayRight, scoreLeft, scoreRight only. Adding one properly is the
    robust long-term fix (see training/README.md) but needs new labeled
    data. Absent that, this exploits a positional prior that held across
    every example overlay format seen so far: the clock sits horizontally
    BETWEEN scoreLeft and scoreRight, at some vertical offset relative to
    the score row that varies by broadcast (directly inline with the scores
    in some formats, in a small graphic above the row in others).

    Candidate vertical bands are searched during the first several frames
    where both score boxes are available, the same search-then-lock pattern
    as NameReader's corner-badge search: whichever candidate's OCR output
    repeatedly parses as a plausible clock value (see parse_clock_text)
    gets locked in and used for the rest of the video. If none do, this
    gives up for the rest of the video - the clock-based start signal
    simply never fires, and the confirmed-score-change signal still works
    independently.

    Because this has no trained class to fall back on, it is inherently
    less certain than the digit/name reading elsewhere in this file, which
    all locate their target via the trained detector first. Treat a locked
    clock reading as good-but-unverified until checked against real
    footage from a specific broadcast.
    """

    # Candidate vertical bands, expressed as (y0, y1) offsets from the score
    # row's vertical center, in units of the score box height. "inline"
    # covers a clock drawn at the same height as the score digits; "above"
    # and "below" cover a clock drawn in a separate graphic element offset
    # from the score row (e.g. image 2's diamond sitting above the scores).
    CANDIDATE_BANDS = {
        "inline": (-0.6, 0.6),
        "above": (-1.8, -0.4),
        "below": (0.4, 1.8),
    }

    def __init__(self, min_conf: float = 30.0, search_attempts: int = 10,
                lock_threshold: int = 3):
        self.min_conf = min_conf
        self.search_attempts = search_attempts
        self.lock_threshold = lock_threshold
        self._state = {"mode": "searching", "attempts": 0,
                       "hits": {c: 0 for c in self.CANDIDATE_BANDS}}
        self.stats = {"reads": 0, "hits": 0, "mode": "searching"}

    def read(self, frame: np.ndarray, score_left_box, score_right_box) -> Optional[float]:
        """Returns seconds remaining, or None if unreadable/not yet locked."""
        if score_left_box is None or score_right_box is None:
            return None
        x1, x2 = score_left_box.xmax, score_right_box.xmin
        if x2 <= x1:
            return None  # boxes overlap or are out of expected left/right order

        sb_h = ((score_left_box.ymax - score_left_box.ymin)
               + (score_right_box.ymax - score_right_box.ymin)) / 2.0
        sb_cy = ((score_left_box.ymin + score_left_box.ymax) / 2.0
                + (score_right_box.ymin + score_right_box.ymax) / 2.0) / 2.0
        if sb_h <= 0:
            return None

        state = self._state
        if state["mode"] == "none":
            return None
        if state["mode"] == "locked":
            return self._read_band(frame, x1, x2, sb_cy, sb_h, state["band"])

        state["attempts"] += 1
        self.stats["reads"] += 1
        names = list(self.CANDIDATE_BANDS)
        band = names[(state["attempts"] - 1) % len(names)]
        result = self._read_band(frame, x1, x2, sb_cy, sb_h, band)

        if result is not None:
            self.stats["hits"] += 1
            state["hits"][band] += 1
            if state["hits"][band] >= self.lock_threshold:
                state["mode"] = "locked"
                state["band"] = band
                self.stats["mode"] = f"locked:{band}"

        if state["mode"] == "searching" and state["attempts"] >= self.search_attempts:
            best = max(state["hits"], key=lambda c: state["hits"][c])
            if state["hits"][best] > 0:
                state["mode"] = "locked"
                state["band"] = best
                self.stats["mode"] = f"locked:{best}"
            else:
                state["mode"] = "none"
                self.stats["mode"] = "none"

        return result

    def _read_band(self, frame: np.ndarray, x1: float, x2: float,
                   sb_cy: float, sb_h: float, band_name: str) -> Optional[float]:
        y0f, y1f = self.CANDIDATE_BANDS[band_name]
        y0 = int(sb_cy + y0f * sb_h)
        y1 = int(sb_cy + y1f * sb_h)
        y0, y1 = max(0, min(y0, y1)), min(frame.shape[0], max(y0, y1))
        x1i, x2i = max(0, int(x1)), min(frame.shape[1], int(x2))
        if y1 <= y0 or x2i <= x1i:
            return None

        patch = frame[y0:y1, x1i:x2i]
        if patch.size == 0:
            return None

        binary = binarize(patch, target_h=64)
        text, conf = _tess(binary, CLOCK_CONFIG)
        if conf < self.min_conf:
            return None
        return parse_clock_text(text)
