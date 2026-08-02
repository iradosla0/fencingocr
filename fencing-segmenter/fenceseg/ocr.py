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

try:
    import pytesseract
    from pytesseract import Output
    _HAVE_TESS = True
except Exception:  # pragma: no cover
    _HAVE_TESS = False

SCORE_CONFIG = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789"
NAME_CONFIG = (
    "--oem 1 --psm 7 -c "
    "tessedit_char_whitelist="
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .-'"
)

MAX_SCORE = 15


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


def format_signature(box, frame_shape) -> str:
    """Coarse fingerprint of where this overlay sits, used to key template banks.

    Quantised to a 12x12 grid so small jitter in the detector box does not
    fragment the bank, but a genuinely different overlay layout gets its own.
    """
    h, w = frame_shape[:2]
    cx = (box.xmin + box.xmax) / 2.0 / max(w, 1)
    cy = (box.ymin + box.ymax) / 2.0 / max(h, 1)
    bh = (box.ymax - box.ymin) / max(h, 1)
    return f"{box.name}:{int(cx * 12)}:{int(cy * 12)}:{int(bh * 40)}"


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
        self.stats = {"template_hits": 0, "tesseract_calls": 0, "rejected": 0}

    def read(self, frame: np.ndarray, box) -> Optional[int]:
        roi = crop(frame, box)
        binary = binarize(roi, self.target_h)
        glyphs = segment_glyphs(binary)

        # Fencing scores are one or two digits. Anything else is a bad crop.
        if not (1 <= len(glyphs) <= 2):
            glyphs = []

        sig = format_signature(box, frame.shape)

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
    """Reads the fencer name plate from an overlayLeft/overlayRight box."""

    def __init__(self, min_conf: float = 45.0, target_h: int = 48):
        self.min_conf = min_conf
        self.target_h = target_h

    def read(self, frame: np.ndarray, box, exclude=None) -> Optional[str]:
        roi_bgr = crop(frame, box, pad=1)
        if exclude is not None:
            roi_bgr = self._mask_out(roi_bgr, box, exclude)
        binary = binarize(roi_bgr, self.target_h)
        text, conf = _tess(binary, NAME_CONFIG)
        text = re.sub(r"\s+", " ", text).strip(" .-'")
        if len(text) < 2 or conf < self.min_conf:
            return None
        return text

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
