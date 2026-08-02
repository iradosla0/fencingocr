"""Turn name-plate OCR into "First Last NAT vs. First Last NAT" filenames.

Robustness comes from voting, not from any single frame. A name plate is
on screen for the whole bout, so the pipeline OCRs it on every sampled frame
between the start and end boundary and takes the modal reading. A one-frame
misread is outvoted by the ~200 frames of a typical bout.

Parsing is heuristic because broadcast conventions vary:
    "GEORGIADIS Andreas GRE"      surname-first, caps surname
    "Andreas GEORGIADIS GRE"      given-first
    "A. GEORGIADIS GRE"           initial only
    "GEORGIADIS GRE"              no given name

Per your instruction, whatever is legible goes into the filename; the parser
never blocks on a component it cannot classify.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional

# IOC / FIE three-letter codes seen in international fencing. Not exhaustive:
# an unrecognised 3-letter all-caps token is still accepted as a country when
# it sits in a plausible position.
COUNTRY_CODES = {
    "ALG", "ARG", "ARM", "AUS", "AUT", "AZE", "BEL", "BLR", "BRA", "BUL",
    "CAN", "CHI", "CHN", "COL", "CRO", "CUB", "CYP", "CZE", "DEN", "DOM",
    "ECU", "EGY", "ESP", "EST", "FIN", "FRA", "GBR", "GEO", "GER", "GRE",
    "GUA", "HKG", "HUN", "INA", "IND", "IRI", "IRL", "ISR", "ITA", "JPN",
    "KAZ", "KOR", "KSA", "KUW", "LAT", "LTU", "LUX", "MAR", "MAS", "MEX",
    "MDA", "MGL", "MON", "NED", "NGR", "NOR", "NZL", "PAN", "PER", "PHI",
    "POL", "POR", "PUR", "QAT", "ROU", "RSA", "RUS", "SGP", "SLO", "SRB",
    "SUI", "SVK", "SWE", "THA", "TPE", "TUN", "TUR", "UKR", "USA", "UZB",
    "VEN", "VIE", "AIN", "FIE", "ROC",
}

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WS = re.compile(r"\s+")


@dataclass
class Fencer:
    first: str = ""
    last: str = ""
    country: str = ""
    raw: str = ""

    def display(self, space: str = " ") -> str:
        parts = [p for p in (self.first, self.last, self.country) if p]
        if not parts:
            parts = [self.raw] if self.raw else ["Unknown"]
        return space.join(_WS.sub(" ", " ".join(parts)).split())


def _titlecase(tok: str) -> str:
    # Capitalise each run of letters. Handles O'Brien, Jean-Luc, de la Cruz
    # and bare initials like "A." without mangling the punctuation.
    return re.sub(r"[A-Za-z]+", lambda m: m.group(0).capitalize(), tok.lower())


def parse_plate(text: str) -> Fencer:
    """Best-effort split of one plate reading into first / last / country."""
    raw = _WS.sub(" ", (text or "").strip())
    if not raw:
        return Fencer()

    tokens = [t for t in raw.split(" ") if t]
    country = ""

    # Country code: prefer a known code; otherwise a bare 3-letter all-caps
    # token at either end of the string.
    for i in (len(tokens) - 1, 0):
        if 0 <= i < len(tokens):
            t = re.sub(r"[^A-Za-z]", "", tokens[i]).upper()
            if len(t) == 3 and t in COUNTRY_CODES:
                country = t
                tokens.pop(i)
                break
    if not country and tokens:
        for i in (len(tokens) - 1, 0):
            if 0 <= i < len(tokens):
                t = tokens[i]
                if len(t) == 3 and t.isalpha() and t.isupper():
                    country = t.upper()
                    tokens.pop(i)
                    break

    if not tokens:
        return Fencer(country=country, raw=raw)

    # Surname convention: broadcasts put the surname in full caps.
    caps = [t for t in tokens if t.isalpha() and t.isupper() and len(t) > 1]
    rest = [t for t in tokens if t not in caps]

    if caps and rest:
        last = _titlecase(" ".join(caps))
        first = _titlecase(" ".join(rest))
    elif caps and not rest:
        # Everything is caps: fall back to word order. FIE plates are
        # surname-first, so the first token is the surname.
        last = _titlecase(caps[0])
        first = _titlecase(" ".join(caps[1:]))
    else:
        # No case signal at all.
        if len(tokens) == 1:
            last, first = _titlecase(tokens[0]), ""
        else:
            last = _titlecase(tokens[0])
            first = _titlecase(" ".join(tokens[1:]))

    return Fencer(first=first.strip(), last=last.strip(), country=country, raw=raw)


def vote(readings: Iterable[Optional[str]], min_votes: int = 2) -> Optional[str]:
    """Modal plate reading across a bout, ignoring Nones."""
    vals = [r for r in readings if r]
    if not vals:
        return None
    counts = Counter(vals)
    text, n = counts.most_common(1)[0]
    if n < min_votes and len(vals) >= min_votes:
        return None
    return text


def sanitize(name: str, max_len: int = 180) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = _ILLEGAL.sub("", name)
    name = _WS.sub(" ", name).strip().strip(".")
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "bout"


def bout_filename(left: Optional[str], right: Optional[str],
                  index: int, space: str = " ") -> str:
    """Build "First Last NAT vs. First Last NAT", degrading gracefully."""
    if not left and not right:
        return f"bout_{index:03d}"
    lf = parse_plate(left or "").display(space)
    rf = parse_plate(right or "").display(space)
    if not left:
        lf = "Unknown"
    if not right:
        rf = "Unknown"
    return sanitize(f"{lf} vs. {rf}")


def dedupe(name: str, taken: set) -> str:
    if name not in taken:
        taken.add(name)
        return name
    i = 2
    while f"{name} ({i})" in taken:
        i += 1
    out = f"{name} ({i})"
    taken.add(out)
    return out
