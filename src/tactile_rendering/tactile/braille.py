"""Stage 3: Braille transcription.

Converts label strings to UEB Grade 1 (uncontracted) braille.

v1 uses a built-in ASCII-to-UEB lookup table. When system liblouis
becomes available, the `transcribe` function should delegate to
`louis.translateString(['en-ueb-g2.ctb'], text)` for Grade 2
(contracted) output, and to Nemeth tables for math expressions.
"""

from __future__ import annotations

from dataclasses import dataclass

# UEB Grade 1: one-to-one mapping from ASCII to Unicode braille.
# Unicode braille block: U+2800 to U+28FF.
# Dot positions: 1=0x01, 2=0x02, 3=0x04, 4=0x08, 5=0x10, 6=0x20, 7=0x40, 8=0x80
# UEB letters use dots 1-6 only; capital indicator is dot 6 (⠠).

_CAPITAL_INDICATOR = "\u2820"  # dot 6

_UEB_ALPHA = {
    "a": "\u2801",  # dot 1
    "b": "\u2803",  # dots 1-2
    "c": "\u2809",  # dots 1-4
    "d": "\u2819",  # dots 1-4-5
    "e": "\u2811",  # dots 1-5
    "f": "\u280B",  # dots 1-2-4
    "g": "\u281B",  # dots 1-2-4-5
    "h": "\u2813",  # dots 1-2-5
    "i": "\u280A",  # dots 2-4
    "j": "\u281A",  # dots 2-4-5
    "k": "\u2805",  # dots 1-3
    "l": "\u2807",  # dots 1-2-3
    "m": "\u280D",  # dots 1-3-4
    "n": "\u281D",  # dots 1-3-4-5
    "o": "\u2815",  # dots 1-3-5
    "p": "\u280F",  # dots 1-2-3-4
    "q": "\u281F",  # dots 1-2-3-4-5
    "r": "\u2817",  # dots 1-2-3-5
    "s": "\u280E",  # dots 2-3-4
    "t": "\u281E",  # dots 2-3-4-5
    "u": "\u2825",  # dots 1-3-6
    "v": "\u2827",  # dots 1-2-3-6
    "w": "\u283A",  # dots 2-4-5-6
    "x": "\u282D",  # dots 1-3-4-6
    "y": "\u283D",  # dots 1-3-4-5-6
    "z": "\u2835",  # dots 1-3-5-6
}

_UEB_DIGITS = {
    # UEB uses the number indicator (⠼, dots 3-4-5-6) followed by
    # the letter-value of the digit (1=a, 2=b, ... 0=j).
    "1": "\u2801",  # a-value
    "2": "\u2803",  # b-value
    "3": "\u2809",  # c-value
    "4": "\u2819",  # d-value
    "5": "\u2811",  # e-value
    "6": "\u280B",  # f-value
    "7": "\u281B",  # g-value
    "8": "\u2813",  # h-value
    "9": "\u280A",  # i-value
    "0": "\u281A",  # j-value
}
_NUMBER_INDICATOR = "\u283C"  # dots 3-4-5-6

_UEB_PUNCTUATION = {
    " ": "\u2800",  # blank braille cell
    ".": "\u2832",  # dots 2-5-6
    ",": "\u2802",  # dot 2
    ";": "\u2822",  # dots 2-6
    ":": "\u2812",  # dots 2-5
    "!": "\u2816",  # dots 2-3-5
    "?": "\u2826",  # dots 2-3-6
    "'": "\u2804",  # dot 3
    "-": "\u2824",  # dots 3-6
    "/": "\u280C",  # dots 3-4
    "(": "\u2836",  # dots 1-2-6  (UEB opening paren with indicator)
    ")": "\u2836",  # dots 3-4-5  (simplified for v1)
}


@dataclass
class BrailleLabel:
    """A label with both print and braille representations."""
    print_text: str
    braille_text: str
    legend_number: int | None = None


def _char_to_braille(ch: str, in_number: bool) -> tuple[str, bool]:
    """Convert a single character to its UEB Grade 1 braille equivalent.

    Returns (braille_string, currently_in_number_mode).
    """
    if ch.isdigit():
        prefix = _NUMBER_INDICATOR if not in_number else ""
        return prefix + _UEB_DIGITS[ch], True

    # Exiting number mode
    in_number = False

    if ch.isupper():
        lower = ch.lower()
        if lower in _UEB_ALPHA:
            return _CAPITAL_INDICATOR + _UEB_ALPHA[lower], False

    if ch.lower() in _UEB_ALPHA:
        return _UEB_ALPHA[ch.lower()], False

    if ch in _UEB_PUNCTUATION:
        return _UEB_PUNCTUATION[ch], False

    # Unknown character — use empty braille cell as placeholder.
    return "\u2800", False


def transcribe(text: str) -> str:
    """Transcribe a string to UEB Grade 1 braille (Unicode characters).

    This is a v1 implementation using a built-in lookup table.
    Production use should replace this with liblouis for Grade 2
    contracted braille and Nemeth math support.
    """
    if not text:
        return ""

    result = []
    in_number = False
    for ch in text:
        braille, in_number = _char_to_braille(ch, in_number)
        result.append(braille)
    return "".join(result)


def transcribe_labels(labels: list[tuple[str, str]]) -> list[BrailleLabel]:
    """Transcribe a list of (id, text) pairs and assign legend numbers.

    Returns a list of BrailleLabel objects with sequential legend numbers
    starting from 1.
    """
    results = []
    for i, (element_id, text) in enumerate(labels, start=1):
        braille = transcribe(text) if text else ""
        results.append(BrailleLabel(
            print_text=text or "",
            braille_text=braille,
            legend_number=i,
        ))
    return results
