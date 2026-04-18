import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tactile.braille import transcribe, transcribe_labels, BrailleLabel


def test_transcribe_simple_word():
    result = transcribe("Computer")
    assert len(result) > 0
    # Should start with capital indicator (dot 6) for uppercase C
    assert result[0] == "\u2820"


def test_transcribe_lowercase():
    result = transcribe("screen")
    # 's' = dots 2-3-4 = \u280E
    assert result[0] == "\u280E"


def test_transcribe_empty():
    assert transcribe("") == ""


def test_transcribe_with_space():
    result = transcribe("a b")
    # 'a' + space + 'b'
    assert "\u2800" in result  # blank braille cell for space


def test_transcribe_digits():
    result = transcribe("42")
    # Should start with number indicator
    assert result[0] == "\u283C"


def test_transcribe_labels_assigns_numbers():
    labels = [("n1", "Computer"), ("n2", "Screen"), ("n3", "Keyboard")]
    results = transcribe_labels(labels)
    assert len(results) == 3
    assert all(isinstance(r, BrailleLabel) for r in results)
    assert results[0].legend_number == 1
    assert results[1].legend_number == 2
    assert results[2].legend_number == 3
    assert results[0].print_text == "Computer"
    assert len(results[0].braille_text) > 0


def test_transcribe_labels_console_table(capsys):
    """Print a readable table for M4 demonstration."""
    labels = [
        ("n1", "Computer"),
        ("n2", "Screen"),
        ("n3", "Keyboard"),
        ("n4", "Mouse"),
        ("n5", "Speakers"),
    ]
    results = transcribe_labels(labels)
    print("\n--- M4 Braille Transcription ---")
    print(f"{'#':<4} {'Print':<12} {'Braille'}")
    print("-" * 36)
    for r in results:
        print(f"{r.legend_number:<4} {r.print_text:<12} {r.braille_text}")
    print("-" * 36)
