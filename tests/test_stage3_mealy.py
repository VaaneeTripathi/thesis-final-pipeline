"""Unit tests for the Mealy machine (stage3_mealy.py).

Covers every row in the transition table plus error recovery for
undefined (state, input) pairs.
"""
import pytest
from pipeline.stage3_mealy import (
    MealyMachine,
    process_events,
    B, I, O, E,
    SIG_C, SIG_A, SIG_H, SIG_E, SIG_D, OMEGA, TAU,
    OUT_S, OUT_P, OUT_SP, OUT_EPS,
)

def test_b_creation_to_o_emits_p():
    m = MealyMachine()
    assert m.state == B
    out = m.step(SIG_C)
    assert m.state == O
    assert out == OUT_P


def test_b_tau_stays_b_emits_eps():
    m = MealyMachine()
    out = m.step(TAU)
    assert m.state == B
    assert out == OUT_EPS


def test_i_addition_to_o_emits_sp():
    m = MealyMachine(state=I)
    out = m.step(SIG_A)
    assert m.state == O
    assert out == OUT_SP


def test_i_highlighting_to_o_emits_sp():
    m = MealyMachine(state=I)
    out = m.step(SIG_H)
    assert m.state == O
    assert out == OUT_SP


def test_i_erasure_to_o_emits_sp():
    m = MealyMachine(state=I)
    out = m.step(SIG_E)
    assert m.state == O
    assert out == OUT_SP


def test_i_complete_erasure_to_e_emits_sp():
    m = MealyMachine(state=I)
    out = m.step(SIG_D)
    assert m.state == E
    assert out == OUT_SP


def test_i_tau_stays_i_emits_s():
    m = MealyMachine(state=I)
    out = m.step(TAU)
    assert m.state == I
    assert out == OUT_S


def test_o_penlift_to_i_emits_s():
    m = MealyMachine(state=O)
    out = m.step(OMEGA)
    assert m.state == I
    assert out == OUT_S


def test_e_penlift_to_b_emits_eps():
    m = MealyMachine(state=E)
    out = m.step(OMEGA)
    assert m.state == B
    assert out == OUT_EPS


def test_full_creation_sequence():
    """B --CREATION--> O --pen_lift--> I"""
    m = MealyMachine()
    assert m.step(SIG_C) == OUT_P
    assert m.state == O
    assert m.step(OMEGA) == OUT_S
    assert m.state == I


def test_addition_cycle():
    """I --ADDITION--> O --pen_lift--> I (repeatable)"""
    m = MealyMachine(state=I)
    for _ in range(3):
        assert m.step(SIG_A) == OUT_SP
        assert m.state == O
        assert m.step(OMEGA) == OUT_S
        assert m.state == I


def test_complete_erasure_cycle():
    """I --COMPLETE_ERASURE--> E --pen_lift--> B"""
    m = MealyMachine(state=I)
    assert m.step(SIG_D) == OUT_SP
    assert m.state == E
    assert m.step(OMEGA) == OUT_EPS
    assert m.state == B


def test_lecture_sequence_22_operations():
    """Approximate sequence for architecture-lecture-9mins.mp4:
    4 CREATIONs, 15 ADDITIONs, 3 COMPLETE_ERASUREs."""
    m = MealyMachine()

    # First creation: B -> O -> I
    assert m.step(SIG_C) == OUT_P
    assert m.step(OMEGA) == OUT_S
    assert m.state == I

    # Second creation is actually ADDITION from I
    assert m.step(SIG_A) == OUT_SP
    assert m.step(OMEGA) == OUT_S
    assert m.state == I

    # Complete erasure cycle
    assert m.step(SIG_D) == OUT_SP
    assert m.step(OMEGA) == OUT_EPS
    assert m.state == B

    # New creation after erasure
    assert m.step(SIG_C) == OUT_P
    assert m.step(OMEGA) == OUT_S
    assert m.state == I


def test_idle_timeout_emits_snapshot():
    """Idle timeout from I should emit S (snapshot) and stay in I."""
    m = MealyMachine(state=I)
    out = m.step(TAU)
    assert out == OUT_S
    assert m.state == I



def test_history_recorded():
    m = MealyMachine()
    m.step(SIG_C, context={"timestamp": 5.0})
    assert len(m.history) == 1
    entry = m.history[0]
    assert entry["from_state"] == B
    assert entry["to_state"] == O
    assert entry["symbol"] == SIG_C
    assert entry["output"] == OUT_P
    assert entry["context"] == {"timestamp": 5.0}


def test_reset_clears_state():
    m = MealyMachine()
    m.step(SIG_C)
    m.step(OMEGA)
    assert m.state == I
    m.reset()
    assert m.state == B
    assert m.history == []



def test_undefined_transition_emits_eps_no_state_change():
    """CREATION from state I is undefined -- should log warning, not crash."""
    m = MealyMachine(state=I)
    out = m.step(SIG_C)
    assert out == OUT_EPS
    assert m.state == I  # state unchanged
    assert m.history[0]["error"] == "undefined_transition"


def test_undefined_transition_from_o():
    """CREATION from state O is undefined."""
    m = MealyMachine(state=O)
    out = m.step(SIG_C)
    assert out == OUT_EPS
    assert m.state == O


def test_b_addition_treated_as_creation():
    """VLM misclassifies first drawing as ADDITION — machine should still emit P and move to O."""
    m = MealyMachine()
    out = m.step(SIG_A)
    assert out == OUT_P
    assert m.state == O


def test_b_highlighting_treated_as_creation():
    m = MealyMachine()
    out = m.step(SIG_H)
    assert out == OUT_P
    assert m.state == O


def test_b_erasure_is_noop():
    m = MealyMachine()
    out = m.step(SIG_E)
    assert out == OUT_EPS
    assert m.state == B


def test_b_complete_erasure_is_noop():
    m = MealyMachine()
    out = m.step(SIG_D)
    assert out == OUT_EPS
    assert m.state == B


def test_b_omega_is_noop():
    m = MealyMachine()
    out = m.step(OMEGA)
    assert out == OUT_EPS
    assert m.state == B


def test_o_consecutive_addition_emits_p():
    """Two ADDITIONs without pen-lift in between — both should emit P."""
    m = MealyMachine(state=O)
    out = m.step(SIG_A)
    assert out == OUT_P
    assert m.state == O


def test_o_highlighting_emits_p():
    m = MealyMachine(state=O)
    out = m.step(SIG_H)
    assert out == OUT_P
    assert m.state == O


def test_o_erasure_moves_to_e():
    m = MealyMachine(state=O)
    out = m.step(SIG_E)
    assert out == OUT_P
    assert m.state == E


def test_o_complete_erasure_moves_to_e():
    m = MealyMachine(state=O)
    out = m.step(SIG_D)
    assert out == OUT_P
    assert m.state == E


def test_o_tau_moves_to_i():
    m = MealyMachine(state=O)
    out = m.step(TAU)
    assert out == OUT_EPS
    assert m.state == I


def test_e_addition_moves_to_o():
    m = MealyMachine(state=E)
    out = m.step(SIG_A)
    assert out == OUT_P
    assert m.state == O


def test_e_tau_moves_to_b():
    m = MealyMachine(state=E)
    out = m.step(TAU)
    assert out == OUT_EPS
    assert m.state == B


def test_all_addition_sequence_produces_output():
    """Full sequence where VLM returns only ADDITIONs (no CREATION) — should produce real output."""
    m = MealyMachine()
    outputs = []
    for _ in range(5):
        outputs.append(m.step(SIG_A))  # ADDITION from B or I
        outputs.append(m.step(OMEGA))  # pen-lift

    # First ADDITION from B → P (implicit creation)
    assert outputs[0] == OUT_P
    # First OMEGA → S
    assert outputs[1] == OUT_S
    # Subsequent ADDITIONs from I → SP
    assert all(o == OUT_SP for o in outputs[2::2])
    # Subsequent OMEGAs → S
    assert all(o == OUT_S for o in outputs[3::2])


def test_process_events_returns_history():
    events = [
        {"symbol": SIG_C},
        {"symbol": OMEGA, "context": {"segment_id": 1}},
        {"symbol": TAU},
    ]
    history = process_events(events)
    assert len(history) == 3
    assert history[0]["output"] == OUT_P
    assert history[1]["output"] == OUT_S
    assert history[2]["output"] == OUT_S


def test_process_events_fresh_machine_each_call():
    events = [{"symbol": SIG_C}]
    h1 = process_events(events)
    h2 = process_events(events)
    assert h1[0]["from_state"] == B
    assert h2[0]["from_state"] == B
