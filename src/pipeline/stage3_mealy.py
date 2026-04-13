from __future__ import annotations
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
B = "B"   # Blank
I = "I"   # Idle
O = "O"   # Operating
E = "E"   # Erasing

# ---------------------------------------------------------------------------
# Input symbols
# ---------------------------------------------------------------------------
SIG_C = "CREATION"
SIG_A = "ADDITION"
SIG_H = "HIGHLIGHTING"
SIG_E = "ERASURE"
SIG_D = "COMPLETE_ERASURE"
OMEGA = "pen_lift"      # activity segment ends (pen-lift from Stage 1)
TAU   = "idle_timeout"  # no activity for > IDLE_TIMEOUT seconds

# ---------------------------------------------------------------------------
# Output tokens
# ---------------------------------------------------------------------------
OUT_S   = "S"   # emit static IR snapshot
OUT_P   = "P"   # emit operation IR
OUT_SP  = "SP"  # emit both
OUT_EPS = ""    # epsilon -- emit nothing

# ---------------------------------------------------------------------------
# Transition table  (state, input) -> (next_state, output)
# Defined exactly per intermediate-representation/README.md lines 248-342.
# ---------------------------------------------------------------------------
_TABLE: dict[tuple[str, str], tuple[str, str]] = {
    (B, SIG_C): (O, OUT_P),
    (B, TAU):   (B, OUT_EPS),
    (I, SIG_A): (O, OUT_SP),
    (I, SIG_H): (O, OUT_SP),
    (I, SIG_E): (O, OUT_SP),
    (I, SIG_D): (E, OUT_SP),
    (I, TAU):   (I, OUT_S),
    (O, OMEGA): (I, OUT_S),
    (E, OMEGA): (B, OUT_EPS),
}


@dataclass
class MealyMachine:
    """Formal Mealy transducer driving IR schema generation.

    Inputs come from two sources:
      - VLMOperation.operation_type  (SIG_C / SIG_A / SIG_H / SIG_E / SIG_D)
      - Temporal events from Stage 1 (OMEGA) and timeout logic (TAU)

    Outputs:
      S  -> trigger Stage 4 (static IR snapshot)
      P  -> trigger Stage 5 (operation IR)
      SP -> trigger both
      "" -> epsilon, no output
    """
    state: str = B
    history: list[dict] = field(default_factory=list)

    def step(self, symbol: str, context: dict | None = None) -> str:
        """Consume one input symbol, transition, return output token."""
        key = (self.state, symbol)
        if key not in _TABLE:
            log.warning(
                "Undefined transition (%s, %s) -- state unchanged, emitting epsilon. "
                "Context: %s",
                self.state, symbol, context,
            )
            self.history.append({
                "from_state": self.state,
                "symbol": symbol,
                "to_state": self.state,
                "output": OUT_EPS,
                "error": "undefined_transition",
                "context": context,
            })
            return OUT_EPS

        next_state, output = _TABLE[key]
        self.history.append({
            "from_state": self.state,
            "symbol": symbol,
            "to_state": next_state,
            "output": output,
            "context": context,
        })
        log.debug(
            "Mealy: %s --%s--> %s  [%s]",
            self.state, symbol, next_state, output or "epsilon",
        )
        self.state = next_state
        return output

    def reset(self) -> None:
        self.state = B
        self.history.clear()


def process_events(events: list[dict]) -> list[dict]:
    """Run a fresh Mealy machine over a sequence of events.

    Each event dict must have:
        "symbol"  : one of the input constants above
        "context" : (optional) arbitrary metadata passed through to history

    Returns the history list after all events have been processed.
    Each entry has: from_state, symbol, to_state, output, context.
    """
    machine = MealyMachine()
    for event in events:
        symbol = event["symbol"]
        context = event.get("context")
        machine.step(symbol, context)
    return machine.history
