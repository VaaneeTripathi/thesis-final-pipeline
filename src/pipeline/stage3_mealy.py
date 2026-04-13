# States
B = "B"   # Blank
I = "I"   # Idle
O = "O"   # Operating
E = "E"   # Erasing

# Input symbols (match VLMOperation.operation_type values)
SIG_C = "CREATION"
SIG_A = "ADDITION"
SIG_H = "HIGHLIGHTING"
SIG_E = "ERASURE"
SIG_D = "COMPLETE_ERASURE"
OMEGA = "pen_lift"        # from Stage 1 temporal segments
TAU   = "idle_timeout"    # from config.IDLE_TIMEOUT

| Current | Input | Next | Output |
|---|---|---|---|
| B | σ_c | O | P |
| B | τ | B | ε |
| I | σ_a | O | S·P |
| I | σ_h | O | S·P |
| I | σ_e | O | S·P |
| I | σ_d | E | S·P |
| I | τ | I | S |
| O | ω | I | S |
| E | ω | B | ε |

