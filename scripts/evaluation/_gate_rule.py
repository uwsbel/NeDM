"""The declared action-sensitivity verdict rule, in one place.

Kept in its own module with NO imports so that any tool can read it without
pulling in torch. Two tools answer the same question -- the gate for a single
surrogate, gate_seed_aggregate for a set of surrogate seeds -- and a second copy
of these numbers is exactly how a threshold drifts between them unnoticed.

Declared before any arm existed and NOT changed since.
"""
PRIMARY = "body_vel"      # named explicitly, never selected by prefix: selecting
                          # "vel_body" by prefix gave 2 channels for the 34-channel
                          # state and 3 for the 40-channel one, so the headline corr
                          # was computed over different channel sets per arm
VH = ["0.5", "1.0"]       # declared horizons
GLO, GHI = 0.5, 2.0       # gain must sit within 2x either way
CMIN = 0.5                # corr and cosine floor
NMIN = 8                  # fewest usable pairs a horizon may be judged on
