# Static-shard production startup audit

The [operational snapshot](startup_01.json) passed: **4,958 observed native terrain checks and 4,958 settled-start checks were valid**, with no shard failure rows. The snapshot contained 3,653 distinct validated completion entries and 58,102.8 measured seconds. This is an early snapshot while collection continues, not the final dataset inventory or an unbiased goal-success estimate.

Completed outcomes at that snapshot were 3,457 goals, 177 prolonged-blockage stops, eight rollovers, nine terrain-boundary exits and two horizon timeouts. All 177 blockage-stop event sequences retained the declared minimum 24 seconds, two-second confirmation and eight-second recovery tail. The histories also retained 263 canceled pending stops; a cancellation restarts the timing before a later sustained blockage can terminate an episode.

Independent raw-array checks verified two delayed blockage stops at 42.8 and 102.75 seconds. Each had 201 consecutive qualifying two-second windows through its final confirmation and tail; the maximum XY diameters were 0.24477 m and 0.24862 m, respectively, below the 0.25 m bound. Applied throttle and nonparking conditions also held. A rollover endpoint measured roll -60.95 degrees, and a boundary exit measured x=-40.0506 m, satisfying the declared terminal conditions. Every checked episode passed its artifact hashes, rich solver-step integration, post-step risk coverage and actual terminal-state checks.

The static-shard queue has shown no duplicate-claim or native-startup failures in this audit. Frozen collector, controller, physics and runtime files were not modified. Final quota and split accounting remain the responsibility of the global validated-ledger monitor.
