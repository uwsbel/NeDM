# Lessons learned

Findings that cost real time and would cost it again. **Add an entry whenever
something confuses you for more than an hour.**

Scope split:

- **This folder** — cross-cutting lessons: simulator gotchas, representation
  traps, tooling. Things a future agent needs *before* it starts.
- [`.claude/lessons_learned.md`](../../../.claude/lessons_learned.md) — the
  original running notes, RL-configuration focused. Still authoritative for what
  it covers; not superseded.
- `docs/vision/**/implementation_notes.md` — per-study narrative, including what
  was tried and abandoned. Richer, but you have to know which study to read.

| File | Covers |
|---|---|
| [`chrono-sensor.md`](chrono-sensor.md) | Chrono::Sensor camera and depth gotchas |
| [`representation-traps.md`](representation-traps.md) | Latent and reconstruction failure modes |
| [`rl-in-nrd.md`](rl-in-nrd.md) | Training policies inside a learned model |
| [`chrono-versions.md`](chrono-versions.md) | Chrono 9 vs 10 API moves, and the OptiX/driver coupling |

## Template

```markdown
## <short imperative title>

**Cost:** <how long this took to find> · **Found:** <date> · **Applies to:** <scope>

**Expected:** ...
**Happened:** ...
**Cause:** ...
**Fix:** ...
**Evidence:** <file:line, run name, or measured number>
```

Write the *cause*, not just the fix. A fix without a cause gets cargo-culted
into places it does not apply.
