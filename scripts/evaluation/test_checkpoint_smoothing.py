"""Verify window=1 reproduces the old behaviour exactly, and window>1 smooths."""
import statistics as st
class T:
    def __init__(self, w):
        self.checkpoint_metric_window = w
        self._ckpt_metric_history = []
    def _smoothed_metric(self, value):
        self._ckpt_metric_history.append(value)
        w = self.checkpoint_metric_window
        if w <= 1: return value
        window = self._ckpt_metric_history[-w:]
        o = sorted(window); mid = len(o)//2
        return o[mid] if len(o) % 2 else 0.5*(o[mid-1]+o[mid])

seq = [1.855, 2.167, 2.603, 1.945, 3.331, 0.524, 3.574, 2.773, 3.038, 2.206]
for w in (1, 5):
    t = T(w)
    out = [round(t._smoothed_metric(v), 3) for v in seq]
    best = min(out)
    print(f"  window={w}: best selected = {best}   series={out}")
assert [T(1)._smoothed_metric(v) for v in seq] == seq, "window=1 must be identity"
print("  window=1 is exactly the identity -> existing configs reproduce")
