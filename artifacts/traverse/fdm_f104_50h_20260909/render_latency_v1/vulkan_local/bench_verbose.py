"""Run the shared render_latency_bench.py unchanged, but with Chrono::Sensor verbose output switched on.

With verbose on, the Vulkan-RT engine prints the physical device it opened ("Chrono::Sensor Vulkan RT device: <name>")
or, if device creation fails, "Chrono::Sensor Vulkan RT GPU unavailable, using CPU fallback: ..." -- without it a
silent fallback to Chrono's own host (CPU) tracer would be indistinguishable from lavapipe in the timing logs.
All arguments are passed straight through to the harness.
"""
import runpy, sys
from pathlib import Path

import pychrono.sensor as sens

_Base = sens.ChSensorManager


class VerboseManager(_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SetVerbose(True)


sens.ChSensorManager = VerboseManager
harness = Path('/home/harry/NeDM-traverse_mppi/scripts/render_latency_bench.py')
sys.argv = [str(harness)] + sys.argv[1:]
runpy.run_path(str(harness), run_name='__main__')
