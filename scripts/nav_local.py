"""Run scripts/nav_runner.py on luffy with the source-built Chrono (OptiX on the RTX 5090).

The build at /home/harry/chrono/build was compiled against the system Python 3.12 and its numpy 1.26, while torch
only exists in the conda env (built against numpy 2). Loading the system numpy first and appending the conda
site-packages afterwards gives one process with the Chrono build, numpy 1.26 and CUDA torch (checked: tensors
round-trip, numpy stays 1.26).

  PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 scripts/nav_local.py <nav_runner arguments>
"""
import runpy, sys
from pathlib import Path
import numpy  # noqa: F401  -- must be the system numpy, before anything from the conda env

CONDA_SITE = '/home/harry/miniconda3/envs/nedm/lib/python3.12/site-packages'
if CONDA_SITE not in sys.path:
    sys.path.append(CONDA_SITE)
runner = str(Path(__file__).resolve().parent / 'nav_runner.py')
sys.argv = [runner] + sys.argv[1:]
runpy.run_path(runner, run_name='__main__')
