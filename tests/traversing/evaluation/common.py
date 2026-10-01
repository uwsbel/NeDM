"""Shared switches of the evaluation tests (not a test module): the release restore base and its Hub mirror."""
import os
from pathlib import Path

DATA = Path(os.environ['NEDM_DATA']) if os.environ.get('NEDM_DATA') else None
# tar members (decision states, pick folders, smoke tasks) are checked through their item's index.csv.gz in the mirror
CACHE_DIR = DATA and Path(os.environ.get('NEDM_RELEASE_CACHE') or DATA / 'artifacts/hf_release/download')
CACHE = bool(CACHE_DIR and (CACHE_DIR / 'traversing/evaluation').is_dir())
