from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

root = Path('/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/full_cohort_v2')
source = root.parent/'snapshots/campaign_v2'
declaration_path = next((root/'batches').glob('declaration*.json'))
declaration = json.loads(declaration_path.read_text())
common = declaration['common_contract']


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


checks = []
for kind, mapping in [('source', common['source_sha256']), ('runtime', common['runtime']['file_sha256'])]:
    for name, expected in mapping.items():
        path = source/name if kind == 'source' else Path(name)
        actual = digest(path)
        assert actual == expected, (kind, name, expected, actual)
        checks.append({'kind': kind, 'file': str(path), 'sha256': actual, 'bytes': path.stat().st_size})
report = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'job_id': '412066',
          'declaration_sha256': digest(declaration_path), 'all_current_files_match_initial_fingerprints': True,
          'source_files_checked': sum(row['kind'] == 'source' for row in checks),
          'runtime_files_checked': sum(row['kind'] == 'runtime' for row in checks),
          'total_bytes_checked': sum(row['bytes'] for row in checks), 'checks': checks,
          'scope': 'Current frozen source/runtime compared to launch fingerprint. No protected-test outputs inspected.'}
print(json.dumps(report, indent=2))
