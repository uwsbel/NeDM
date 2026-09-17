from pathlib import Path
from datetime import datetime, timezone
import json
import collections

root = Path('/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/full_cohort_v2')
declaration = json.loads(next((root/'batches').glob('declaration*.json')).read_text())
tasks = {t['task_id']: t for t in declaration['tasks']}
assert len(tasks) == 450
expected_splits = collections.Counter(t['split'] for t in tasks.values())
assert dict(expected_splits) == {'train': 360, 'val': 90}
assert len(declaration['observations']) == 30
assert all('/snapshots/campaign_v2/' in t['case'] for t in tasks.values())
ledger = next((root/'batches').glob('ledger*.jsonl'))
rows = [json.loads(line) for line in ledger.read_text().splitlines()]
errors = [r for r in rows if r['status'] == 'failed']
seen = set()
outcomes, positive = collections.Counter(), collections.Counter()
sample_total = interval_total = solver_total = post_total = 0
field_coverage = {}
raw_bytes = joins = 0
source_sets, runtime_sets = set(), set()
for path in (root/'raw').glob('*/*/batch_complete.json'):
    marker = json.loads(path.read_text())
    task = marker['contract']['task']
    assert task == tasks[task['task_id']]
    assert task['task_id'] not in seen
    seen.add(task['task_id'])
    outcome = json.loads((path.parent/'outcome.json').read_text())
    outcomes[outcome['status']] += 1
    for key in ['asset_contact', 'sustained_near_stop', 'bounded_blockage_v1', 'safe_goal_reached']:
        positive[key] += bool(outcome[key])
    rich = json.loads((path.parent/'rich_telemetry.json').read_text())
    n = rich['interval_rows']
    assert rich['sample_rows_including_terminal'] == n+1
    assert n == outcome['frames']
    sample_total += n+1
    interval_total += n
    solver_total += rich['solver_step_work_interval_count']
    post_total += rich['post_step_risk_interval_count']
    assert rich['solver_step_work_interval_count'] == n
    assert rich['post_step_risk_interval_count'] == n
    for key, spec in rich['fields'].items():
        counter = field_coverage.setdefault(key, {'finite': 0, 'missing': 0, 'episodes': 0})
        counter['finite'] += spec['finite_count']
        counter['missing'] += spec['missing_count']
        counter['episodes'] += 1
    join = json.loads((path.parent/'observation_join.json').read_text())
    assert join['matched']
    joins += 1
    provenance = json.loads((path.parent/'simulation_provenance.json').read_text())
    source_sets.add(json.dumps(provenance['source_sha256'], sort_keys=True))
    runtime_sets.add(json.dumps(provenance['runtime_sha256'], sort_keys=True))
    raw_bytes += sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file())
observations = []
for path in (root/'observations').glob('*/batch_complete.json'):
    data = json.loads((path.parent/'observation.json').read_text())
    observations.append({'scene': path.parent.name, 'valid_depth_fraction': data['valid_depth_fraction'],
                         'raw_size': [data['camera']['width'], data['camera']['height']],
                         'model_image_size': data['camera']['model_image_size']})
report = {
    'observed_utc': datetime.now(timezone.utc).isoformat(), 'job_id': '412066',
    'declared_routes': 450, 'declared_splits': dict(expected_splits), 'completed_routes': len(seen),
    'remaining_routes': 450-len(seen), 'declared_observations': 30, 'completed_observations': len(observations),
    'errors': errors, 'outcome_counts': dict(outcomes), 'positive_episode_flags': dict(positive),
    'sample_rows_including_terminals': sample_total, 'physics_intervals': interval_total,
    'solver_step_work_intervals': solver_total, 'post_step_risk_intervals': post_total,
    'observation_joins_matched': joins, 'distinct_source_maps': len(source_sets),
    'distinct_runtime_maps': len(runtime_sets),
    'rich_fields_missing': {k: v for k, v in field_coverage.items() if v['missing']},
    'rich_field_count': len(field_coverage), 'rich_field_coverage': field_coverage,
    'raw_bytes_completed': raw_bytes, 'observations': observations,
    'depth_fraction_note': 'Global camera footprint exceeds 240 m terrain; outside-terrain rays are invalid. Full geometry checks assess in-arena coverage separately.',
    'scope': 'Completed outputs from train/validation job 412066 only. No protected-test directory is inspected.'}
print(json.dumps(report, indent=2))
