"""Build the nav_v1 task list: every (mission, arm) run, sharded round-robin so each shard mixes arms and arenas."""
import argparse, json
from pathlib import Path

ARMS = {
    'W':   dict(mode='waypoint', period=2.0, latency_s=0.0),
    'R2':  dict(mode='periodic', period=2.0, latency_s=0.0),
    'R1':  dict(mode='periodic', period=1.0, latency_s=0.0),
    'R1L': dict(mode='periodic', period=1.0, latency_s=-2.0),
    'R2L': dict(mode='periodic', period=2.0, latency_s=-2.0),
    'R1S': dict(mode='periodic', period=1.0, latency_s=0.0, keep_current=True, switch_margin=0.0),
    # limited-range sensing: the overhead frame is cropped to 20 m around the vehicle, so a route 25-35 m long
    # reaches past what has been seen and replanning is the only way to learn the rest
    'R1rand': dict(mode='periodic', period=1.0, latency_s=0.0, pick='random'),
    'W20':  dict(mode='waypoint', period=2.0, latency_s=0.0, sense_radius_m=20.0),
    'R2_20': dict(mode='periodic', period=2.0, latency_s=0.0, sense_radius_m=20.0),
    'R1_20': dict(mode='periodic', period=1.0, latency_s=0.0, sense_radius_m=20.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--missions', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--arms', nargs='+', default=['W', 'R2', 'R1'])
    ap.add_argument('--shards', type=int, default=10)
    ap.add_argument('--save-frames', nargs='*', default=[], help='mission ids whose decision frames to store')
    a = ap.parse_args()
    ms = sorted(p.stem for p in Path(a.missions).glob('*.json'))
    # All arms of a mission go in ONE shard: Chrono on this cluster is deterministic per node but not across
    # nodes, so a paired comparison has to run its arms on the same node.
    tasks = []
    for i, m in enumerate(ms):
        for arm in a.arms:
            tasks.append(dict(mission=m, arm=arm, **ARMS[arm], shard=i % a.shards,
                              save_frames=bool(m in a.save_frames)))
    json.dump(tasks, open(a.out, 'w'), indent=1)
    print(f'{len(tasks)} runs over {a.shards} shards ({len(ms)} missions x {len(a.arms)} arms)')


if __name__ == '__main__':
    main()
