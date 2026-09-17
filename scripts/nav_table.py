"""Markdown tables from nav_analyze's summary.json, ready to paste into the report."""
import argparse, json


def row(a, s, ref=None, paired=None):
    n = s['n']
    cells = [a, f"{s['complete']}/{n}", f"{100*s['waypoint_rate']:.1f}%", f"{s['unsafe']}",
             f"{s['legs']['with_slide']}/{s['legs']['n']}", f"{s['median_time_s']:.1f} s",
             f"{s['decisions_mean']:.0f}", f"{s['lat_p50_s']:.1f} s"]
    if paired:
        cells.append(f"{paired['time_mean_diff_s']:+.1f} [{paired['time_ci'][0]:+.1f}, {paired['time_ci'][1]:+.1f}]")
        cells.append(f"{paired['complete_arm_only']} / {paired['complete_ref_only']}")
    else:
        cells += ['-', '-']
    return '| ' + ' | '.join(cells) + ' |'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True); ap.add_argument('--reference', default='W')
    a = ap.parse_args()
    S = json.load(open(a.summary))
    arms = [x for x in ('W', 'R2', 'R1', 'R1L', 'R1S', 'R1rand', 'W20', 'R2_20', 'R1_20') if x in S['by_arm']]
    head = ('| arm | missions completed | waypoints reached | unsafe missions | legs with a slide | median time '
            '| decisions | sense-to-plan (p50) | travel time vs ' + a.reference + ' | completed only by (arm / '
            + a.reference + ') |')
    print(head)
    print('|' + '---|' * (head.count('|') - 1))
    for arm in arms:
        print(row(arm, S['by_arm'][arm], paired=S['paired_vs_reference'].get(arm)))
    print()
    print('| arm | dev: completed | dev: waypoints | unseen: completed | unseen: waypoints |')
    print('|---|---|---|---|---|')
    for arm in arms:
        d, u = S['by_arm_dev'].get(arm, {}), S['by_arm_unseen'].get(arm, {})
        if not d or not u:
            continue
        print(f"| {arm} | {d['complete']}/{d['n']} | {100*d['waypoint_rate']:.1f}% | "
              f"{u['complete']}/{u['n']} | {100*u['waypoint_rate']:.1f}% |")
    print()
    print('| arm | ' + ' | '.join(sorted({k for arm in arms for k in S['by_arm'][arm]['statuses']})) + ' |')
    keys = sorted({k for arm in arms for k in S['by_arm'][arm]['statuses']})
    print('|' + '---|' * (len(keys) + 1))
    for arm in arms:
        st = S['by_arm'][arm]['statuses']
        print(f"| {arm} | " + ' | '.join(str(st.get(k, 0)) for k in keys) + ' |')
    print()
    print('| arm | render | back-project | candidates | corridors | risk model | total (p50) |')
    print('|---|---|---|---|---|---|---|')
    for arm in arms:
        L = S['latency'][arm]
        g = lambda k: f"{L.get(k, {}).get('p50', float('nan')):.2f} s"
        print(f"| {arm} | {g('render_s')} | {g('backproject_s')} | {g('propose_s')} | {g('corridor_s')} | "
              f"{g('score_s')} | {L['sense_to_plan_s']['p50']:.2f} s |")


if __name__ == '__main__':
    main()
