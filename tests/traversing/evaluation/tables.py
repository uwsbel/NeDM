"""The schemas of ``traversing/results/*.csv`` and their row builders (not a test module): the released tables rebuilt
from relabelled drives (test_labels A1) and the per-table policy for drives without a label (labels.ON_MISSING).
Moved out of the library: no pipeline step writes these tables (Task.meta holds none of their id columns).
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass

from nedm.traversing.evaluation.labels import UNLABELLED, _status, drive_labels


def code(src, label):
    """'S' | 's' | 'U' | 'F' of one drive (run folder, mapping or Record) under `label`."""
    return drive_labels(src, label)['code']


@dataclass(frozen=True)
class Table:
    """One traversing/results CSV. Wide: a row per task, a code column per arm. Long (header given): a row per
    (task, arm) with the label's own columns, task by task (M3) or arm by arm (M1, arm_major)."""
    ids: tuple                     # task columns (Task.meta keys), CSV order
    arms: dict                     # arm column (wide) or 'arm' value (long) -> label kind
    on_missing: str                # crash / launch_failed cell: 'drop_pair' (every arm of the task '-'), 'U' or '-'
    header: tuple = ()
    arm_major: bool = False

    @property
    def columns(self):
        return self.header or self.ids + tuple(self.arms)


def _arms(names, label):
    return {a: label for a in names.split()}


_M2_IDS = ('group_id', 'arena', 'world', 'suite_stratum', 'terrain_stratum')
_M2_3S = 'oracle_tag_3s pooled_3s shared_masked_3s shared_hist_3s'
_M4A_IDS = ('pair_id', 'arena', 'arena_role', 'eval_set', 'terrain_cluster')
_M4A = ('f104_only_ens1{0} f104_only_ens2{0} two_arenas_same_total{0} three_arenas_same_total_ens1{0} '
        'three_arenas_same_total_ens2{0} three_arenas_all_data{0}')
_M4B_IDS = ('task_id', 'arena', 'world', 'suite_part', 'terrain_cluster', 'task_type')
# on_missing: labels.ON_MISSING (test_tables_on_missing_is_the_library_policy)
TABLES = {
    'm1_navigation_missions': Table(
        ('mission_id', 'arena', 'arena_seen_in_training', 'arena_group'),
        _arms('plan_once_per_waypoint replan_every_2s replan_every_1s replan_every_1s_delay_charged', 'mission'), '-',
        ('arm', 'mission_id', 'arena', 'arena_seen_in_training', 'arena_group', 'waypoints_total', 'waypoints_reached',
         'status', 'outcome_code', 'backward_slide', 'elapsed_s', 'mission_outcome_sha256'), arm_major=True),
    'm2_shared_risk_soil': Table(_M2_IDS, _arms(
        'specialist_soil_standing specialist_rigid_standing oracle_tag_standing shared_hist_standing '
        f'specialist_soil_3s specialist_rigid_3s {_M2_3S} specialist_soil_1s oracle_tag_1s pooled_1s shared_hist_1s '
        'shared_hist_early_rows_1s transformer_1s oracle_tag_0p5s pooled_0p5s shared_hist_0p5s '
        'shared_hist_early_rows_0p5s transformer_0p5s shared_hist_early_rows_0p5s_grad transformer_0p5s_grad',
        'rollback'), 'drop_pair'),
    'm2_shared_risk_rigid': Table(_M2_IDS, _arms(
        'specialist_rigid_standing specialist_soil_standing oracle_tag_standing shared_hist_standing '
        f'specialist_rigid_3s specialist_soil_3s {_M2_3S} shared_hist_early_rows_0p5s shared_hist_early_rows_0p5s_grad',
        'rollback'), 'drop_pair'),
    'm3_tracker_routes': Table(
        ('world', 'route_id', 'stratum'), _arms('pid_native pid_held_50ms nrd_policy_v2', 'tracker'), '-',
        ('world', 'route_id', 'stratum', 'arm', 'status', 'completed', 'unsafe', 'outcome_code', 'near_stop_40s_fired',
         'xtrack_station_winsor_mean_m', 'speed_abs_err_mean_mps', 'mean_abs_action_change', 'positive_work_kj')),
    'm4_unseen_arenas_hmmwv_soil': Table(_M4A_IDS, _arms(_M4A.format('') + ' straight_route_6mps', 'rollback'), '-'),
    'm4_unseen_arenas_hmmwv_rigid': Table(_M4A_IDS, _arms(
        _M4A.format('_fixed2mps') + ' straight_route_2mps ' + _M4A.format('_speedfree') + ' straight_route_6mps',
        'rollback'), '-'),
    'm4_vehicles_f104_soil': Table(_M4B_IDS, {  # the Gator-study index has no belly field; the HMMWV no belly record
        **_arms('gator_own_model_sampling', 'rollback'), **_arms('gator_own_model_sampling_grad', 'rollback_belly'),
        **_arms('gator_own_model_tiers0to6_sampling gator_hmmwv_model_sampling gator_straight_6mps '
                'hmmwv_own_model_sampling hmmwv_own_model_sampling_grad hmmwv_straight_6mps', 'rollback'),
        **_arms('polaris_own_model_sampling_grad polaris_own_model_sampling polaris_corrected_driveline_grad_routes '
                'polaris_straight_6mps', 'rollback_belly')}, '-'),
    'm4_polaris_unseen_soil': Table(
        ('task_id', 'arena', 'arena_kind', 'world', 'terrain_cluster', 'task_type'),
        _arms('polaris_own_model_sampling_grad polaris_own_model_sampling polaris_straight_6mps', 'rollback_belly'),
        'U'),
    'm4_vehicle_smoke': Table(  # hmmwv_stored (no belly record) and m113_* (refused vehicle): released cells only
        ('route_id', 'group_id', 'route_kind', 'speed_profile', 'task_type'), _arms(
            'gator_redrive_1ms gator_redrive_0p5ms hmmwv_stored polaris_stock polaris_power_corrected '
            'polaris_open_diff polaris_soil_wheels_0p33m m113_stock_gearing m113_regeared_4x', 'goal_belly'), '-'),
}
_FMT = dict(xtrack_station_winsor_mean_m='{:.4f}', speed_abs_err_mean_mps='{:.4f}', mean_abs_action_change='{:.6f}',
            positive_work_kj='{:.2f}', elapsed_s='{:.2f}')


def _cell(t, arm, lab, drop):
    if lab is None or drop:
        return '-'
    if isinstance(lab, str):
        _status(lab, UNLABELLED)
        return t.on_missing
    if lab['label'] != t.arms[arm]:
        raise ValueError(f'{arm}: labelled {lab["label"]!r}, the table reads {t.arms[arm]!r}')
    return lab['code']


def table_rows(name, entries):
    """Rows (header first) of TABLES[name]. entries: (meta, cells) per task in table order; cells maps an arm to its
    drive_labels dict, 'crash' or 'launch_failed'; an arm missing from cells was not run ('-'; no long-table row)."""
    t, rows, entries = TABLES[name], [], list(entries)
    keys = [tuple(str(meta.get(c)) for c in t.ids) for meta, _ in entries]
    n = Counter(keys)
    for (meta, cells), k in zip(entries, keys):
        if set(cells) - set(t.arms) or set(t.ids) - set(meta) or n[k] > 1:
            raise KeyError(f'{name} task {k}: unknown arms {sorted(set(cells) - set(t.arms))}, missing task columns '
                           f'{sorted(set(t.ids) - set(meta))} or a duplicate task')
    if not t.header:
        for meta, cells in entries:
            drop = t.on_missing == 'drop_pair' and any(isinstance(v, str) for v in cells.values())
            rows.append([str(meta[c]) for c in t.ids] + [_cell(t, a, cells.get(a), drop) for a in t.arms])
        return [list(t.columns)] + rows
    order = [(a, e) for a in t.arms for e in entries] if t.arm_major else [(a, e) for e in entries for a in t.arms]
    for arm, (meta, cells) in order:
        if cells.get(arm) is not None:
            lab = cells[arm]
            v = {**{c: meta[c] for c in t.ids}, 'arm': arm, 'outcome_code': _cell(t, arm, lab, False)}
            if isinstance(lab, str):                        # no drive: its status, the drive columns empty
                rows.append([str({**v, 'status': lab}.get(c, '')) for c in t.header])
            else:
                v = {**lab, **v}
                rows.append([_FMT[c].format(v[c]) if c in _FMT else str(v[c]) for c in t.header])
    return [list(t.header)] + rows


def table_csv(name, entries):
    """CSV text of TABLES[name] as released ('\\n' line ends)."""
    buf = io.StringIO()
    csv.writer(buf, lineterminator='\n').writerows(table_rows(name, entries))
    return buf.getvalue()
