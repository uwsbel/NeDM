#!/usr/bin/env python3
"""Scientific report from actual online Chrono outputs; optional CPU forecast replay.

No scene geometry, future sensor image, authored hazard or training routine is
opened. Forecast replay reconstructs only the recorded causal history prefix.
Reference lines are commands, forecasts are finite model predictions, and
physical curves/video are separately measured simulator outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k].copy() for k in values.files}


def read_json(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def parse_run(value):
    label, sep, location = value.partition('=')
    if not sep:
        location, label = value, Path(value).name
    return label, Path(location).resolve()


def slug(label):
    result = re.sub(r'[^A-Za-z0-9_-]+', '_', label).strip('_')
    if not result:
        raise ValueError('Run label needs an alphanumeric character')
    return result


def require_hash(path, expected, description):
    actual = sha(path)
    if expected != actual:
        raise ValueError(f'{description} SHA mismatch: {path}')
    return actual


def observed_map(obs, camera, half, size=600):
    """Bin registered measured depth hits in XY for visualization only."""
    from nedm.traverse.camera import CameraModel
    model = CameraModel(width=int(camera['width']), height=int(camera['height']),
                        hfov_rad=float(camera['hfov_rad']), cam_height_m=float(camera['cam_height_m']))
    depth = np.asarray(obs['depth_m'], float)
    if depth.shape != (model.height, model.width) or obs['rgb'].shape != (*depth.shape, 3):
        raise ValueError('Raw observation and declared camera shapes disagree')
    x, y, z = model.depth_to_world(depth, convention='ray', ray_scale=float(camera['depth_ray_scale']))
    valid = np.isfinite(depth) & (depth > 0.) & (depth < camera['max_depth_m'])
    valid &= (np.abs(x) < half) & (np.abs(y) < half)
    xx = ((x[valid] + half) / (2*half) * size).astype(int)
    yy = ((y[valid] + half) / (2*half) * size).astype(int)
    indices = yy*size + xx
    count = np.bincount(indices, minlength=size*size)
    height = np.bincount(indices, weights=z[valid], minlength=size*size) / np.maximum(count, 1)
    rgb = np.stack([np.bincount(indices, weights=obs['rgb'][..., c][valid]/255., minlength=size*size)
                    / np.maximum(count, 1) for c in range(3)], -1)
    height[count == 0] = np.nan
    rgb[count == 0] = .94
    return {'rgb': rgb.reshape(size, size, 3), 'height': height.reshape(size, size),
            'valid_fraction': float(np.mean(count > 0)), 'extent': [-half, half, -half, half],
            'scope': 'Visualization from fixed measured RGB-D hit points; bins with no observed surface are blank. Occluded ground is not inferred.'}


def decision_candidates(decision, protocol, anchor_pose, goal, code_root):
    """Prefer the actual recorded proposals; legacy fresh-family replay is explicit."""
    from nedm.traverse.fdm_diverse_planner import check_reference_contract, propose_route_families
    if 'candidate_references' in decision:
        families = decision['candidate_references']
        if not isinstance(families, list) or not families:
            raise ValueError('Recorded candidate list is empty or malformed')
        source = 'Actual candidate_references saved by the online decision'
        helper = None
        if protocol.get('candidate_policy'):
            helper_path = 'src/nedm/traverse/fdm_online_candidates.py'
            require_hash(code_root/helper_path, protocol['source_sha256'][helper_path], 'Candidate policy source')
            from nedm.traverse.fdm_online_candidates import reference_fingerprint
            helper = reference_fingerprint
        for route in families:
            check_reference_contract(route)
            xy = np.asarray(route['waypoints'], float); speed = np.asarray(route['speeds'], float)
            if speed.shape != (len(xy),) or not np.isfinite(speed).all():
                raise ValueError('Recorded candidate speeds are malformed')
            if np.linalg.norm(xy[-1]-goal) > .25:
                raise ValueError('Recorded candidate endpoint differs from the supplied goal')
            if helper is not None and helper(route) != route.get('meta', {}).get('reference_sha256'):
                raise ValueError('Recorded candidate geometry/speed fingerprint disagrees')
        return families, source
    if protocol.get('candidate_policy'):
        raise ValueError('Retention-policy run is missing recorded candidates; cannot substitute freshly generated routes')
    return propose_route_families(anchor_pose, goal, speeds=tuple(protocol['candidate_speeds_mps']),
        offsets=tuple(protocol['candidate_offsets_m']), step_m=.5), 'Legacy fresh-family policy reconstructed from exact executed source and recorded context'


def future_reference_xy(route):
    """Display the causally unexecuted suffix of retained references, never reroute."""
    xy = np.asarray(route['waypoints'], float); stations = np.asarray(route['stations'], float)
    station = float(route.get('meta', {}).get('fdm_station', stations[0]))
    first = min(int(np.searchsorted(stations, station, side='left')), len(xy)-1)
    return xy[first:]


def load_run(label, folder, observation_path, obs, args):
    from nedm.traverse.fdm_data import build_history
    from nedm.traverse.fdm_diverse_planner import propose_route_families
    protocol = read_json(folder/'online_protocol.json')
    outcome = read_json(folder/'outcome.json')
    require_hash(observation_path, protocol['scene_observation_sha256'], 'Observation')
    equality = read_json(folder/'anchor_equality.json')
    if not equality.get('matched'):
        raise ValueError('Saved physical frame-zero equality did not pass')
    trajectory = load_npz(folder/'trajectory.npz')
    rich = load_npz(folder/'rich_telemetry.npz')
    intervals = load_npz(folder/'rich_intervals.npz')
    n = len(trajectory['state'])
    if trajectory['pose'].shape != (n, 3) or trajectory['action'].shape != (n, 3):
        raise ValueError('Core physical trajectory arrays disagree')
    if len(rich['time_s']) != n+1 or len(intervals['duration_s']) != n:
        raise ValueError('Rich sample / interval arrays do not align with physical trajectory')
    if not np.all(np.diff(rich['time_s']) > 0.) or not np.isfinite(rich['time_s']).all():
        raise ValueError('Actual sample timestamps must be finite and strictly increasing')
    decisions = sorted((folder/'decisions').glob('decision_*.json'))
    plan_once = protocol.get('planning_mode', 'receding') == 'plan_once'
    if plan_once and (len(decisions) != 1 or outcome.get('planning_decisions') != 1):
        raise ValueError('Plan-once report requires exactly one recorded decision')
    if args.decision_index >= len(decisions):
        raise ValueError(f'{label}: requested decision {args.decision_index}, only {len(decisions)} exist')
    decision_path = decisions[args.decision_index]
    decision = read_json(decision_path)
    frame = int(decision['frame'])
    if not 0 <= frame < n:
        raise ValueError('Decision anchor is not a measured physical state')
    # This prefix is the entire permissible state/action history. Even though
    # the report contains later physical measurements, they never enter replay.
    history = build_history(trajectory['state'][:frame+1], trajectory['action'][:frame+1],
                            trajectory['pose'][:frame+1], frame)
    history_hash = hashlib.sha256(history.tobytes()).hexdigest()
    if history_hash != decision['history_sha256']:
        raise ValueError('Reconstructed causal history SHA disagrees with the executed decision')
    if not np.array_equal(np.asarray(decision['anchor_pose']), trajectory['pose'][frame]):
        raise ValueError('Decision anchor pose disagrees with measured trajectory')
    if not np.array_equal(np.asarray(decision['anchor_state17'], np.float32), trajectory['state'][frame]):
        raise ValueError('Decision anchor state disagrees with measured trajectory')
    if not np.isclose(decision['time_s'], rich['time_s'][frame], atol=1e-6, rtol=0.):
        raise ValueError('Decision recording time disagrees with actual telemetry')
    expected_source = protocol.get('source_sha256', {})
    planner_source = 'src/nedm/traverse/fdm_diverse_planner.py'
    require_hash(args.code_root/planner_source, expected_source[planner_source], 'Candidate generator source')
    families, candidate_source = decision_candidates(decision, protocol, trajectory['pose'][frame], obs['goal_xy'], args.code_root)
    # All proposals are retained, including rejected ones. Different cruise
    # speeds on the same geometry are counted but drawn as one spatial line.
    selected = decision['decision'].get('route')
    if plan_once and selected is not None:
        if frame != 0 or any(not np.array_equal(np.asarray(selected[key]),
                np.asarray(outcome['final_reference'][key])) for key in ('waypoints', 'speeds')):
            raise ValueError('Plan-once execution did not retain the launch reference')
    forecast = None
    replay = {'available': False, 'reason': 'No checkpoint supplied' if args.checkpoint is None else 'Planner abstained; no selected reference'}
    if args.checkpoint is not None:
        require_hash(args.checkpoint, protocol['model_checkpoint_sha256'], 'Checkpoint')
        for source in ('src/nedm/traverse/fdm_diverse_model.py', 'src/nedm/traverse/fdm_diverse_data.py'):
            require_hash(args.code_root/source, expected_source[source], 'Forecast source')
        if selected is not None:
            import torch
            from nedm.traverse.fdm_diverse_model import load_rgbd_checkpoint
            from nedm.traverse.fdm_diverse_planner import RGBDReferenceScorer
            torch.set_num_threads(args.cpu_threads)
            model, checkpoint = load_rgbd_checkpoint(args.checkpoint, device='cpu')
            if int(checkpoint['step']) != int(protocol['checkpoint_step']):
                raise ValueError('Checkpoint step disagrees with executed protocol')
            scorer = RGBDReferenceScorer(model, obs['rgbd'], history, trajectory['pose'][frame], obs['goal_xy'],
                elapsed_s=float(decision['time_s']), control=protocol['image_intervention'])
            output = scorer.predict([selected])
            local = output['trajectory'][0, :, :2]
            pose = trajectory['pose'][frame]; c, s = np.cos(pose[2]), np.sin(pose[2])
            world = pose[:2] + np.stack((c*local[:, 0]-s*local[:, 1], s*local[:, 0]+c*local[:, 1]), -1)
            times = float(decision['time_s']) + (np.arange(len(world))+1)*float(model.config.dt)
            forecast = {'world_xy': world, 'time_s': times, 'positive_work_kj': output['work'][0, :, 0],
                        'attitude_rad': output['attitude'][0], 'event_probability': output['event_probability'][0]}
            replay = {'available': True, 'device': 'cpu', 'checkpoint_sha256': sha(args.checkpoint),
                      'checkpoint_step': checkpoint['step'], 'history_sha256': history_hash,
                      'history_prefix_last_frame': frame, 'forecast_dt_s': model.config.dt,
                      'forecast_horizon_s': len(world)*model.config.dt,
                      'scope': ('Launch forecast for the same full reference executed by native PID; one decision and unchanged final reference verified. Only the finite forecast interval can be compared with its measured rollout.'
                          if plan_once else 'Selected-route conditional forecast recomputed from the recorded causal prefix. Subsequent replans can change the actual continuation; this is not a fixed-reference counterfactual validation.')}
    return {'label': label, 'folder': folder, 'protocol': protocol, 'outcome': outcome, 'trajectory': trajectory,
            'rich': rich, 'intervals': intervals, 'decision': decision, 'decision_path': decision_path,
            'source_process_result': read_json(folder.parent/(folder.name+'.result.json')) if (folder.parent/(folder.name+'.result.json')).exists() else None,
            'frame': frame, 'families': families, 'candidate_source': candidate_source, 'selected': selected, 'forecast': forecast, 'replay': replay,
            'checks': {'observation_sha256': sha(observation_path), 'anchor_equality_passed': True,
                       'history_sha256': history_hash, 'decision_sha256': sha(decision_path),
                       'trajectory_sha256': sha(folder/'trajectory.npz')}}


def draw_routes(ax, run, obs, image_map, *, zoom=False):
    ax.imshow(image_map['rgb'], origin='lower', extent=image_map['extent'], alpha=.67)
    seen = set()
    for route in run['families']:
        xy = future_reference_xy(route)
        fingerprint = np.asarray(xy, np.float32).tobytes()
        if fingerprint in seen:
            continue
        ax.plot(xy[:, 0], xy[:, 1], color='#65666a', lw=1.0, alpha=.58,
                label='Proposed reference geometries' if not seen else None)
        seen.add(fingerprint)
    if run['selected'] is not None:
        xy = future_reference_xy(run['selected'])
        ax.plot(xy[:, 0], xy[:, 1], color='#e78b17', lw=2.5, label='Selected reference (command)')
    poses = np.vstack([run['trajectory']['pose'], run['trajectory']['terminal_pose']])
    frame = run['frame']
    ax.plot(poses[:, 0], poses[:, 1], color='#1674bf', lw=2.3, label='Actual Chrono traversal')
    if frame:
        ax.plot(poses[:frame+1, 0], poses[:frame+1, 1], color='#073654', lw=2.3, label='Measured prefix')
    predicted = run['forecast']
    if predicted is not None:
        points = np.vstack([poses[frame, :2], predicted['world_xy']])
        ax.plot(points[:, 0], points[:, 1], '--', color='#d72665', lw=2.8,
                label=f"FDM forecast ({run['replay']['forecast_horizon_s']:g} s)")
    ax.scatter(*poses[frame, :2], s=90, marker='o', c='white', edgecolors='black', zorder=8, label='Decision anchor')
    ax.scatter(*obs['goal_xy'], s=150, marker='*', c='#15a474', edgecolors='white', zorder=8, label='Supplied goal')
    ax.set_aspect('equal'); ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)')
    if zoom:
        points = [poses[frame, :2]]
        until = run['decision']['time_s'] + (run['replay'].get('forecast_horizon_s', 12.))
        mask = (run['rich']['time_s'] >= run['decision']['time_s']) & (run['rich']['time_s'] <= until)
        points.extend(poses[mask, :2])
        if predicted is not None:
            points.extend(predicted['world_xy'])
        points = np.asarray(points); center = (points.max(0)+points.min(0))/2
        radius = max(8., np.ptp(points, axis=0).max()/2+4.)
        ax.set_xlim(center[0]-radius, center[0]+radius); ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_title('Finite forecast and actual continuation')
    else:
        half = image_map['extent'][1]; ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        ax.set_title(f"{len(run['families'])} proposals / {len(seen)} geometries; no outcome filtering")
    ax.grid(alpha=.15)


def overview(run, obs, image_map, out):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 7.8), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=(1., .14), width_ratios=(1., 1.25, 1.))
    raw = fig.add_subplot(grid[0, 0]); raw.imshow(obs['rgb']); raw.set_axis_off()
    raw.set_title('Original pre-drive RGB observation\nRGB-D is fixed throughout this run')
    world = fig.add_subplot(grid[0, 1]); draw_routes(world, run, obs, image_map)
    zoom = fig.add_subplot(grid[0, 2]); draw_routes(zoom, run, obs, image_map, zoom=True)
    handles, labels = world.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=10, frameon=False)
    outcome = run['outcome']; note = fig.add_subplot(grid[1, :]); note.set_axis_off()
    note.text(0, .9, f"{run['label']} ({run['protocol']['cost_mode']})  |  decision t = {run['decision']['time_s']:.2f} s  |  measured {outcome['elapsed_s']:.2f} s  |  "
        f"goal progress {outcome['goal_progress_m']:.1f} m  |  positive mechanical work {outcome['positive_work_kj']:.1f} kJ  |  status: {outcome['status']}", fontsize=12)
    continuation = ('One launch decision; PID follows the same full reference.'
        if run['protocol'].get('planning_mode') == 'plan_once' else 'Actual continuation includes later replans.')
    note.text(0, .45, 'Measured-depth map. Reference lines are commands; dashed magenta is the finite forecast; blue is actual physics. ' + continuation, fontsize=10)
    if run['selected'] is None:
        note.text(0, 0., 'This decision abstained: no selected reference or forecast is shown.', color='#a42222', fontsize=11)
    elif run['source_process_result'] is not None and run['source_process_result'].get('exit_code') != 0:
        note.text(0, 0., f"Source process exited {run['source_process_result']['exit_code']}; displayed physical artifacts are independently verified. This does not relabel the job as successful.", fontsize=10, color='#914a20')
    elif run['forecast'] is None:
        note.text(0, 0., 'Forecast unavailable: supply the exact executed checkpoint for verified CPU replay.', fontsize=10)
    fig.savefig(out/'overview.png', dpi=180, facecolor='white'); fig.savefig(out/'overview.pdf', facecolor='white')
    plt.close(fig)


def candidate_scores(run, out):
    """Report saved scores; do not rerank, omit risky candidates, or use outcomes."""
    import matplotlib.pyplot as plt
    rows = run['decision']['decision'].get('family_scores', [])
    if not rows:
        return {'available': False, 'reason': 'Decision contains no scored kinematically valid families'}
    indices = np.array([row['family_index'] for row in rows])
    selected = run['decision']['decision'].get('selected_family_index')
    colors = ['#e78b17' if i == selected else '#80919f' for i in indices]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained')
    for ax, field, title, unit in (
        (axes[0], 'estimated_time_to_goal_s', 'Extrapolated time-to-goal cost', 'Seconds'),
        (axes[1], 'estimated_route_work_kj', 'Extrapolated positive-work cost', 'Mechanical work (kJ)'),
    ):
        values = [np.nan if row.get(field) is None else row[field] for row in rows]
        ax.bar(indices, values, color=colors); ax.set_title(title); ax.set_ylabel(unit)
    x = np.arange(len(rows))
    for offset, field, color, label in ((-.2, 'contact_probability', '#a64564', 'Contact'),
                                      (.2, 'low_progress_probability', '#547c42', 'Stall / low progress')):
        values = [np.nan if row.get(field) is None else row[field] for row in rows]
        axes[2].bar(x+offset, values, width=.4, color=color, label=label)
    axes[2].set_xticks(x, indices); axes[2].set_ylim(0., 1.); axes[2].legend(fontsize=9)
    axes[2].set_title('Finite-horizon predicted risks'); axes[2].set_ylabel('Probability')
    for ax in axes:
        ax.set_xlabel('Original geometric-family / speed proposal index'); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Saved pre-refinement family scores | orange: family supplying selected reference | no physical outcome filtering', fontsize=12)
    fig.savefig(out/'candidate_scores.png', dpi=170); plt.close(fig)
    return {'available': True, 'scored_rows': len(rows), 'selected_family_index': selected,
            'scope': 'Saved base-family scores before MPPI refinement. Selected refined route may have a different cost; no unrecorded candidate forecasts are invented.'}


def telemetry(runs, obs, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), layout='constrained', sharex=True)
    colors = plt.get_cmap('tab10').colors
    for index, run in enumerate(runs):
        color = colors[index % len(colors)]; label = run['label']; rich = run['rich']; t = rich['time_s']
        pose = np.vstack([run['trajectory']['pose'], run['trajectory']['terminal_pose']])
        distance = np.linalg.norm(pose[:, :2]-obs['goal_xy'], axis=1)
        axes[0, 0].plot(t, distance[0]-distance, color=color, label=label)
        axes[0, 1].plot(t, rich['vel_body_x_mps'], color=color, label=label)
        work = np.r_[0., np.cumsum(run['intervals']['engine_interface_positive_work_kj'])]
        axes[1, 0].plot(t, work, color=color, label=label)
        axes[1, 1].plot(t, rich['engine_interface_power_kw'], color=color, label=label)
        slips = np.column_stack([rich[k] for k in ('tire_fl_longitudinal_slip', 'tire_fr_longitudinal_slip',
                                                  'tire_rl_longitudinal_slip', 'tire_rr_longitudinal_slip')])
        absolute_slip = np.max(np.abs(slips), axis=1)
        axes[2, 0].plot(t, absolute_slip, color=color, label=label)
        peak = int(np.argmax(absolute_slip))
        axes[2, 0].annotate(f'{absolute_slip[peak]:,.4g}', (t[peak], absolute_slip[peak]),
                            xytext=(-10, 10), textcoords='offset points', ha='right', fontsize=8, color=color)
        axes[2, 1].plot(t, np.degrees(rich['roll_rad']), color=color, label=f'{label}: roll')
        axes[2, 1].plot(t, np.degrees(rich['pitch_rad']), color=color, ls='--', label=f'{label}: pitch')
        for ax in axes.flat:
            ax.axvline(run['decision']['time_s'], color=color, ls=':', lw=.8, alpha=.7)
        prediction = run['forecast']
        if prediction is not None:
            axes[1, 0].plot(prediction['time_s'], work[run['frame']]+prediction['positive_work_kj'],
                            color=color, ls='--', alpha=.7, label=f'{label}: selected-route forecast')
    titles = [('Goal progress', 'Progress (m)'), ('Measured forward body velocity', 'Velocity (m/s)'),
              ('Positive engine-interface work', 'Mechanical work (kJ)'), ('Engine-interface mechanical power', 'Power (kW)'),
              ('Measured maximum absolute tire slip', 'Native longitudinal slip (1)'), ('Measured roll and pitch', 'Angle (degrees)')]
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(alpha=.2); ax.legend(fontsize=8, loc='best')
    axes[2, 0].set_yscale('symlog', linthresh=.1)
    axes[2, 0].set_ylim(bottom=0.)
    axes[2, 0].set_title('Native tire slip: full range retained (symlog above 0.1)')
    for ax in axes[-1]:
        ax.set_xlabel('Actual recording time (s; settlement excluded)')
    fig.suptitle('Measured Chrono telemetry | dotted vertical line: displayed decision | dashed work: conditional FDM replay', fontsize=13)
    fig.savefig(out/'telemetry.png', dpi=170, facecolor='white'); fig.savefig(out/'telemetry.pdf', facecolor='white'); plt.close(fig)


def slip_diagnostics(runs, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
    values = []
    for index, run in enumerate(runs):
        rich = run['rich']; t = rich['time_s']; color = plt.get_cmap('tab10')(index)
        native = np.column_stack([rich[f'tire_{wheel}_longitudinal_slip'] for wheel in ('fl', 'fr', 'rl', 'rr')])
        speed = np.column_stack([rich[f'tire_{wheel}_horizontal_slip_speed_mps'] for wheel in ('fl', 'fr', 'rl', 'rr')])
        maximum = np.max(np.abs(native), axis=1); peak = int(np.argmax(maximum))
        axes[0].plot(t, maximum, color=color, label=run['label'])
        axes[0].annotate(f"Peak {maximum[peak]:,.6g} at {t[peak]:.2f} s", (t[peak], maximum[peak]),
                         xytext=(15 if t[peak] < 1. else -15, -18), textcoords='offset points',
                         ha='left' if t[peak] < 1. else 'right', va='top', fontsize=9, color=color,
                         bbox={'facecolor':'white', 'alpha':.85, 'edgecolor':'none', 'pad':2})
        for wheel, values_per_wheel in zip(('FL', 'FR', 'RL', 'RR'), speed.T):
            axes[1].plot(t, values_per_wheel, color=color, lw=1., alpha=.65,
                         label=f"{run['label']}: four wheels" if wheel=='FL' else None)
        values.append({'run':run['label'], 'native_absolute_slip_peak':float(maximum[peak]),
                       'peak_recording_time_s':float(t[peak]),
                       'body_forward_velocity_at_peak_mps':float(rich['vel_body_x_mps'][peak]),
                       'maximum_absolute_horizontal_slip_speed_mps':float(np.max(np.abs(speed)))})
    axes[0].set_yscale('symlog', linthresh=.1); axes[0].set_ylim(bottom=0.)
    axes[0].set_ylabel('Native absolute slip (dimensionless)')
    axes[0].set_title('Full native ratio retained: sensitive to a near-zero velocity denominator')
    axes[1].set_ylabel('Horizontal slip speed (m/s)'); axes[1].set_xlabel('Actual recording time (s)')
    axes[1].set_title('Circumferential tire speed minus horizontal wheel-hub speed: derived approximation on slopes')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('Slip ratio and slip speed are different quantities; no numerical clipping or removed extremes')
    fig.savefig(out/'slip_diagnostics.png', dpi=170); plt.close(fig)
    dump(out/'slip_diagnostics.json', {'measured':values, 'scope':'Native tire ratio and separately derived horizontal slip speed; raw arrays unchanged'})


def encode_video(run, out, ffmpeg, ffprobe):
    metadata_path = run['folder']/'frame_metadata.json'
    if not metadata_path.exists():
        return {'available': False, 'reason': 'No actual PNG frame timestamp metadata saved'}
    metadata = read_json(metadata_path); rows = metadata['frames']
    if len(rows) < 2:
        return {'available': False, 'reason': 'At least two physical frame timestamps are required'}
    times = np.asarray([r['simulation_time_s'] for r in rows], float)
    recording = np.asarray([r['recording_time_s'] for r in rows], float)
    if not np.all(np.diff(times) > 0.) or not np.allclose(times-times[0], recording-recording[0], atol=1e-6, rtol=0):
        raise ValueError('Physical video timestamps are nonmonotone or inconsistent')
    poses = np.vstack([run['trajectory']['pose'], run['trajectory']['terminal_pose']])
    for row in rows:
        if not np.allclose(row['actual_pose'], poses[int(row['telemetry_frame'])], atol=1e-8, rtol=0):
            raise ValueError('Video frame pose differs from the physical telemetry anchor')
    if not rows[-1]['terminal']:
        raise ValueError('Actual video has no measured terminal endpoint')
    if not shutil.which(ffmpeg) or not shutil.which(ffprobe):
        raise RuntimeError('ffmpeg and ffprobe are required for timestamp-aware encoding')
    # Each original PNG appears in sequence. Per-frame durations preserve the
    # actual physical timestamps; the final still is held for one millisecond.
    # A high image input timebase avoids the concat demuxer 25fps quantization.
    lines = ['ffconcat version 1.0']; frame_hashes = []
    for index, row in enumerate(rows):
        path = (run['folder']/row['file']).resolve()
        if not path.is_relative_to(run['folder']) or not path.is_file():
            raise ValueError('Frame path must be an existing file within the run directory')
        escaped = str(path).replace("'", "'\\''")
        duration = times[index+1]-times[index] if index+1 < len(times) else .001
        lines += [f"file '{escaped}'", 'option framerate 1000', f'duration {duration:.9f}']
        frame_hashes.append({'file': row['file'], 'sha256': sha(path), 'simulation_time_s': times[index]})
    concat = out/'actual_frames.ffconcat'; concat.write_text('\n'.join(lines)+'\n')
    target = out/'actual_chrono.mp4'
    command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-n', '-safe', '0', '-f', 'concat', '-i', str(concat),
               '-fps_mode', 'vfr', '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p', '-video_track_timescale', '1000', str(target)]
    subprocess.run(command, check=True)
    command_probe = [ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                     'frame=best_effort_timestamp_time:format=duration', '-of', 'json', str(target)]
    probe = json.loads(subprocess.run(command_probe, check=True, capture_output=True, text=True).stdout)
    encoded_times = np.asarray([float(r['best_effort_timestamp_time']) for r in probe['frames']])
    if len(encoded_times) != len(times) or np.max(np.abs(encoded_times-(times-times[0]))) > .002:
        raise ValueError('Encoded video frame count/timestamps fail physical timing verification')
    result = {'available': True, 'file': str(target), 'sha256': sha(target), 'frames': len(rows),
              'physical_duration_s': float(times[-1]-times[0]), 'container_duration_s': float(probe['format']['duration']),
              'maximum_timestamp_error_s': float(np.max(np.abs(encoded_times-(times-times[0])))),
              'terminal_hold_s': .001, 'pixels': 'Original actual Chrono PNGs; no invented frames, motion interpolation or predicted vehicle rendering',
              'command': command, 'frame_sources': frame_hashes}
    dump(out/'video_provenance.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='LABEL=ONLINE_OUTPUT; repeat for matched runs')
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--code-root', type=Path, default=ROOT, help='Exact online source snapshot for source-verified replay')
    parser.add_argument('--decision-index', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--encode-video', action='store_true')
    parser.add_argument('--ffmpeg', default='ffmpeg'); parser.add_argument('--ffprobe', default='ffprobe')
    args = parser.parse_args()
    if args.decision_index < 0 or args.cpu_threads < 1:
        parser.error('Decision index must be nonnegative and CPU thread count positive')
    args.code_root = args.code_root.resolve(); sys.path.insert(0, str(args.code_root/'src'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rcParams
    rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use a new empty report directory; preserve earlier artifacts')
    args.out.mkdir(parents=True, exist_ok=True)
    observation_path = args.observation/'observation.npz' if args.observation.is_dir() else args.observation
    observation_path = observation_path.resolve(); obs = load_npz(observation_path)
    camera = read_json(observation_path.with_suffix('.json'))['camera']
    runs = [load_run(label, folder, observation_path, obs, args) for label, folder in map(parse_run, args.run)]
    if len(set(slug(run['label']) for run in runs)) != len(runs):
        raise ValueError('Run labels must have distinct output names')
    half = max(float(run['protocol']['mppi_config']['arena_half_extent_m']) for run in runs)
    image_map = observed_map(obs, camera, half)
    entries = []
    for run in runs:
        out = args.out/slug(run['label']); out.mkdir()
        overview(run, obs, image_map, out)
        score_report = candidate_scores(run, out)
        if run['forecast'] is not None:
            np.savez_compressed(out/'selected_forecast_replay.npz', **run['forecast'])
        video = encode_video(run, out, args.ffmpeg, args.ffprobe) if args.encode_video else {'available': False, 'reason': 'Encoding not requested'}
        entry = {'label': run['label'], 'run': run['folder'], 'outcome': run['outcome'], 'checks': run['checks'],
                 'forecast_replay': run['replay'], 'candidate_source': run['candidate_source'], 'source_process_result': run['source_process_result'], 'saved_candidate_scores': score_report, 'video': video}
        dump(out/'report.json', entry); entries.append(entry)
    telemetry(runs, obs, args.out)
    slip_diagnostics(runs, args.out)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
    im = ax.imshow(image_map['height'], origin='lower', extent=image_map['extent'], cmap='terrain')
    ax.set_aspect('equal'); ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)')
    ax.set_title('Visible surface elevation from the fixed measured depth image')
    fig.colorbar(im, ax=ax, label='Observed elevation (m)')
    fig.savefig(args.out/'observed_elevation.png', dpi=170); plt.close(fig)
    dump(args.out/'report_manifest.json', {'runs': entries, 'script_sha256': sha(__file__),
         'observation_sha256': sha(observation_path), 'camera': camera,
         'observed_map': {k:v for k,v in image_map.items() if k not in ('rgb','height')},
         'boundary': 'No authored terrain/hazard files opened. Forecasts use the fixed sensor image and verified causal history only. Later physical measurements are used exclusively for reporting.'})
    print(json.dumps({'out': str(args.out.resolve()), 'runs': len(runs), 'forecast_replays': sum(run['forecast'] is not None for run in runs)}))


if __name__ == '__main__':
    main()
