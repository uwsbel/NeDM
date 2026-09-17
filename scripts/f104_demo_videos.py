"""Render the 15 demo episodes with Blender and encode them with captions.

Each caption states what the model predicted BEFORE the run and what Chrono then did, so the two can be
compared without opening any json.
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
MAXF = 700            # 35 s: a stalled run stays stalled to the 120 s horizon; the clip says so
OUT = ROOT + '/night2_v1/demo_v1'
VID = 'artifacts/f104_demo_v1'
BLENDER = '/snap/bin/blender'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
GRADE = 'eq=contrast=1.22:saturation=1.45:gamma=0.97:brightness=-0.015,unsharp=5:5:0.4'
RGB = {'optimal': '0.05,0.80,0.25', 'suboptimal': '0.95,0.62,0.05', 'risky': '0.90,0.10,0.10'}
HEX = {'optimal': '#7CFC7C', 'suboptimal': '#FFC857', 'risky': '#FF6B6B'}
TITLE = {'optimal': 'WHAT THE PLANNER DRIVES', 'suboptimal': 'A MIDDLING ALTERNATIVE',
         'risky': 'WHAT THE PLANNER REJECTS'}
RANK = {'optimal': 'best of 256 candidates', 'risky': 'worst of 256 candidates'}


def outcome(m):
    """No colons: ffmpeg drawtext would read one as the start of the next option."""
    if m['fail']:
        what = ('STALLED, stopped at' if 'blockage' in m['status'] else
                'NEVER ARRIVED in' if m['status'] == 'timeout' else m['status'].replace('_', ' ').upper() + ' at')
        return f"{what} {m['elapsed']:.0f} s, {m['back_s']:.0f} s of it sliding backwards"
    slid = f", slid back {m['back_s']:.1f} s" if m['unsafe'] else ''
    return f"reached goal in {m['elapsed']:.1f} s{slid}, max tilt {m['max_tilt']:.0f} deg"


def render(m):
    rid = m['id']; d = f'{OUT}/blend/{rid}'
    png = f'{VID}/{rid}/png'
    if os.path.exists(png) and len(os.listdir(png)) > 10:
        return rid, 'cached'
    nf = min(json.load(open(d + '/blend/blender_export_summary.json'))['captured_frames'], MAXF)
    os.makedirs(png, exist_ok=True)
    cmd = [BLENDER, '-b', '--factory-startup', '--python', 'scripts/f104_demo_render_blender.py', '--',
           '--asset', os.path.abspath(d + '/blend/exported.assets.py'), '--frames', str(nf),
           '--out', os.path.abspath(png), '--res', '1280x720',
           '--route', os.path.abspath(f'{OUT}/ribbons/{rid}.npy'), '--color', RGB[m['role']],
           '--goal', ','.join(str(x) for x in m['goal'])]
    with open(f'{VID}/{rid}/render.log', 'w') as lg:
        subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, check=True)
    return rid, f'{len(os.listdir(png))} frames'


def encode(m):
    rid = m['id']; role = m['role']
    l1 = f"Scenario {m['scenario']}/5   {TITLE[role]}   ({RANK.get(role, 'no. %d of 256' % (m['rank'] + 1))})"
    l2 = f"predicted risk {100 * m['risk']:.2f} pct   ->   in Chrono  {outcome(m)}"
    vf = (f"{GRADE},drawbox=x=0:y=0:w=iw:h=104:color=black@0.55:t=fill,"
          f"drawtext=fontfile={FONT}:text='{l1}':x=26:y=22:fontsize=27:fontcolor=white,"
          f"drawtext=fontfile={FONT}:text='{l2}':x=26:y=61:fontsize=21:fontcolor={HEX[role]}")
    if m.get('trimmed'):
        vf += (f",drawtext=fontfile={FONT}:text='clip cut at 35 s - the run continued to the "
               f"{m['elapsed']:.0f} s stop':x=26:y=h-40:fontsize=20:fontcolor=white@0.75")
    out = f"{VID}/s{m['scenario']}_{'123'[('optimal', 'suboptimal', 'risky').index(role)]}_{role}.mp4"
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '20',
                    '-i', f'{VID}/{rid}/png/frame%04d.png', '-vf', vf,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                    '-movflags', '+faststart', out], check=True)
    return out


def refresh(meta):
    """Caption the episode that was actually filmed: re-label from the Blender re-run, not the screening run."""
    from f104_n2_analyze import labels
    for m in meta:
        d = f"{OUT}/blend/{m['id']}/run"
        if os.path.exists(d + '/outcome.json'):
            before = m['status'], round(m['elapsed'], 2)
            m.update(labels(d))
            m['reproduced'] = bool(before == (m['status'], round(m['elapsed'], 2)))
        nf = json.load(open(f"{OUT}/blend/{m['id']}/blend/blender_export_summary.json"))['captured_frames']
        m['trimmed'] = bool(nf > MAXF)
    json.dump(meta, open(OUT + '/video_meta.json', 'w'), indent=1)
    n = sum(m.get('reproduced', False) for m in meta)
    print(f'{n}/{len(meta)} filmed episodes reproduced the screening run exactly')
    return meta


def main():
    sys.path.insert(0, 'scripts')
    meta = refresh(json.load(open(OUT + '/video_meta.json')))
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only:
        meta = [m for m in meta if only in m['id']]
    os.makedirs(VID, exist_ok=True)
    for m in meta:
        os.makedirs(f"{VID}/{m['id']}", exist_ok=True)
    with ThreadPoolExecutor(3) as ex:
        for rid, st in ex.map(render, meta):
            print('rendered', rid, st, flush=True)
    for m in meta:
        print('wrote', encode(m), flush=True)


if __name__ == '__main__':
    main()
