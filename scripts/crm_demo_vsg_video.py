"""Overlay (title, planner risk, time, speed, throttle, wheel slip, minimap) on VSG soil-particle frames -> mp4."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, 'src')
from nedm.traverse.terrain import TerrainMap

run, out_mp4 = Path(sys.argv[1]), sys.argv[2]
rows = json.load(open(run / 'overlay.json')); meta = json.load(open(run / 'title.json'))
route = json.load(open(run / 'reference.json')); case = json.load(open(run / 'case.json')); outcome = json.load(open(run / 'outcome.json'))
frames = sorted((run / 'frames').glob('img_*.png'))
n = min(len(frames), len(rows))
font = lambda s: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', s)
tm = TerrainMap.from_dir(Path('assets/traverse/arena_f104_50h_v1'))
h = np.flipud(tm.height_grid); g = ((h - h.min()) / (h.max() - h.min()) * 255).astype(np.uint8)
thumb = Image.blend(Image.fromarray(g).resize((220, 220)).convert('RGB'), Image.new('RGB', (220, 220), (150, 120, 80)), 0.35)
px = lambda x, y: (110 + x * 2.75, 110 - y * 2.75)
d0 = ImageDraw.Draw(thumb); d0.line([px(x, y) for x, y in route['waypoints']], fill=(20, 220, 240), width=2)
gx, gy = case['goal_xy']; c = px(gx, gy); d0.ellipse([c[0] - 7, c[1] - 7, c[0] + 7, c[1] + 7], outline=(240, 40, 200), width=2)
tmp = run / 'video_frames'; tmp.mkdir(exist_ok=True)
for f in tmp.glob('*.jpg'): f.unlink()
end = 'GOAL REACHED' if outcome['status'] == 'goal_reached' else 'BOGGED DOWN (goal not reached)'
track = []
for k in range(n):
    r = rows[k]; img = Image.open(frames[k]).convert('RGB'); W, H = img.size
    d = ImageDraw.Draw(img, 'RGBA'); d.rectangle([0, 0, W, 70], fill=(0, 0, 0, 165))
    d.text((14, 8), meta['title'], fill=(255, 255, 255), font=font(24))
    d.text((14, 40), f"planner risk {100 * meta['risk']:.1f} %    t = {r['t']:5.1f} s    speed {r['vx']:4.1f} m/s (commanded {r['cmd']:3.1f})    throttle {r['thr']:.2f}"
                     f"    worst wheel slip ratio {r['slip']:6.1f}", fill=(255, 235, 170), font=font(18))
    track.append(px(r['x'], r['y'])); mini = thumb.copy(); md = ImageDraw.Draw(mini)
    if len(track) > 1: md.line(track, fill=(255, 255, 255), width=2)
    x, y = track[-1]; md.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 60, 60))
    img.paste(mini, (W - 236, H - 236))
    if k >= n - 1:
        d.rectangle([0, H // 2 - 40, W, H // 2 + 40], fill=(0, 0, 0, 170)); d.text((W // 2 - 260, H // 2 - 20), f"{end} at {outcome['elapsed_s']:.1f} s", fill=(255, 255, 255), font=font(34))
    img.save(tmp / f'v_{k:05d}.jpg', quality=88)
for j in range(40):  # hold the last frame 2 s
    (tmp / f'v_{n + j:05d}.jpg').write_bytes((tmp / f'v_{n - 1:05d}.jpg').read_bytes())
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '20', '-i', str(tmp / 'v_%05d.jpg'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', out_mp4], check=True)
for f in tmp.glob('*.jpg'): f.unlink()
tmp.rmdir(); print(out_mp4, n, 'frames', outcome['status'], outcome['elapsed_s'])
