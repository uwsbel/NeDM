from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent;source=BASE/'patch_geometry_v1';out=BASE/'patch_panels_v2';out.mkdir(exist_ok=False)
report=json.loads((source/'patch_geometry.json').read_text())
for name,result in report['trials'].items():
 with np.load(source/(name+'_patches.npz')) as z:patches=z['patches'];times=z['times_s']
 chosen=[0,.45,.5,2,4] if 'rough' in name and name.endswith('energy') else ([0,2,4,10,12] if 'rough' in name else [0,10,11.45,12,20])
 fig,axes=plt.subplots(2,5,figsize=(13,5.7),constrained_layout=True)
 for col,t in enumerate(chosen):
  i=int(abs(times-t).argmin());a=result['anchors'][i];j=a['nearest_rock_patch_index'];rgb=np.clip((patches[i,j,:3].transpose(1,2,0)+1)/2,0,1);elevation=patches[i,j,3]*40
  axes[0,col].imshow(rgb,interpolation='nearest');im=axes[1,col].imshow(elevation,cmap='terrain',interpolation='nearest');uv=a['rock_center_patch_pixel_uv']
  for row in range(2):
   ax=axes[row,col]
   if a['rock_center_inside_patch']:ax.plot(*uv,'+',color='red',ms=8)
   ax.set_xlim(-.5,15.5);ax.set_ylim(15.5,-.5);ax.set_xticks([0,7.5,15],labels=['+4','0','−4']);ax.set_yticks([0,7.5,15],labels=['+4','0','−4'])
  title=f'Anchor {times[i]:g} s / command +{(j+1)*.2:g} s'
  if not a['rock_center_inside_patch']:title+=f'\nRock centre outside: {a["rock_distance_to_patch_center_m"]:.1f} m'
  axes[0,col].set_title(title,fontsize=9);fig.colorbar(im,ax=axes[1,col],fraction=.05,label='elevation m')
 fig.suptitle(name.replace('diverse_v1_test_','')+'\nExact command crops: 8 m × 8 m, 16² pixels; top = forward, left = vehicle-left. Red cross = rock centre when inside.',fontsize=11)
 fig.savefig(out/(name+'_patches.png'),dpi=180);plt.close(fig)
