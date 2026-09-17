"""Static scientific plots from the complete audited protected report only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

root=Path(__file__).resolve().parent
report_path=root.parent/'online_protected_test_v1_cohort_01/cohort_report.json'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
freeze_sha='29684ce59cdd68e9b1f7310dcd2e63c55d021f6ff6712b83f99d286722efeb33'
report=json.loads(report_path.read_text())
assert report['audit']['freeze_sha256']==freeze_sha and not report['audit']['errors']
rows=report['all_trials'];assert len(rows)==30 and all(row['artifact_valid'] for row in rows)
arms=[row['arm'] for row in report['unconditional_arms']]
scenes=sorted({row['scene_id'] for row in rows})
lookup={(row['scene_id'],row['arm']):row for row in rows}
labels=['RGB-D\ntime','RGB-D\nenergy','Blank\ntime','Blank\nenergy','Original best\nreceding / time']
matrix=np.array([[lookup[scene,arm]['verified_schema_safe_goal'] for arm in arms] for scene in scenes],int)
fig=plt.figure(figsize=(14,10),layout='constrained')
grid=fig.add_gridspec(2,1,height_ratios=[3.5,1.3])
ax=fig.add_subplot(grid[0])
ax.imshow(matrix,cmap=ListedColormap(['#f2c9c2','#c4e5d7']),vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(5),[label+f"\n{sum(matrix[:,i])}/6 safe" for i,label in enumerate(labels)])
ax.set_yticks(range(6),[scene.replace('diverse_v1_test_','').removesuffix('_00').replace('_',' ').title() for scene in scenes])
ax.set_xticks(np.arange(-.5,5,.5),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True)
ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False)
for i,scene in enumerate(scenes):
    for j,arm in enumerate(arms):
        row=lookup[scene,arm]
        if row['verified_schema_safe_goal']:
            label=f"Safe goal\n{row['elapsed_s']:.2f} s"
        else:
            events=[name for name,key in [('contact','schema_contact'),('stall','bounded_blockage_v1'),('rollover','schema_rollover')] if row[key]]
            label=f"Timeout: {row['elapsed_s']:.0f} s\n"+(', '.join(events) if events else 'no declared failure event')
        ax.text(j,i,label,ha='center',va='center',fontsize=9,color='#1f2933')
ax.set_title('Protected evaluation: all 30 frozen trials passed artifact audit\nGreen = full goal without declared contact, rollover or bounded stall',pad=14)
paired=report['paired_safe_tradeoffs']['selected_rgbd_energy_vs_time']
absolute=[]
for pair in paired:
    if not pair['eligible']:continue
    scene=pair['scene_id'];time_row=lookup[scene,'selected_rgbd_time'];energy_row=lookup[scene,'selected_rgbd_energy']
    absolute.append([scene.replace('diverse_v1_test_','').removesuffix('_00').replace('_',' ').title(),
        f"{time_row['elapsed_s']:.2f}",f"{energy_row['elapsed_s']:.2f}",f"{pair['time_increase_fraction']*100:+.3f}%",
        f"{time_row['positive_work_kj']:.3f}",f"{energy_row['positive_work_kj']:.3f}",f"{pair['work_saving_fraction']*100:+.3f}%"])
table_ax=fig.add_subplot(grid[1]);table_ax.axis('off')
table_ax.set_title('Selected RGB-D: absolute time/work and fractions for the four safe pairs',fontsize=12,pad=4)
table=table_ax.table(cellText=absolute,colLabels=['Scene','Time 0 (s)','Time 0.02 (s)','Time increase','Work 0 (kJ)','Work 0.02 (kJ)','Work saving'],
    cellLoc='center',colWidths=[.18,.125,.125,.13,.15,.15,.14],bbox=[0,.20,1,.73])
table.auto_set_font_size(False);table.set_fontsize(10)
for (r,c),cell in table.get_celld().items():
    cell.set_edgecolor('#dddddd')
    if r==0:cell.set_facecolor('#e5eef4')
    elif r%2:cell.set_facecolor('#f7f8fa')
table_ax.text(.5,.05,'Work is integrated positive engine-interface mechanical work. Failed runs remain above; no test-based parameter selection.',
    ha='center',va='center',transform=table_ax.transAxes,fontsize=10)
fig.savefig(root/'all_30_trials.png',dpi=190);fig.savefig(root/'all_30_trials.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
colors={'selected_rgbd_energy_vs_time':'#176eaa','matched_blank_energy_vs_time':'#777777'}
stats={}
for group,pairs in report['paired_safe_tradeoffs'].items():
    eligible=[row for row in pairs if row['eligible']]
    x=np.array([row['time_increase_fraction']*100 for row in eligible]);y=np.array([row['work_saving_fraction']*100 for row in eligible])
    label='RGB-D' if group.startswith('selected') else 'Blank'
    ax.scatter(x,y,s=58,color=colors[group],label=f'{label}: {len(eligible)} safe pairs')
    for row,xx,yy in zip(eligible,x,y):
        short=row['scene_id'].replace('diverse_v1_test_','').removesuffix('_00').replace('_',' ')
        offset=(6,6)
        if short=='valley network':offset=(6,-12)
        if short=='cross slopes':offset=(6,-12)
        ax.annotate(short,(xx,yy),xytext=offset,textcoords='offset points',fontsize=8,color=colors[group])
    stats[group]={'safe_pairs':len(eligible),'mean_work_saving_fraction':float(y.mean()/100),
                  'mean_time_increase_fraction':float(x.mean()/100)}
ax.axhline(0,color='#999999',linewidth=.8);ax.axvline(0,color='#999999',linewidth=.8)
ax.set_xlabel('Time increase with energy term (%)');ax.set_ylabel('Positive engine-interface work saving (%)')
ax.set_title('Energy coefficient 0.02 versus 0: descriptive safe-pair comparisons\nEvery failed pair remains in the all-trial matrix; no test-based selection')
ax.legend(loc='lower right');ax.margins(x=.2,y=.25)
fig.savefig(root/'paired_safe_energy.png',dpi=190);fig.savefig(root/'paired_safe_energy.pdf');plt.close(fig)
runtime_signatures={row['source_runtime_signature'] for row in rows}
assert len(runtime_signatures)==1
proof={'freeze_sha256':freeze_sha,'audited_report_sha256':sha(report_path),'script_sha256':sha(Path(__file__)),
       'all_30_trials_retained':True,'single_shared_source_runtime_signature':True,
       'source_runtime_signature_sha256':hashlib.sha256(next(iter(runtime_signatures)).encode()).hexdigest(),
       'paired_summary':stats,'training_controls_proof_sha256':sha(root.parent/'protected_matched_training_controls_v1.json'),
       'figures_sha256':{name:sha(root/name) for name in ('all_30_trials.png','all_30_trials.pdf','paired_safe_energy.png','paired_safe_energy.pdf')},
       'model_or_parameter_selection_performed':False,'simulation_or_training_performed':False}
(root/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
