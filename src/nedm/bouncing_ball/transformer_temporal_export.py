"""Curate a reproducible experiment bundle without copying training datasets."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bundle-name',default='review_bundle');args=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    assert Path(args.bundle_name).name==args.bundle_name
    root=args.root;out=root/('export_'+args.bundle_name);out.mkdir(exist_ok=False)
    selection=json.loads((root/'selection.json').read_text())
    report=json.loads((root/'certification/paired_results.json').read_text())
    def copy(source,destination):
        destination=out/destination;destination.parent.mkdir(parents=True,exist_ok=True)
        if source.is_dir():shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else:shutil.copyfile(source,destination)
    for name in ('selection.json','summary.json','comparison.csv','comparison.png','candidate_names.txt'):
        copy(root/name,Path(name))
    for name in ('frozen','certification','targeting','logs'):
        copy(root/name,Path(name))
    for candidate in selection['candidates']:
        name=candidate['name'];folder=root/'runs'/name
        for filename in ('run_config.json','complete.json','validation.json','gradient_check.json',
                         'free_core_validation.json','train_log.jsonl','free_core.pt'):
            source=folder/filename
            if source.exists():copy(source,Path('runs')/name/filename)
    experiments=root.parent
    for name,path in (
        ('prior_fullstate_117mm',experiments/'ball_transformer_contact_20261001T192800Z/frozen/transformer_state5_binary_refined/best.pt'),
        ('historical_transformer_v2',experiments/'ball_span_v1_20260930/runs/nrd_v2_certified/best.pt'),
        ('analytical_mlp_reference',experiments/'ball_precision_20261001/runs/certified_v2/best.pt')):
        assert digest(path)==report['models'][name]['checkpoint_sha256']
        copy(path,Path('references')/name/'best.pt')
    snapshots=sorted(root.glob('code_v*'),key=lambda p:int(p.name.removeprefix('code_v')))
    for source in snapshots:
        copy(source,Path('source_snapshots')/source.name)
    for label,path in (
        ('training',experiments/'ball_span_v1_20260930/final_data'),
        ('fresh_test',root/'fresh_test'),('certification',root/'certification_data')):
        copy(path/'campaign_index.json',Path('datasets')/label/'campaign_index.json')
    (out/'README.txt').write_text(
        'All 19 candidates were frozen on validation before the 900 fresh Chrono episodes.\n'
        'Native model step:10ms. Reported trajectory RMSE is not a pointwise/continuous-time bound.\n'
        'selection.json retains the preregistered primary; summary.json includes tail and scalar-control results.\n'
        'frozen/ contains exact evaluated checkpoints; references/ contains paired old baselines.\n'
        'source_snapshots/ preserves training, certification and report code versions.\n'
        'runs/*/run_config.json records training/config/source/dataset hashes and AMD job IDs.\n'
        'datasets/ records remote source paths/hashes; large collection datasets remain on AMD.\n'
        'targeting/ contains model-only gradient histories and independent physical Chrono replays.\n'
        f'Use nedm.bouncing_ball.model.load_model(checkpoint,device) with source_snapshots/{snapshots[-1].name}/src.\n'
        'Heavy work and verification ran only on AMD compute; no files were published or committed.\n')
    manifest={str(path.relative_to(out)):dict(bytes=path.stat().st_size,sha256=digest(path))
              for path in sorted(out.rglob('*')) if path.is_file()}
    (out/'manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
    archive=root/(args.bundle_name+'.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=2) as tar:tar.add(out,arcname=args.bundle_name)
    archive_hash=digest(archive)
    (root/(args.bundle_name+'.sha256')).write_text(archive_hash+'  '+archive.name+'\n')
    print(json.dumps(dict(complete=True,archive=str(archive),sha256=archive_hash,
                         files=len(manifest),bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__':main()
