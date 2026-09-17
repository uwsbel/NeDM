"""Train a config on ALL training groups (dev fold included) for deployment in the planner."""
import json, sys, os
import numpy as np
sys.path.insert(0, 'scripts')
from f104_night_train import Data, train
cfg = json.loads(sys.argv[1]); out = sys.argv[2]; seeds = [int(s) for s in sys.argv[3].split(',')]
os.makedirs(out, exist_ok=True)
D = Data(); d = D.d
full = (d['source'].astype(str) == 'designed') & (d['split'].astype(str) == 'train')
print(f'full-train routes: {int(full.sum())} (fit fold was {int(D.fit.sum())})', flush=True)
for s in seeds:
    c = dict(cfg, seed=s)
    train(c, D=D, fit_mask=full, save=f'{out}/{cfg["name"]}_full_s{s}.pt')
    print(f'  saved {cfg["name"]} seed {s}', flush=True)
