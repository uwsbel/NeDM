#!/usr/bin/env python3
"""arena_gator_20260925 E5a (resumed): run scripts/ag_offline_auc.py with TF32 switched off for cuDNN and matmuls, as
scripts/ag_score_offline.py does by default. With the RTX 5090's default TF32 convolutions member logits differ from the
trainer's by up to 0.015 (NOTES_E6a 1.6); with TF32 off the scorer reproduced the trainers' own W_unsafe exactly
(difference 0.0 on 22 matching row sets, e5/offline/auc_rigid_prelim_1115.json). Same arguments as ag_offline_auc.py."""
import runpy, sys
from pathlib import Path
import torch

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
tool = Path(__file__).resolve().parent / 'ag_offline_auc.py'
sys.argv = [str(tool)] + sys.argv[1:]
runpy.run_path(str(tool), run_name='__main__')
