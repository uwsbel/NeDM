#!/bin/bash
# arena_gator_20260925 soil stage 1: does pytorch/2.10.0 (the training recipe) work on this node's GPUs?
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -o /work1/dannegrut/harry/experiments/arena_gator_20260925/tools/s1/logs/%x_%j.out
unset NEDM_VEHICLE
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
echo "host=$(hostname) part=$SLURM_JOB_PARTITION"
rocminfo | grep -E "^\s+Name:\s+gfx" | sort | uniq -c
python3.12 - <<'PY'
import torch, time
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'n', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
d = torch.device('cuda:0')
a = torch.randn(512, 512, device=d); b = a @ a; torch.cuda.synchronize(); print('matmul ok', float(b.abs().mean()))
c = torch.nn.Conv2d(5, 16, 3).to(d); y = c(torch.randn(8, 5, 96, 32, device=d)); y.sum().backward(); print('conv ok')
g = torch.nn.GRU(15, 64, batch_first=True).to(d); o, h = g(torch.randn(8, 40, 15, device=d)); o.sum().backward(); print('gru ok')
t0 = time.time()
for _ in range(50):
    y = c(torch.randn(256, 5, 96, 32, device=d)); y.sum().backward()
torch.cuda.synchronize(); print('50 conv steps', round(time.time() - t0, 2), 's')
PY
echo "exit: $?"
