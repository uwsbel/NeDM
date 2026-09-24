#!/usr/bin/env python3
"""Candidate stopping rules replayed over every logged finetune.jsonl.

The guard (spike > 0.6 AND reward drop > 25%) missed two bad capacity-study runs whose
windows left the corpus region at most 44% and 58% of the time. This prints, per run,
the first iteration at which each candidate rule would have fired, so rules can be
compared against Chrono labels. Usage: guard_rules.py "<glob of finetune.jsonl>" ...
"""
W=50
def rules(L):
    out={}; best=None; buf=[]
    for r in L:
        buf.append(r); buf=buf[-W:]
        if len(buf)<W: continue
        sp=sum(x["ood"]>0.005 for x in buf)/W; mr=sum(x["reward"] for x in buf)/W
        best=mr if best is None else max(best,mr); drop=(best-mr)/max(abs(best),1e-9)
        vl=max(x["value_loss"] for x in buf)
        for k,c in (("current",sp>0.6 and drop>0.25),("spike.4&drop.25",sp>0.4 and drop>0.25),
                    ("drop>1",drop>1.0),("drop>.5",drop>0.5),("vloss>1000",vl>1000),("vloss>1000&drop>.25",vl>1000 and drop>0.25)):
            if c and k not in out: out[k]=r["iter"]
    return out
for f in sorted(set(sum([glob.glob(os.path.expanduser(p),recursive=True) for p in sys.argv[1:]],[]))):
    try: L=[json.loads(l) for l in open(f)]
    except Exception: continue
    if len(L)<W or "value_loss" not in L[0]: continue
    o=rules(L)
    print(f.replace(os.path.expanduser("~"),"~"), len(L), o if o else "-")
