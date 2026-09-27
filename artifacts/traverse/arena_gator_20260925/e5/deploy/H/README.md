# H (task B): the HMMWV f104 rigid planner trained on exactly the ids the Gator validated

The Gator validated all 24,000 of the 24,000 rigid f104 pool ids (`e5/ids/gator_rigid_validated.json`: 24,000 selected,
0 rejected, native-height check passed on all 24,000). The H training file built from those ids,
`G3/e4/subsets/H_f104_hmmwv_rigid.npz`, is byte-identical to the M1 file `G3/e4/subsets/M1_f104_hmmwv_rigid.npz`
(both sha256 `3174999e25828668cb0a3138084a248d140b1daceb42dff2bf62b1f96df370cf`; `e5/ids/H_vs_M1.json`: same id set, same
row order).

So H is not trained again: with the same file, the same recipe, the same seeds 0-4 and the same kind of GPU it is the
M1a ensemble. Use the checkpoints in `../M1a/` (sha256 in `../M1a/SHA256SUMS`, repeated below). The holdout-mode twin of
H is `e5/train/offline_rigid/M1_rigid_holdout_s*.pt`.
