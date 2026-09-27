# H_soil (task B): the HMMWV f104 soil planner trained on exactly the ids the Gator validated (tiers 0-6)

The Gator validated all 8,399 of the 8,399 soil ids of tiers 0-6 (`e5/ids_soil/gator_soil_validated_t0-6.json`: 8,399
selected, 0 rejected by the soil QA check, launch check passed on all, Gator vehicle record on all). The H training file
built from those ids, `G3/e4/soil_s1/subsets/H_f104_hmmwv_soil.npz`, is byte-identical to the M1 soil file
`G3/e4/soil_s1/subsets/M1_f104_hmmwv_soil.npz` (both sha256 `48a29d16451b179e0e3dca103c1194e2740f63f8e16f9fef2214b43ee2c62d67`;
`e5/ids_soil/H_vs_M1_soil.json`: same id set, same row order).

So H is not trained again: with the same file, recipe, seeds 0-4 and GPU type it is the M1a soil ensemble. Use the
checkpoints in `../M1a_soil/` (sha256 in `../M1a_soil/SHA256SUMS`, repeated with `../M1a_soil/` paths in `SHA256SUMS` here, so `sha256sum -c SHA256SUMS` works in this folder, as in the rigid `../H/`; paths fixed by the verifier, VERIFY_S1.md). The holdout-mode twin of H is
`e5/train/soil_s1/offline_soil/M1_soil_holdout_s*.pt`. (Same convention as the rigid `../H/`.)
