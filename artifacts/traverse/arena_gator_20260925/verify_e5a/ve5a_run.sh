#!/bin/bash
# VERIFY_E5a: extract light columns + sha256 of every E5a npz (login node, nice/ionice, read only)
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925; T=$G3/tools/verify_e5a
source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1
unset NEDM_VEHICLE
F="$G3/e4/f104_hmmwv/ci_f104_hmmwv_both.npz $G3/e4/g203_hmmwv/ci_g203_hmmwv_rigid.npz $G3/e4/g228_hmmwv/ci_g228_hmmwv_rigid.npz $G3/e4/f104_gator/ci_f104_gator_rigid.npz $(ls $G3/e4/subsets/*.npz) $(ls $G3/e5/eval/*.npz)"
nice -n 15 ionice -c3 $NRD_PYTHON -u $T/ve5a_extract_light.py $T/out $F > $T/out_extract.log 2>&1
echo "extract exit $?" >> $T/out_extract.log
nice -n 15 ionice -c3 sha256sum $F > $T/out/sha256.txt 2>&1
echo "sha exit $?" >> $T/out_extract.log
