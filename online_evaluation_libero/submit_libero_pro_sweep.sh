#!/bin/bash
# Submit the LIBERO-PRO perturbation sweep as 8 SLURM jobs:
#   2 checkpoints x 4 base suites (goal/spatial/10/object).
# Each job evaluates the 5 perturbation variants of one base suite
# (_object/_swap/_lan/_task/_env) = 5 suites x 10 tasks x 50 rollouts.
#
# Prerequisite: LIBERO-PRO perturbation bddl_files + init_files must already be
# populated under LIBERO-PRO/libero/libero/{bddl_files,init_files}/.
set -euo pipefail
cd /data/user_data/ziyinx/analogical_manipulation_clean

WRAPPER=online_evaluation_libero/eval_libero_pro_suite.sh
BDDL_ROOT=/data/user_data/ziyinx/LIBERO-PRO/libero/libero/bddl_files
bases=(goal spatial 10 object)
perturbations=(object swap lan task env)

# tag -> PRESET (which maps to checkpoint + model in eval_libero_pro.sh)
declare -A PRESET=( [motion]=3dfavla_motion [subset]=3dfavla_motion_subset )

for tag in motion subset; do
    for base in "${bases[@]}"; do
        # Only include perturbation suites whose bddl files are actually present.
        suites=""
        for p in "${perturbations[@]}"; do
            s="libero_${base}_${p}"
            if compgen -G "$BDDL_ROOT/$s/*.bddl" > /dev/null; then
                suites="${suites:+$suites }$s"
            else
                echo "[skip] $s has no bddl files; excluding from job lpro_${tag}_${base}" >&2
            fi
        done
        if [[ -z "$suites" ]]; then
            echo "[warn] no suites with data for base=$base; not submitting lpro_${tag}_${base}" >&2
            continue
        fi
        sbatch --job-name="lpro_${tag}_${base}" \
            --export="ALL,CONDA_ENV=flowervla,PRESET=${PRESET[$tag]},SUITES=${suites},EXP=LIBERO-PRO,SEED=0" \
            "$WRAPPER"
    done
done
