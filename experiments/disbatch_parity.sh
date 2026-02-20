#!/bin/bash
module purge

### Define the list of N values (system sizes). Modify as needed.
Ns=(32)
Ps=(13)
Qs=(0.5)
# make sure outputs dir exists
mkdir -p outputs
mkdir -p disbatchfiles

# Tasks file: collect commands to run later (one per line)
TASKS="TASKS_PARITY"
>"${TASKS}"

echo "[INFO] Preparing to generate task list"
for N in "${Ns[@]}"; do
  for P in "${Ps[@]}"; do
    for Q in "${Qs[@]}"; do
        OUTF="outputs/pfaffian.N${N}_P${P}_Q${Q}.out"
        echo "[INFO] Adding task -> N=${N}, P=${P}, Q=${Q} -> logging to ${OUTF}"
        cmd="export ENABLE_JAXMG=1 && source ~/blurred_sampling/.venv/bin/activate && python -u parity_experiment_pfaffian.py --N=$N --power=$P --q=$Q --driver_type=vanilla --h=0.125&> ${OUTF}"
        echo "$cmd" >> "${TASKS}"
    done
  done
done
echo "[INFO] Wrote tasks to ${TASKS} (one command per line)."
module load disBatch/beta
## Submit disBatch job and write stdout/stderr into the repo's outputs/ and errors/ folders
sbatch --output=outputs/out.out -n 1 -c 8 -p gpu --gpus-per-task=4 --time=48:00:00 disBatch TASKS_PARITY -p disbatchfiles/
echo "[INFO] Done launching jobs"