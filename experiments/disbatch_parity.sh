#!/bin/bash
module purge

### Define the list of N values (system sizes). Modify as needed.
Ns=(16)
Ps=(10 11 12 13)

# make sure outputs dir exists
mkdir -p outputs
mkdir -p disbatchfiles

# Tasks file: collect commands to run later (one per line)
TASKS="TASKS_PARITY"
>"${TASKS}"

echo "[INFO] Preparing to generate task list"
for N in "${Ns[@]}"; do
  for P in "${Ps[@]}"; do
        OUTF="outputs/hadamard_vanilla.N${N}_P${P}.out"
        echo "[INFO] Adding task -> N=${N}, P=${P} -> logging to ${OUTF}"
        cmd="source ~/blurred_sampling/.venv/bin/activate && python -u parity_experiment.py --N=$N --power=$P &> ${OUTF}"
        echo "$cmd" >> "${TASKS}"
  done
done
echo "[INFO] Wrote tasks to ${TASKS} (one command per line)."
module load disBatch/beta
## Submit disBatch job and write stdout/stderr into the repo's outputs/ and errors/ folders
sbatch --output=outputs/out.out -n 5 -c 4 -p gpu --gpus-per-task=4 --time=48:00:00 disBatch TASKS_PARITY -p disbatchfiles/
echo "[INFO] Done launching jobs"