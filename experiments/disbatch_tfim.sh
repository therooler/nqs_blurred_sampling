#!/bin/bash
module purge

### Define the list of N values (system sizes). Modify as needed.
Ns=(8)
Ps=(16)

# make sure outputs dir exists
mkdir -p outputs
mkdir -p disbatchfiles

# Tasks file: collect commands to run later (one per line)
TASKS="TASKS_TFIM_QUENCH"
>"${TASKS}"

echo "[INFO] Preparing to generate task list"
for N in "${Ns[@]}"; do
  for P in "${Ps[@]}"; do
        OUTF="outputs/tfim_quench.N${N}_P${P}.out"
        echo "[INFO] Adding task -> N=${N}, P=${P} -> logging to ${OUTF}"
        cmd="source ~/blurred_sampling/.venv/bin/activate && python -u tfim_quench_experiment.py --L=$N --power=$P --kernel_size=6 --hc_multiplier=0.1 &> ${OUTF}"
        echo "$cmd" >> "${TASKS}"
  done
done
echo "[INFO] Wrote tasks to ${TASKS} (one command per line)."
module load disBatch/beta
## Submit disBatch job and write stdout/stderr into the repo's outputs/ and errors/ folders
sbatch --output=outputs/out.out -n 1 -c 8 -p gpu --gpus-per-task=4 --time=48:00:00 disBatch TASKS_TFIM_QUENCH -p disbatchfiles/
echo "[INFO] Done launching jobs"