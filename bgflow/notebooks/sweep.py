# python ./sampling_diff_energies.py --lr=0.005 --batch_size=64 --n_layers=4 --use_hartmann=1





import subprocess
import os

if __name__ == "__main__":

    # base directories
    run_idx = 0
    job_script_directory = "/network/scratch/g/guangyuan.wang/comp400/temp_job_scripts"
    out_dir = "/network/scratch/g/guangyuan.wang/comp400/standard_bg_exp/out"
    os.makedirs(job_script_directory, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # hyperparameter grids
    learning_rates = [1e-3, 5e-3, 1e-2]
    batch_sizes = [32, 64]
    n_coupling_layers = [2, 4, 6]
    use_hartmann_flags = [False, True]

    # iterate over all combinations
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for n_layers in n_coupling_layers:
                for use_hartmann in use_hartmann_flags:

                    time = "24:00:00"

                    # define command to run
                    python_run_command = (
                        f"python ./sampling_diff_energies.py "
                        f"--lr={lr} "
                        f"--batch_size={batch_size} "
                        f"--n_layers={n_layers} "
                        f"--use_hartmann={int(use_hartmann)}"
                    )

                    # slurm job script content
                    job_script_content = f"""#!/usr/bin/bash
echo Start time
echo "`date +%Y:%m:%d-%H:%M:%S`"

module unload python
module load anaconda

cd /network/scratch/g/guangyuan.wang/comp400/
conda activate ./gp_env

echo {python_run_command}
{python_run_command}

echo Stop time
echo "`date +%Y:%m:%d-%H:%M:%S`"
"""

                    # job name and file
                    job_name = f"bg_lr{lr}_bs{batch_size}_nl{n_layers}_{'hart' if use_hartmann else 'dw'}"
                    job_script_filename = os.path.join(
                        job_script_directory, f"{job_name}.sh"
                    )

                    with open(job_script_filename, "w") as job_script_file:
                        job_script_file.write(job_script_content)

                    # slurm submission
                    launch_command = (
                        f"sbatch --job-name={job_name} "
                        f"--time={time} --gres=gpu:1 -c 2 --mem=24G "
                        f"--output={out_dir}/{job_name}.out "
                        f"{job_script_filename}"
                    )
                    subprocess.run(launch_command, shell=True, executable="/usr/bin/bash")

                    run_idx += 1

    print(f"total runs submitted: {run_idx}")
