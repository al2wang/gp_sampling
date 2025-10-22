import subprocess
import os

if __name__ == '__main__':

    run_idx = 0
    job_script_directory = "network/scratch/g/guangyuan.wang/comp400/temp_job_scripts"
    out_dir = "standard_gp_exp/out"
    os.makedirs(job_script_directory, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    while run_idx <= 160:
        time = '24:00:00'
        python_run_command = f"python ./Standard_GP/baselines/run_script.py --index={run_idx}"
        job_script_content = f'''#!/usr/bin/bash
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
        '''
        job_name = f"gp_test_{run_idx}"
        job_script_filename = os.path.join(
            job_script_directory, f"{job_name}.sh")

        run_idx += 1
        
        with open(job_script_filename, 'w') as job_script_file:
            job_script_file.write(job_script_content)
        launch_command = f'sbatch --job-name={job_name} --time={time} --gres=gpu:1 -c 2 --mem=24G --output={out_dir}/{job_name}.out {job_script_filename}'
        subprocess.run(launch_command, shell=True,
                        executable='/usr/bin/bash')

    print(f"total runs submitted: {run_idx}")