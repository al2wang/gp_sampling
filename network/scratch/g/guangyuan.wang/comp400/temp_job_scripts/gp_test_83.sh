#!/usr/bin/bash
        echo Start time
        echo "`date +%Y:%m:%d-%H:%M:%S`"
        module unload python
        module load anaconda

        cd /network/scratch/g/guangyuan.wang/comp400/
        conda activate ./gp_env

        echo python ./Standard_GP/baselines/run_script.py --index=83
        python ./Standard_GP/baselines/run_script.py --index=83
        echo Stop time
        echo "`date +%Y:%m:%d-%H:%M:%S`"
        