#!/bin/bash

# Call using: >bash /data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo/apps/automation/automate_launch.sh

echo "Executing $0 on: " `date`
echo "\n"

# echo "Default Parameter: $0"  # Name of script file
# echo "First Parameter: $1"    # Code Path
# echo "Second Parameter: $2"   # Data Directory Path
# echo "Third Parameter: $3"
# echo "Fourth Parameter: $4"
# echo "\n"

# Setup working directories and parameters: for code, data, logs, models, tile_id
## Calling process must ensure these folders exist

self_path=$0

python_launch_string=$1
# echo "python_launch_string: $python_launch_string"

qsub_launch_string=$2
# echo "qsub_launch_string: $qsub_launch_string"


# data_directory=$1
# conda_environment_path=$2
# sen2cor_path=$3  
# code_directory=$1
# python_filename=$2
# logs_directory=$4
# models_directory=$5
# tile_id=$6

# echo "Code Directory: $code_directory"
# echo "Python Executable : $python_filename"
# echo "Data Directory: $data_directory"
# echo "Logs Directory: $logs_directory"
# echo "Models Directory: $models_directory"
# echo "TileID : $tile_id"

# TODO: ADD CODE TO APPLY DEFAULT PATHS & PARAMETERS IF VALUES ARE NOT PASSED IN...
# code_directory="$wd/qsub_processing_logs"
# data_directory="/data/clcr/shared/IMPRESS/Ivan/kenya_national_prod/models"
# logs_directory="/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo/apps/change_detection"
# models_directory="/home/i/ir81/pyeo_prod_0_8_0/pyeo/pyeo/apps/model_creation"


# qsub_launch_string = "cd $data_directory; module load python; source activate $conda_environment_path; 
# SEN2COR_HOME=$sen2cor_path; export SEN2COR_HOME; 
# python $pyeo_root/tile_based_change_detection_from_cover_maps.py $wd/mato.ini --tile $tile --build_composite --chunks 10 --download --download_source aws --skip --dev --quicklooks"

# qsub_parameter_string = qsub -N $tile -o $wd/qsub_processing_logs/$tile\_o.txt -e $wd/qsub_processing_logs/$tile\_e.txt -l walltime=0:23:59:00,nodes=1:ppn=16,vmem=64Gb"

echo $python_launch_string | $qsub_launch_string


# echo "Process launched via qsub:"
# echo "Use 'showq -u ir81' or 'watch -n 10 qstat -a -nl' to see progress. Use tracejob(ID) to see resource use. Use qdel ID to kill a job."
