import os
import subprocess
import time
import pandas as pd

print(f"automation_test.py execution started in folder {os.getcwd()}")

# Now launch a set of instances using qsub for parallelism

data_directory = "/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo/apps/automation"
sen2cor_path = "/home/i/ir81/Sen2Cor-02.09.00-Linux64"
conda_environment_path = "/home/i/ir81/miniconda3/envs/pyeo_env"
code_directory = "/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo/apps/automation"
config_directory = "/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo/apps/automation"
config_filename = "pyeo.ini"
python_executable = "_random_duration_test_program.py"
tile_list = ["36MYE", "36MWD", "36MYC", "36MZC"]
# tile = '36MYE'
new_line = "\n"

# log_path = '/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo/apps/automation'
# models_path = '/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo/apps/automation'
# pbs_outputs_path = '/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo/apps/automation'
# pbs_errors_path = '/data/clcr/shared/IMPRESS/Ivan/pyeo/pyeo/pyeo/apps/automation'

# pbs_processing_options = '-l walltime=00:10:00,nodes=1:ppn=16,vmem=2Gb'  # '-l walltime=0:23:59:00,nodes=1:ppn=16,vmem=64Gb'
# pbs_output_options = ' -o o.txt'
# pbs_process_name = f'- N test_instance{i}'


def qstat_to_dataframe():
    # Run qstat command and capture the stdout
    result = subprocess.run(["qstat"], capture_output=True)
    # Decode the byte string into a regular string
    output = result.stdout.decode("utf-8")
    # Split the output into lines
    lines = output.split("\n")
    # Remove any empty lines
    lines = [line.strip() for line in lines if line.strip()]
    if len(output) > 0:
        # Extract the header and data rows
        header = lines[0].split()
        data_rows = [line.split() for line in lines[2:]]
        # Create the pandas DataFrame
        df = pd.DataFrame(
            data_rows, columns=["JobID", "Name", "User", "TimeUsed", "Status", "Queue"]
        )
        return df
    else:
        return pd.DataFrame()  # Return an empty dataframe is no output from qstat


for tile in tile_list:
    print(
        f"{new_line}*** Preparing to launch {python_executable} for tile: {tile} ***{new_line}"
    )
    # os.system('echo "python _random_duration_test_program.py" | qsub -N f'test_instance{i}' -o o.txt -l walltime=00:10:00,nodes=1:ppn=16,vmem=2Gb')
    # os.system('python _random_duration_test_program.py')
    # result = subprocess.run(shell_command_string, capture_output=True, text=True, shell=True)
    # print(result.stdout)

    print(
        f"automation_test.py: Checking if tile {tile} is already being processed and, if so, delete process to avoid possible conflicts"
    )
    df = qstat_to_dataframe()
    if not df.empty:
        current_tile_processes_df = df[df["Name"] == tile]
        # print('current_tile_processes_df')
        # print(current_tile_processes_df)
        for index, p in current_tile_processes_df.iterrows():
            print(p)
            if p["Status"] in ["Q", "R"]:
                job_id = p["JobID"].split(".")[0]
                print(f"{new_line}Deleting job: {job_id} {new_line}")
                os.system(f"qdel {job_id}")

    print("automation_test.py: Preparing to launch tile processing of tile {tile}")

    python_launch_string = f"cd {data_directory}; module load python; source activate {conda_environment_path}; SEN2COR_HOME={sen2cor_path}; export SEN2COR_HOME; python {os.path.join(code_directory, python_executable)} {os.path.join(config_directory, config_filename)} {tile}"
    qsub_launch_string = f'qsub -N {tile} -o {os.path.join(data_directory, tile + "_o.txt")} -e {os.path.join(data_directory, tile + "_e.txt")} -l walltime=00:00:02:00,nodes=1:ppn=16,vmem=64Gb'
    shell_command_string = (
        f"./automate_launch.sh '{python_launch_string}' '{qsub_launch_string}'"
    )

    # new_line = '\n'
    # print(f'{python_launch_string=}{new_line}')
    # print(f'{qsub_launch_string=}{new_line}')
    # print(f'{shell_command_string=}{new_line}')

    result = subprocess.run(
        shell_command_string, capture_output=True, text=True, shell=True
    )
    print(f" Subprocess launched for tile {tile}, return value: {result.stdout}")


print("automation_test.py: subprocess launching completed")


print("automation_test.py: subprocess monitoring started")

monitoring_cycles = 60
monitoring_period_seconds = 10
for i in range(monitoring_cycles):
    time.sleep(monitoring_period_seconds)
    print(
        f"automation_test.py: Checking which tiles are still being processed after {i * monitoring_period_seconds} seconds"
    )

    df = qstat_to_dataframe()
    if not df.empty:
        for tile in tile_list:
            current_tile_processes_df = df[df["Name"] == tile]
            # print('current_tile_processes_df')
            # print(current_tile_processes_df)
            for index, p in current_tile_processes_df.iterrows():
                # print(p)
                if p["Status"] in ["Q", "R"]:
                    job_id = p["JobID"].split(".")[0]
                    print(
                        f'Tile {tile} still running pid: {job_id}, status; {p["Status"]} '
                    )
    else:
        break


print("automation_test.py subprocesses completed")
