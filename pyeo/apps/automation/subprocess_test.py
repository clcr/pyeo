import subprocess

shell_command_string = (
    "ls /data/clcr/shared/IMPRESS/Ivan; ls /data/clcr/shared/IMPRESS/Ivan"
)
# shell_command_string = 'touch /data/clcr/shared/Ivan'
result = subprocess.run(
    shell_command_string, capture_output=True, text=True, shell=True
)
print(result.stdout)
