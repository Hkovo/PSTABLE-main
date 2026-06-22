import subprocess


def run_command(cmd_list):
    try:
        print(f"\033[34m[INFO] Running:\033[0m {' '.join(cmd_list)}")
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\033[31m[ERROR] Command failed with code {e.returncode}:\033[0m {' '.join(cmd_list)}")
    except Exception as e:
        print(f"\033[31m[UNEXPECTED ERROR] {e}:\033[0m {' '.join(cmd_list)}")
