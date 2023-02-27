"""
This file will update the files from GPFS
"""

import os
import shutil

def backup_script(folders_to_save=["data_utils", "loss", "models", "tools", "utils", "visualization"]):
    target_folder = "/DB/public/sizhewei/SyncStation/opencood"
    source_folder = "/DB/rhome/sizhewei/percp/OpenCOOD/opencood"
    for folder_name in folders_to_save:
        t_path = os.path.join(target_folder, folder_name)
        s_path = os.path.join(source_folder, folder_name)
        if os.path.exists(t_path):
            print(f"{folder_name} exists! Deleting...")
            shutil.rmtree(t_path, ignore_errors=True)
            print(f"{folder_name} deleted! Syncing...")
        shutil.copytree(s_path, t_path, ignore= shutil.ignore_patterns("__pycache__", "*.c", ".so", ".pyx"))
        print(f"{folder_name} Synced!")
    print("=== DB to GPFS sync successfully! ===")

if __name__ == "__main__":
    backup_script()
