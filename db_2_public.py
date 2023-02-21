"""
This file will update the files from GPFS
"""

import os

def backup_script(folders_to_save=["data_utils", "loss", "models", "tools", "utils", "visualization"]):
    target_folder = "/DB/data/sizhewei/SyncStation/opencood"
    source_folder = "/DB/rhome/sizhewei/percp/OpenCOOD/opencood"
    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(source_folder, folder_name)
        shutil.copytree(source_folder, target_folder, ignore= shutil.ignore_patterns("__pycache__", "*.c", ".so", ".pyx"))
        print("Folder %s Synced!" % folder_name)

if __name__ == "__main__":
    backup_script()
