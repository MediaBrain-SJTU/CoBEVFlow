"""
This file will update the files in DB from GPFS(public)
"""

import os
import shutil
import argparse

def opt_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--backup", "--b", type=int, default=1, help="backup the files before updating")
    opt = parser.parse_args()
    return opt

def backup_script(opt, folders_to_save=["data_utils", "loss", "models", "tools", "utils", "visualization"]):
    target_folder = "/DB/rhome/sizhewei/percp/OpenCOOD/opencood"
    source_folder = "/DB/public/sizhewei/SyncStation/v100_2_gpfs/opencood"
    backup_file = os.path.join(target_folder, 'backup')
    for folder_name in folders_to_save:
        t_path = os.path.join(target_folder, folder_name)
        s_path = os.path.join(source_folder, folder_name)
        if os.path.exists(t_path):
            if opt.backup==1:
                if not os.path.exists(path=backup_file):
                    os.mkdir(path=backup_file)
                print(f"{folder_name} exists! Making a backup...")
                backup_path = os.path.join(backup_file, folder_name)
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path, ignore_errors=True)
                shutil.copytree(t_path, backup_path, ignore= shutil.ignore_patterns("__pycache__", "*.c", ".so", ".pyx"))
                shutil.rmtree(t_path, ignore_errors=True)
                print(f"{folder_name} backuped! Syncing...")
            else:
                print(f"{folder_name} exists! Deleting...")
                shutil.rmtree(t_path, ignore_errors=True)
                print(f"{folder_name} deleted! Syncing...")  
        shutil.copytree(s_path, t_path, ignore= shutil.ignore_patterns("__pycache__", "*.c", ".so", ".pyx"))
        print(f"{folder_name} Synced!")
        print("=====================================")

if __name__ == "__main__":
    opt = opt_parser()
    backup_script(opt)
