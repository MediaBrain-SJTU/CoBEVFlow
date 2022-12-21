# transfer all lidar file from .pcd to .npy 
# sizhewei @ <sizhewei@sjtu.edu.cn>
# 2022/12/20

import os
import shutil
import open3d as o3d
import numpy as np
from pypcd import pypcd
from tqdm import tqdm
from tqdm.contrib import tenumerate
from tqdm.auto import trange

def pcd_to_np(pcd_file):
    """
    Read  pcd and return numpy array.

    Parameters
    ----------
    pcd_file : str
        The pcd file that contains the point cloud.

    Returns
    -------
    pcd : o3d.PointCloud
        PointCloud object, used for visualization
    pcd_np : np.ndarray
        The lidar data in numpy format, shape:(n, 4)

    """
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def extract_timestamps(yaml_files):
    """
    Given the list of the yaml files, extract the mocked timestamps.

    Parameters
    ----------
    yaml_files : list
        The full path of all yaml files of ego vehicle

    Returns
    -------
    timestamps : list
        The list containing timestamps only.
    """
    timestamps = []

    for file in yaml_files:
        res = file.split('/')[-1]

        timestamp = res.replace('.yaml', '')
        timestamps.append(timestamp)

    return timestamps


if __name__ == "__main__":
    root_dir = "/GPFS/rhome/yifanlu/workspace/dataset/OPV2V/test"

    save_dir = "/DB/public/OPV2V_w_npy/test"

    # first load all paths of different scenarios
    scenario_folders = sorted([os.path.join(root_dir, x)
                                for x in os.listdir(root_dir) if
                                os.path.isdir(os.path.join(root_dir, x))])
    scenario_folders_name = sorted([x
                                for x in os.listdir(root_dir) if
                                os.path.isdir(os.path.join(root_dir, x))])

    # loop over all scenarios
    for (i, scenario_folder) in tenumerate(scenario_folders):
        # if i > 1:
        #     break
        # at least 1 cav should show up
        cav_list = sorted([x 
                            for x in os.listdir(scenario_folder) if 
                            os.path.isdir(os.path.join(scenario_folder, x))])
        assert len(cav_list) > 0

        # loop over all CAV data
        for (j, cav_id) in tenumerate(cav_list):
            # save all yaml files to the dictionary
            cav_path = os.path.join(scenario_folder, cav_id)
            
            # use the frame number as key, the full path as the values
            yaml_files = \
                sorted([os.path.join(cav_path, x)
                        for x in os.listdir(cav_path) if
                        x.endswith('.yaml')])
            timestamps = extract_timestamps(yaml_files)	

            for k, timestamp in tenumerate(timestamps):
                # if k > 30:
                #     break
                
                json_file = os.path.join(cav_path, timestamp + '.json')
                yaml_file = os.path.join(cav_path, timestamp + '.yaml')

                lidar_file = os.path.join(cav_path, timestamp + '.pcd')
                lidar = pcd_to_np(lidar_file)

                save_path = os.path.join(save_dir, scenario_folders_name[i], cav_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    # print("======Path {} created successfully!======".format(save_path))
                
                new_lidar_file = os.path.join(save_path, timestamp + '.npy')
                if not os.path.exists(new_lidar_file):
                    np.save(new_lidar_file, lidar)

                new_param_file = os.path.join(save_path, timestamp + '.json')
                if not os.path.exists(new_param_file):
                    shutil.copyfile(json_file, new_param_file)

                new_yaml_file = os.path.join(save_path, timestamp + '.yaml')
                if not os.path.exists(new_yaml_file):
                    shutil.copyfile(yaml_file, new_yaml_file)

                # print(timestamp)