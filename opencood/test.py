# 验证 timestamp 的数量和regular 的关系
import os
import sys
import numpy as np 
import time
from tqdm import tqdm
from tqdm.contrib import tenumerate
from tqdm.auto import trange

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
        if res.endswith('.yaml'):
            timestamp = res.replace('.yaml', '')
        elif res.endswith('.json'):
            timestamp = res.replace('.json', '')
        else:
            print("Woops! There is no processing method for file {}".format(res))
            sys.exit(1)
        timestamps.append(timestamp)

    return timestamps

# 处理的数据集
data_seperate_name = "train"
# pcd 文件路径
root_dir = "/GPFS/public/OPV2V_irregular_npy/" + data_seperate_name

# first load all paths of different scenarios
scenario_folders = sorted([os.path.join(root_dir, x)
                            for x in os.listdir(root_dir) if
                            os.path.isdir(os.path.join(root_dir, x))])
scenario_folders_name = sorted([x
                            for x in os.listdir(root_dir) if
                            os.path.isdir(os.path.join(root_dir, x))])

# loop over all scenarios
for (i, scenario_folder) in enumerate(scenario_folders):
    # if i!=43:
    #     continue
    print("==== scenario %s ====" % (scenario_folders_name[i]))
    # start_time = time.time()
    
    # copy timestamps npy file
    timestamps_file = os.path.join(scenario_folder, 'timestamps.npy')
    time_annotations = np.load(timestamps_file)

    # iterate all cav in this scenario
    cav_list = sorted([x 
                        for x in os.listdir(scenario_folder) if 
                        os.path.isdir(os.path.join(scenario_folder, x))], 
                        key=lambda y: int(y))
    assert len(cav_list) > 0

    yaml_files = sorted([x
                        for x in os.listdir(os.path.join(scenario_folder, cav_list[0])) if
                        x.endswith(".json")], 
                        key=lambda y:float((y.split('/')[-1]).split('.json')[0]))
    start_timestamp = int(float(extract_timestamps(yaml_files)[0]))
    while(1):
        time_id_json = ("%.3f" % float(start_timestamp)) + ".json"
        time_id_yaml = ("%.3f" % float(start_timestamp)) + ".yaml"
        if not (time_id_json in yaml_files or time_id_yaml in yaml_files):
            start_timestamp += 1
        else:
            break

    end_timestamp = int(float(extract_timestamps(yaml_files)[-1]))
    if start_timestamp%2 == 0:
        # even
        end_timestamp = end_timestamp-1 if end_timestamp%2==1 else end_timestamp
    else:
        end_timestamp = end_timestamp-1 if end_timestamp%2==0 else end_timestamp
    num_timestamps = int((end_timestamp - start_timestamp)/2 + 1)

    all_timestamps = []
    for (j, cav_id) in enumerate(cav_list):  
        if j==0: # ego
            timestamps = [start_timestamp+2*i for i in range(num_timestamps)]
        else:
            timestamps = list(time_annotations[j-1, :])
        all_timestamps.append(timestamps)
        print("cav %s has %d timestamps" % (cav_id, len(timestamps)))
    all_timestamps = np.array(all_timestamps)
    order_flag = (all_timestamps[1:, :] - all_timestamps[0, :]) < 0
    if int(order_flag.max()) > 0:
        print("Fuck!")
        

        # # use the frame number as key, the full path as the values
        # yaml_files = \
        #     sorted([os.path.join(cav_path_yaml, x)
        #             for x in os.listdir(cav_path_yaml) if
        #             x.endswith('.yaml')])
        # timestamps = extract_timestamps(yaml_files)	
        
    """
        for k, timestamp in tenumerate(timestamps):
            # if k > 30:
            #     break

            # new save dir
            save_path = os.path.join(save_dir, scenario_folders_name[i], cav_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                # print("======Path {} created successfully!======".format(save_path))

            # change pcd to npy 
            lidar_file = os.path.join(cav_path_pcd, timestamp + '.pcd')
            new_lidar_file = os.path.join(save_path, timestamp + '.npy')
            if not os.path.exists(new_lidar_file):
                lidar = pcd_to_np(lidar_file)
                np.save(new_lidar_file, lidar)

            # change yaml to json
            yaml_file = os.path.join(cav_path_yaml, timestamp + '.yaml')
            json_file = os.path.join(cav_path_yaml, timestamp + '.json')
            new_json_file = os.path.join(save_path, timestamp + '.json')
            if os.path.exists(json_file): # 原数据集有json文件
                if not os.path.exists(new_json_file):
                    shutil.copyfile(json_file, new_json_file)
            else:
                if not os.path.exists(new_json_file):
                    params = load_yaml(yaml_file)
                    json_str = json.dumps(params, indent=4)
                    with open(new_json_file, 'w') as json_file:
                        json_file.write(json_str)
            
            # new_yaml_file = os.path.join(save_path, timestamp + '.yaml')
            # if not os.path.exists(new_yaml_file):
            #     shutil.copyfile(yaml_file, new_yaml_file)

            # print(timestamp)
    end_time = time.time()
    print("=== %d-th scenario (%s) finished, consumed %.3f mins. ===" \
        % (i, scenario_folders_name[i], (end_time-start_time)/60.0))
    """