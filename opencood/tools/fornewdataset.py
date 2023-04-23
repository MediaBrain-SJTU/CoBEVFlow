import os
import torch
from scipy import stats
from collections import OrderedDict

def retrieve_base_data(scenario_database, len_record, idx, binomial_n=10, binomial_p=0.1, k=3, is_no_shift=False, is_same_sample_interval=False):
    """
    Given the index, return the corresponding data.

    Parameters
    ----------
    idx : int
        Index given by dataloader.

    Returns
    -------
    data : dict
        The dictionary contains loaded yaml params and lidar data for
        each cav.
        Structure: 
        {
            cav_id_1 : {
                'ego' : true,
                curr : {
                    'params': json_path,
                    'timestamp': string
                },
                past_k : {		                # (k) totally
                    [0]:{
                        'params': json_path,
                        'timestamp': string,
                        'time_diff': float,
                        'sample_interval': int
                    },
                    [1] : {},	(id-1)
                    ...,		
                    [k-1] : {} (id-(k-1))
                },
            }, 
            cav_id_2 : {		                # (k) totally
                'ego': false, 
                ...
            }, 
            ...
        }
    """
    sample_interval_exp = int(binomial_n * binomial_p)
    # we loop the accumulated length list to get the scenario index
    scenario_index = 0
    for i, ele in enumerate(len_record):
        if idx < ele:
            scenario_index = i
            break
    scenario_database = scenario_database[scenario_index]
    
    # 生成冻结分布函数
    bernoulliDist = stats.bernoulli(binomial_p) 

    data = OrderedDict()
    # 找到 current 时刻的 timestamp_index 这对于每辆车来讲都一样
    curr_timestamp_idx = idx if scenario_index == 0 else \
                    idx - len_record[scenario_index - 1]
    curr_timestamp_idx = curr_timestamp_idx + binomial_n * k
    
    # load files for all CAVs
    for cav_id, cav_content in scenario_database.items():
        '''
        cav_content 
        {
            timestamp_1 : {
                yaml: path,
                lidar: path, 
                cameras: list of path
            },
            ...
            timestamp_n : {
                yaml: path,
                lidar: path, 
                cameras: list of path
            },
            'regular' : {
                timestamp_1 : {},
                ...
                timestamp_n : {}
            },
            'ego' : true / false , 
        },
        '''
        data[cav_id] = OrderedDict()
        
        # 1. set 'ego' flag
        data[cav_id]['ego'] = cav_content['ego']
        
        # 2. current frame, for co-perception lable use
        data[cav_id]['curr'] = {}

        timestamp_key = list(cav_content['regular'].items())[curr_timestamp_idx][0]
        
        # 2.1 load curr params
        # json is faster than yaml
        json_file = cav_content['regular'][timestamp_key]['yaml'].replace("yaml", "json")
        json_file = json_file.replace("OPV2V_irregular_npy", "OPV2V_irregular_npy_updated")
        data[cav_id]['curr']['params'] = json_file

        # 2.3 store curr timestamp and time_diff
        data[cav_id]['curr']['timestamp'] = timestamp_key
        data[cav_id]['curr']['time_diff'] = 0.0
        data[cav_id]['curr']['sample_interval'] = 0

        # 3. past frames, for model input
        data[cav_id]['past_k'] = OrderedDict()
        latest_sample_stamp_idx = curr_timestamp_idx
        # past k frames, pose | lidar | label(for single view confidence map generator use)
        for i in range(k):
            # sample_interval
            if data[cav_id]['ego']:             # ego sample_interval = E(B(n, p))
                if i == 0: # ego-past-0 与 ego-curr 是一样的
                    data[cav_id]['past_k'][i] = data[cav_id]['curr']
                    continue
                sample_interval = sample_interval_exp
                if sample_interval == 0:
                    sample_interval = 1
            else:                               # non-ego sample_interval ~ B(n, p)
                if sample_interval_exp==0 \
                    and is_no_shift \
                        and i == 0:
                    data[cav_id]['past_k'][i] = data[cav_id]['curr']
                    continue
                if is_same_sample_interval:
                    sample_interval = sample_interval_exp
                else:
                    # B(n, p)
                    trails = bernoulliDist.rvs(binomial_n)
                    sample_interval = sum(trails)
                if sample_interval==0:
                    if i==0: # 检查past 0 的实际时间是否在curr 的后面
                        tmp_time_key = list(cav_content.items())[latest_sample_stamp_idx][0]
                        if dist_time(tmp_time_key, data[cav_id]['curr']['timestamp'])>0:
                            sample_interval = 1
                    if i>0: # 过去的几帧不要重复
                        sample_interval = 1                

            # check the timestamp index
            data[cav_id]['past_k'][i] = {}
            latest_sample_stamp_idx -= sample_interval
            timestamp_key = list(cav_content.items())[latest_sample_stamp_idx][0]
            # load the corresponding data into the dictionary
            # load param file: json is faster than yaml
            json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            json_file = json_file.replace("OPV2V_irregular_npy", "OPV2V_irregular_npy_updated")
            data[cav_id]['past_k'][i]['params'] = json_file

            data[cav_id]['past_k'][i]['timestamp'] = timestamp_key
            data[cav_id]['past_k'][i]['sample_interval'] = sample_interval
            data[cav_id]['past_k'][i]['time_diff'] = \
                dist_time(timestamp_key, data[cav_id]['curr']['timestamp'])

    return data

def dist_time(ts1, ts2, i = -1):
    """caculate the time interval between two timestamps

    Args:
        ts1 (string): time stamp at some time
        ts2 (string): current time stamp
        i (int, optional): past frame id, for debug use. Defaults to -1.
    
    Returns:
        time_diff (float): time interval (ts1 - ts2)
    """
    if not i==-1:
        return -i
    else:
        return (float(ts1) - float(ts2))


# debug_path = '/remote-home/share/OPV2V_irregular_npy'
debug_path = '/dssg/home/acct-seecsh/seecsh/sizhewei/data_sftp'
scenario_database = torch.load(os.path.join(debug_path, 'scenario_database.pt'))
len_record = torch.load(os.path.join(debug_path, 'len_record.pt'))


unit_data = retrieve_base_data(scenario_database, len_record, 1)