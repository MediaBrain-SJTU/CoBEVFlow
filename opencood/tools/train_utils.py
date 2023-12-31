# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim
import numpy
from pathlib import Path

def backup_script(full_path, folders_to_save=["models", "data_utils", "utils", "loss", "tools"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)

# def load_saved_model(saved_path, model):
#     """
#     Load saved model if exiseted

#     Parameters
#     __________
#     saved_path : str
#        model saved path
#     model : opencood object
#         The model instance.

#     Returns
#     -------
#     model : opencood object
#         The model instance loaded pretrained params.
#     """
#     assert os.path.exists(saved_path), '{} not found'.format(saved_path)

#     def findLastCheckpoint(save_dir):
#         file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
#         if file_list:
#             epochs_exist = []
#             for file_ in file_list:
#                 result = re.findall(".*epoch(.*).pth.*", file_)
#                 epochs_exist.append(int(result[0]))
#             initial_epoch_ = max(epochs_exist)
#         else:
#             initial_epoch_ = 0
#         return initial_epoch_

#     if os.path.exists(os.path.join(saved_path, 'net_latest.pth')):
#         model.load_state_dict(torch.load(
#             os.path.join(saved_path,
#                          'net_latest.pth')))
#         return 100, model

#     initial_epoch = findLastCheckpoint(saved_path)
#     if initial_epoch > 0:
#         print('resuming by loading epoch %d' % initial_epoch)
#         model.load_state_dict(torch.load(
#             os.path.join(saved_path,
#                          'net_epoch%d.pth' % initial_epoch)), strict=False)

#     return initial_epoch, model

def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
        model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    if file_list:
        assert len(file_list) == 1
        print("resuming best validation model at epoch %d" % \
                eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
        model.load_state_dict(torch.load(file_list[0] , map_location='cpu'), strict=False)
        return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(
            os.path.join(saved_path,
                'net_epoch%d.pth' % initial_epoch), map_location='cpu'), strict=False)

    # ckpt = torch.load(os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch), map_location='cpu')
    # keys = [x for x in ckpt if 'resnet' in x]
    # ckpt_new = {}
    # for k in keys:
    #     old_key = k.split('layer')
    #     start = old_key[0]
    #     end = old_key[1][1:]
    #     mid = int(old_key[1][0]) - 1
    #     new_key = start + 'layer' + str(mid) + end
    #     ckpt_new[new_key] = ckpt[k]
    # for k in ckpt:
    #     if k not in keys:
    #         ckpt_new[k] = ckpt[k]
    # model.load_state_dict(ckpt_new, strict=False)
    return initial_epoch, model

def load_saved_model_diff(saved_path, model, finetune_flag=False):
    """
    Load saved model, model and checkpoint may not be totally same.

    Parameters
    __________
    saved_path : str
        model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    if file_list:
        assert len(file_list) == 1
        print("resuming best validation model at epoch %d" % \
                eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
        trained_model_dict = torch.load(file_list[0] , map_location='cpu')

        # TODO: uncomment lines below to introduce v2xvit v2-based pretrained model
        # old_trained_model_dict = torch.load(file_list[0] , map_location='cpu')
        # trained_model_dict = {}
        # for key, value in old_trained_model_dict.items():
        #     if 'fusion_net.fusion_net.' in key:
        #         new_key = key.replace('fusion_net.fusion_net.', 'fusion_net.')
        #     else:
        #         new_key = key
        #     trained_model_dict[new_key] = value

        # TODO: uncomment lines below to introduce disconet v2-based pretrained model
        # old_trained_model_dict = torch.load(file_list[0] , map_location='cpu')
        # trained_model_dict = {}
        # for key, value in old_trained_model_dict.items():
        #     if 'fusion_net.pixel_weight_layer.' in key:
        #         new_key = key.replace('fusion_net.pixel_weight_layer.', 'pixel_weight_layer.')
        #     else:
        #         new_key = key
        #     trained_model_dict[new_key] = value

        ######## finetune header
        # if finetune_flag:
        # finetune_split_name = ['cls_head.weight', 'cls_head.bias', 'reg_head.weight', 'reg_head.bias']
        # for k in finetune_split_name:
        #     trained_model_dict.update({
        #         'fused_'+k: trained_model_dict[k]})
    #############################

        diff_keys = {k:v for k, v in trained_model_dict.items() if k not in model.state_dict()}
        if diff_keys:
            print(f"!!! Trained model has keys: {diff_keys.keys()}, \
                which are not in the model you have created!!!")
        diff_keys = {k:v for k, v in model.state_dict().items() if k not in trained_model_dict.keys()}
        if diff_keys:
            print(f"!!! Created model has keys: {diff_keys.keys()}, \
                which are not in the model you have trained!!!")
        model.load_state_dict(trained_model_dict, strict=False)
        return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        trained_model_dict = torch.load(\
            os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch), map_location='cpu')
        diff_keys = {k:v for k, v in trained_model_dict.items() if k not in model.state_dict()}
        if diff_keys:
            print(f"!!! Trained model has keys: {diff_keys.keys()}, \
                which are not in the model you have created!!!")
        model.load_state_dict(trained_model_dict, strict=False)

    return initial_epoch, model


def load_two_parts_model(saved_path, model):
    """
    This function is used for intermediate+flow method, we should load two parts of pretrained model:
    1. single detection model
    2. fused detection head: self.cls_head_fused & self.reg_head_fused
    """
    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    assert len(file_list) == 1
    print("loading pretrained single detection model at epoch %d" % \
            eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
    trained_single_model_dict = torch.load(file_list[0], map_location='cpu')

    fused_model_path = '/DB/data/sizhewei/logs/where2comm_irr_no_shift_pretrain/net_epoch_bestval_at18.pth'
    trained_fused_model_dict = torch.load(fused_model_path, map_location='cpu')
    
    final_model_stat_dict = model.state_dict()
    
    state_dict = {k:v for k,v in trained_single_model_dict.items() if k in final_model_stat_dict.keys()}    
    state_dict.update({
        'cls_head_fused.weight': trained_fused_model_dict['cls_head.weight'], 
        'cls_head_fused.bias': trained_fused_model_dict['cls_head.bias'], 
        'reg_head_fused.weight': trained_fused_model_dict['reg_head.weight'], 
        'reg_head_fused.bias': trained_fused_model_dict['reg_head.bias']
    })

    final_model_stat_dict.update(state_dict)

    model.load_state_dict(final_model_stat_dict, strict=False)
    return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model


def setup_train(hypes, log_path="/DB/data/sizhewei"):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = model_name
    # folder_name 需要包含：
    # model_name, time_delay, num_sweep, batch_size, 
    # if 'time_delay' in hypes:
    #     folder_name = folder_name + '_d_' + str(hypes['time_delay'])
    # else:
    #     folder_name = folder_name + '_d_' + str(0)
    # if 'num_sweep_frames' in hypes:
    #     folder_name = folder_name + '_swps_' + str(hypes['num_sweep_frames'])
    # else:
    #     folder_name = folder_name + '_swps_' + str(1)
    # folder_name = folder_name + '_bs_' + str(hypes['train_params']['batch_size'])
    # model_hypes = hypes['model']['args']
    # if 'base_bev_backbone' in model_hypes and model_hypes['base_bev_backbone']['resnet']:
    #     folder_name = folder_name + '_w_resnet'
    # else:
    #     folder_name = folder_name + '_wo_resnet'
    # if 'rain_model' in model_hypes and  model_hypes['rain_model']['multi_scale']:
    #     folder_name = folder_name + '_w_multiscale'
    # else:
    #     folder_name = folder_name + '_wo_multiscale'
    # if 'exp_name' in hypes and len(str(hypes['exp_name']))>0:
    #     folder_name = folder_name + '_' + str(hypes['exp_name'])
    curr_time = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = folder_name + curr_time
    # current_path = os.path.dirname(__file__)
    current_path = Path(log_path)
    current_path = os.path.join(current_path, 'logs/')

    full_path = os.path.join(current_path, folder_name)
    print("full path is: ", full_path)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    # if not os.path.exists(full_path):
    #     os.makedirs(full_path)
    #     # save the yaml file
    #     save_name = os.path.join(full_path, 'config.yaml')
    #     with open(save_name, 'w') as outfile:
    #         yaml.dump(hypes, outfile)

        backup_script(full_path)

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model, is_pre_trained=False):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    #####################################################################
    if is_pre_trained:
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = model.parameters()
    ######################################################################
    if 'args' in method_dict:
        return optimizer_method(params,
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(params,
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, init_epoch=None):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0
    

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    for _ in range(last_epoch):
        scheduler.step()

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):        
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or isinstance(inputs, numpy.int64):  
                # AttributeError: 'numpy.int64' object has no attribute 'to'
                # sizhewei @ 2022/10/04
                # 添加类型 numpy.int64 数据集中的 sample_idx 是此类型, numpy.ndarry 数据集中 time_interval 是此类型
            return inputs
        return inputs.to(device)
