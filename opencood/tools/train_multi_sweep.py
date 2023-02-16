# -*- coding: utf-8 -*-
# Author: Runsheng Xu
# Modified: Yifan Lu
# Support DiscoNet knowledge distillation
# Support automatically evaluating after training finished.

import argparse
import os
import statistics
import sys
sys.path.append(os.getcwd())
import time

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import importlib
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from tqdm import tqdm
from tqdm.contrib import tenumerate
from tqdm.auto import trange


run_test = True
from opencood.data_utils.datasets.intermediate_fusion_dataset_opv2v_irregular import illegal_path_list
compensation = False 

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('### Dataset Building ... ###')
    start_time = time.time()
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=8, # TODO: num_workers改回8
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True, #True, # TODO: shuffle改为True
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,  # TODO: num_workers改回8
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,   # TODO: shuffle改为True
                            pin_memory=True,
                            drop_last=True)
    end_time = time.time()
    print("=== Time consumed: %.1f minutes. ===" % ((end_time - start_time)/60))
    start_time = time.time()
    # for debug use:
    #############################################################################
    # init_epoch = 0
    # epoches = hypes['train_params']['epoches']
    # for epoch in range(init_epoch, max(epoches, init_epoch)):
    #     for i, batch_data in enumerate(train_loader):
    #         if batch_data is None:
    #             continue
    #############################################################################
    
    print('### Creating Model ... ###')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1    
    
    #############################################################################
    # load pre-train model for single view
    is_pre_trained = False
    if 'is_single_pre_trained' in hypes and hypes['is_single_pre_trained']['pre_train_flag']:
        is_pre_trained = True
    if is_pre_trained:
        pretrain_path = hypes['is_single_pre_trained']['pre_train_path']
        initial_epoch = hypes['is_single_pre_trained']['pre_train_epoch']
        pre_train_model = torch.load(os.path.join(pretrain_path, 'net_epoch%d.pth' % initial_epoch))
        model.load_state_dict(pre_train_model, strict=False)
        print("### Pre-trained point pillar {} loaded successfully! ###".format(os.path.join(pretrain_path, 'net_epoch%d.pth' % initial_epoch)))
        fix = hypes['is_single_pre_trained']['pre_train_fix']
        if fix:
            for name, value in model.named_parameters():
                if name in pre_train_model:
                    value.requires_grad = False
    #############################################################################

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # define the loss # TODO
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model, is_pre_trained)
    # lr scheduler setup
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # is_fix = False # TODO: pretrain 记得删除
    # pretrain_path = "/DB/data/sizhewei/logs/opv2v_npj_raindrop_attn_d_0_swps_1_bs_2_w_resnet_w_multiscale_2023_02_04_22_46_26_pretrain"
    # initial_epoch = 17
    # if is_fix:
    #     pre_train_model = torch.load(os.path.join(pretrain_path, 'net_epoch%d.pth' % initial_epoch))
    #     model.load_state_dict(pre_train_model, strict=False)
    #     print("### Pre-trained loaded successfully! ###".format(os.path.join(opt.model_dir, 'net_epoch%d.pth' % initial_epoch)))
    #     for name, value in model.named_parameters():
    #         if name == 'cls_head.weight' or name == 'cls_head.bias':
    #             continue # TODO: pretrain 记得删除
    #         if name in pre_train_model:
    #             value.requires_grad = False

    end_time = time.time()
    print("=== Time consumed: %.1f minutes. ===" % ((end_time - start_time)/60))

    # record training
    writer = SummaryWriter(saved_path)

    start_time = time.time()
    print('### Training start! ###')
    epoches = hypes['train_params']['epoches']
    proj_first = hypes['fusion']['args']['proj_first']
    # used to help schedule learning rate

    ############ For DiscoNet TODO: 用不到的code 可以删除 ##############
    if "kd_flag" in hypes.keys():
        kd_flag = True
        teacher_model_name = hypes['kd_flag']['teacher_model'] # point_pillar_disconet_teacher
        teacher_model_config = hypes['kd_flag']['teacher_model_config']
        teacher_checkpoint_path = hypes['kd_flag']['teacher_path']

        # import the model
        model_filename = "opencood.models." + teacher_model_name
        model_lib = importlib.import_module(model_filename)
        teacher_model_class = None
        target_model_name = teacher_model_name.replace('_', '')

        for name, cls in model_lib.__dict__.items():
            if name.lower() == target_model_name.lower():
                teacher_model_class = cls
        
        teacher_model = teacher_model_class(teacher_model_config)
        teacher_model.load_state_dict(torch.load(teacher_checkpoint_path), strict=False)
        
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        if torch.cuda.is_available():
                teacher_model.to(device)

        teacher_model.eval()

    else:
        kd_flag = False

    time_loaddata = 0
    time_todevice = 0
    time_training = 0
    time_lossandbp = 0
    sample_interval_all_epoch = 0
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        
        # start_time = time.time()
        # times = []
        sample_interval = 0
        i = 0
        for i, batch_data in enumerate(train_loader): 
            # if i==1: # TODO: debug 用
            #     break
            if batch_data is None:
                continue
            # start_time = time.time()
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            # case1 : late fusion train --> only ego needed,
            # and ego is (random) selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well

            # end_time = time.time()
            # time_todevice += (end_time - start_time)
            # start_time = time.time()

            batch_data['ego']['epoch'] = epoch
            # sample_interval += batch_data['ego']['avg_sample_interval'] # debug use 打开
            ouput_dict = model(batch_data['ego'])

            # end_time = time.time()
            # time_training += (end_time - start_time)
            # start_time = time.time()

            if kd_flag:
                teacher_output_dict = teacher_model(batch_data['ego'])
                ouput_dict.update(teacher_output_dict)


            # first argument is always your output dictionary,
            # second argument is always your label dictionary.
            
            if compensation:
                final_loss = ouput_dict['recon_loss'] 
                detection_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            else:
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            if i%10 == 0:
                criterion.logging(epoch, i, len(train_loader), writer)
                # TODO: Uncomment for debug time compensation module
                if compensation:
                    curr_loss = criterion(ouput_dict, batch_data['ego']['label_dict'],mode='curr')
                    latency_loss = criterion(ouput_dict, batch_data['ego']['label_dict'],mode='latency')
                    print('curr_loss',curr_loss.item())
                    print('latency_loss',latency_loss.item())
                    print('recon_loss',ouput_dict['recon_loss'].item())

            # back-propagation
            final_loss.backward()
            optimizer.step()

            # end_time = time.time()
            # time_lossandbp += (end_time - start_time)
            # start_time = time.time()

            torch.cuda.empty_cache()
        
        sample_interval /= i # TODO: 打开
        sample_interval_all_epoch += sample_interval

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            end_time = time.time()
            print('### %d th epoch trained, start validation! Time consumed %.2f ###' % (epoch, (end_time - start_time)/60))
            with torch.no_grad():
                for i, batch_data in tenumerate(val_loader):
                    # TODO: debug use
                    # print(i)
                    # if i == 10:
                    #     break

                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    if kd_flag:
                        teacher_output_dict = teacher_model(batch_data['ego'])
                        ouput_dict.update(teacher_output_dict)

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
                # TODO: debug use
                # print(illegal_path_list)

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                        os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

    end_time = time.time()
    print("Time consumed: %.1f" % ((end_time - start_time)/60))
    sample_interval_all_epoch /= max(epoches, init_epoch) - init_epoch
    print("Avg sample interval of all epochs: %.2f" % sample_interval_all_epoch) 
    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference_multi_sweep.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
