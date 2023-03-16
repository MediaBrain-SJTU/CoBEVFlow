# train | where2comm 
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_multi_sweep.py --hypes_yaml /DB/rhome/sizhewei/percp/OpenCOOD/opencood/hypes_yaml/opv2v/npj/opv2v_irr_past_where2comm_max_multiscale_resnet.yaml #--model_dir /remote-home/share/sizhewei/logs/point_pillar_single_pretrain

# inference | where2comm
# CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_irregular.py --model_dir /DB/data/sizhewei/logs/opv2v_npj_where2comm_max_multiscale_resnet_d_0_swps_1_bs_2_wo_resnet_wo_multiscale_2023_02_14_11_48_00 --fusion_method intermediate

# fine tune | where2comm(single to fused)
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_multi_sweep.py --hypes_yaml /DB/rhome/sizhewei/percp/OpenCOOD/opencood/hypes_yaml/opv2v/npj/opv2v_irr_past_where2comm_max_multiscale_resnet.yaml --model_dir /DB/data/sizhewei/logs/point_pillar_single_trained_fixed_2_fused

# train | where2comm(fused, no multi-scale)
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_multi_sweep.py --hypes_yaml /DB/rhome/sizhewei/percp/OpenCOOD/opencood/hypes_yaml/opv2v/npj/opv2v_irr_past_where2comm_max_multiscale_resnet.yaml --model_dir /DB/data/sizhewei/logs/where2comm_womultiscale_fused_thre_000

# inference | where2comm | no fusion
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_irregular.py --model_dir  --fusion_method no