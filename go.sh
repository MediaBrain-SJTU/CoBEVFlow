# train | where2comm 
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_multi_sweep.py --hypes_yaml /DB/rhome/sizhewei/percp/OpenCOOD/opencood/hypes_yaml/opv2v/npj/opv2v_irr_past_where2comm_max_multiscale_resnet.yaml #--model_dir /remote-home/share/sizhewei/logs/point_pillar_single_pretrain

# inference | where2comm
# CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_irregular.py --model_dir /DB/data/sizhewei/logs/opv2v_npj_where2comm_max_multiscale_resnet_d_0_swps_1_bs_2_wo_resnet_wo_multiscale_2023_02_14_11_48_00 --fusion_method intermediate

# fine tune | where2comm(single to fused)
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_multi_sweep.py --hypes_yaml /DB/rhome/sizhewei/percp/OpenCOOD/opencood/hypes_yaml/opv2v/npj/opv2v_irr_past_where2comm_max_multiscale_resnet.yaml --model_dir /DB/data/sizhewei/logs/point_pillar_single_trained_fixed_2_fused

# train | where2comm(fused, no multi-scale)
# CUDA_VISIBLE_DEVICES=4 python opencood/tools/train_multi_sweep.py --hypes_yaml /DB/rhome/sizhewei/percp/OpenCOOD/opencood/hypes_yaml/opv2v/npj/opv2v_irr_past_where2comm_max_multiscale_resnet.yaml --model_dir /DB/data/sizhewei/logs/where2comm_womultiscale_fused_thre_000

# inference | where2comm | no fusion
CUDA_VISIBLE_DEVICES=2 python opencood/tools/inference_irregular.py --model_dir /DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch --fusion_method no

# inference | where2comm | late fusion | w. delay
CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_irregular.py --model_dir /DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch --fusion_method late --note late_delay_test_sync --p 0.1

# inference | where2comm | feature flow | w. delay
CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_flow.py --model_dir /DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch --fusion_method intermediate_flow --config_suffix interFlow --note feature_flow_onlyReservedArea --p 0.0

# inference | where2comm | box flow | w. delay
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_flow.py --model_dir /DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch --fusion_method late_flow --config_suffix lateFlow --note lateFlow_new --p 0.1

# train | v2vnet | 32ch
CUDA_VISIBLE_DEVICES=5 python opencood/tools/train_multi_sweep.py --hypes_yaml opencood/hypes_yaml/opv2v/npj/opv2v_v2vnet.yaml --model_dir /DB/data/sizhewei/logs/OPV2V_v2vnet_32ch

# inference | v2vnet | 32ch
CUDA_VISIBLE_DEVICES=6 python opencood/tools/inference_irregular.py --model_dir /DB/data/sizhewei/logs/OPV2V_v2vnet_32ch --fusion_method intermediate --note v2vnet_32ch --p 0.0

CUDA_VISIBLE_DEVICES=6 python opencood/tools/train_multi_sweep.py --hypes_yaml opencood/hypes_yaml/opv2v/npj/opv2v_irr_point_pillar_where2comm_flow_module.yaml --pretrained_path /DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/net_epoch_bestval_at16.pth --model_dir /DB/data/sizhewei/logs/where2comm_flow_debug

# train | where2comm + flow_module
CUDA_VISIBLE_DEVICES=3 python opencood/tools/train_multi_sweep.py --hypes_yaml opencood/hypes_yaml/opv2v/npj/opv2v_irr_point_pillar_where2comm_flow_module.yaml --pretrained_path /DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/net_epoch_bestval_at16.pth --model_dir /DB/data/sizhewei/logs/where2comm_flow_module_design_2_addnew_delay_300

# train | where2comm + flow + regular
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python opencood/tools/train_multi_sweep.py --hypes_yaml opencood/hypes_yaml/opv2v/npj/opv2v_irr_point_pillar_where2comm_flow_regular.yaml --pretrained_path /DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/net_epoch_bestval_at16.pth 