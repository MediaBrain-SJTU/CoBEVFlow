rsync -rv --exclude-from=/DB/rhome/sizhewei/percp/OpenCOOD/exclude-file.txt /DB/rhome/sizhewei/percp/OpenCOOD/opencood seecsh@sylogin.hpc.sjtu.edu.cn:/dssg/home/acct-seecsh/seecsh/sizhewei/code/OpenCOOD/


rsync -rv --exclude-from=/DB/rhome/sizhewei/percp/OpenCOOD/exclude_file_checkpoints.txt /DB/data/sizhewei/logs/where2comm_max_multiscale_resnet_32ch/ seecsh@sylogin.hpc.sjtu.edu.cn:/dssg/home/acct-seecsh/seecsh/sizhewei/checkpoint/where2comm_max_multiscale_resnet_32ch/

# hypes_yaml
rsync -rv /DB/rhome/sizhewei/percp/OpenCOOD/opencood/hypes_yaml seecsh@sylogin.hpc.sjtu.edu.cn:/dssg/home/acct-seecsh/seecsh/sizhewei/code/OpenCOOD/opencood/hypes_yaml

rsync -rv /DB/rhome/sizhewei/pkg/pypcd seecsh@sylogin.hpc.sjtu.edu.cn:/dssg/home/acct-seecsh/seecsh/sizhewei/pkg/pypcd

# rsync from xhpang to my path
rsync -rv --exclude-from=/root/percp/exclude-file.txt /remote-home/share/xhpang/OpenCOODv2 /root/percp/OpenCOODv2

rsync -rv /remote-home/share/xhpang/OpenCOODv2/opencood/logs/flow_ob /root/percp/OpenCOODv2/OpenCOODv2/opencood/logs