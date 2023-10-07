## 1. Create env for CoBEVFlow
conda virtual env, environment path...
```bash
# create cobevflow env

# STEP 1.1
# preparation, use any conda env name you like, here I name it cobevflow
conda create --name cobevflow python=3.7 cmake=3.22.1
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install cudnn -c conda-forge
conda install boost

# STEP 1.2
# not need to conda install cudatoolkit
# but specify the PATH, CUDA_HOME, and LD_LIBRARY_PATH, using current cuda
# write it to ~/.bashrc, for example use Vim
vim ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda/bin:$CUDA_HOME
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Note: use your own path of conda virtual environment below, for example, use sizhewei instead of [$YourUserName], use cobevflow instead of [$YourEnvName].
# add head file search directories 
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/[$YourUserName]/anaconda3/envs/[$YourEnvName]/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/[$YourUserName]/anaconda3/envs/[$YourEnvName]/include
# add shared library searching directories
export LIBRARY_PATH=$LIBRARY_PATH:/[$YourUserName]/anaconda3/envs/[$YourEnvName]/lib
# add runtime library searching directories
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/[$YourUserName]/anaconda3/envs/[$YourEnvName]/lib

# STEP 1.3
# go out of Vim and activate it in current shell
source ~/.bashrc
# activate conda env
conda activate cobevflow
```

## 2. Install Spconv==1.2.1
```bash
# STEP 2.1
# clone spcon==1.2.1 
git clone https://github.com/traveller59/spconv.git 
cd spconv
git checkout v1.2.1
git submodule update --init --recursive

# STEP 2.2 
# compile
python setup.py bdist_wheel

# STEP 2.3
# install
cd ./dist
pip install spconv-1.2.1-cp37-cp37m-linux_x86_64.whl

# STEP 2.4
# check if is successfully installed
python 
import spconv
```

## 3. Install CoBEVFlow
```bash
git clone https://github.com/SizheWei/CoBEVFlow.git
cd CoBEVFlow
python setup.py develop
pip install -r requirements.txt

# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace 
# FPVRCNN dependency (optional)
python opencood/pcdet_utils/setup.py build_ext --inplace 
```

## 4.Install pypcd: required by DAIR-V2X LiDAR dataloader
Note that pypcd pip installing is not compatible with Python3. Therefore a modified version should be manually installed as followings.
```bash
# Go to another folder. Do not clone it within CoBEVFlow.
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
```

These steps are almost same as  https://opencood.readthedocs.io/en/latest/md_files/installation.html