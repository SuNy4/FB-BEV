# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117  -f https://download.pytorch.org/whl/torch_stable.html
# Recommended 2.0.0> torch >= 1.12
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.6.0
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.24.0
pip install mmsegmentation==0.24.0
```

**e. Install FB-OCC from source code.**
```shell
git clone https://github.com/NVlabs/FB-BEV.git
cd FB-BEV
pip install -e .
# python setup.py install
```
**f. Check package version.**
```shell
# Use pip list to check packages
numpy==1.23.5
numba==0.53.0
spconv==2.3.6
ipython==8.12.0
yapf==0.40.1
opencv-python==4.9.0.80
open3d==0.18.0
```

**g. Prepare pretrained models.**
```shell
cd FB-BEV
mkdir ckpts
```