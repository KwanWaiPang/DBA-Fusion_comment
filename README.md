[comment]: <> (# DBA-Fusion)

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> DBA-Fusion (复现及中文注释版~仅供个人学习记录用)
  </h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://arxiv.org/pdf/2403.13714">Paper</a> 
  | <a href="https://github.com/GREAT-WHU/DBA-Fusion">Original Github Page</a>
  </h3>
  <div align="center"></div>

# 配置过程记录
* 下载代码
~~~
git clone --recurse-submodules https://github.com/GREAT-WHU/DBA-Fusion.git
~~~
* 创建conda环境
~~~
conda create -n dbaf python=3.10.11
conda activate dbaf

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install gdown tqdm numpy==1.25.0 numpy-quaternion==2022.4.3 opencv-python==4.7.0.72 scipy pyparsing matplotlib h5py 
pip install evo --upgrade --no-binary evo
pip install open3d # optional for visualization (这是可视化的，应该只有MobaXterm可用)
~~~
* 安装GTSAM（作者在原版GTSAM的基础上做了一些基于python的改进，已把此代码push到thirdparty文件内）
~~~
cd thirdparty
git clone https://github.com/ZhouTangtang/gtsam.git
cd gtsam
~~~
