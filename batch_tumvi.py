import os
import subprocess

# 遍历所有的序列
for i in [\
    'corridor1'
    # 'outdoors1',\
    # 'outdoors2',\
    # 'outdoors3',\
    # 'outdoors4',\
    # 'outdoors5',\
    # 'outdoors6',\
    # 'outdoors7',\
    # 'outdoors8',\
    # 'magistrale1',\
    # 'magistrale2',\
    # 'magistrale3',\
    # 'magistrale4',\
    # 'magistrale5',\
    # 'magistrale6'
      ]:
    p = subprocess.Popen("python demo_vio_tumvi.py" +\
    " --imagedir=/home/gwp/DBA-Fusion/dataset/dataset-%s_512_16/mav0/cam0/data"%i +\
    " --imagestamp=/home/gwp/DBA-Fusion/dataset/dataset-%s_512_16/mav0/cam0/data.csv"%i +\
    " --imupath=/home/gwp/DBA-Fusion/dataset/dataset-%s_512_16/mav0/imu0/data.csv"%i +\
    " --gtpath=/home/gwp/DBA-Fusion/dataset/dataset-%s_512_16/dso/gt_imu.csv"%i +\
    " --enable_h5" +\
    " --h5path=/home/gwp/DBA-Fusion/dataset/dataset-%s_512_16.h5"%i +\
    " --resultpath=results/result_%s.txt"%i +\
    " --calib=calib/tumvi.txt" +\
    " --stride=4" +\
    " --max_factors=48" +\
    " --active_window=12" +\
    " --frontend_window=5" +\
    " --frontend_radius=2" +\
    " --frontend_nms=1" +\
    " --far_threshold=0.02" +\
    " --inac_range=3" +\
    " --visual_only=0" +\
    " --translation_threshold=0.2" +\
    " --mask_threshold=-1.0" +\
    " --skip_edge=[-4,-5,-6]" +\
    " --save_pkl" +\
    " --pklpath=results/%s.pkl"%i +\
    " --show_plot",shell=True)
    p.wait()
