import os
import subprocess

# 遍历所有的序列
for i in [\
    'indoor_flying1_data',\
    'indoor_flying2_data',\
    'indoor_flying3_data',\
    'indoor_flying4_data',\
      ]:
    p = subprocess.Popen("python demo_vio_mvsec.py" +\
    " --imagedir=/media/lfl-data2/MVSEC/%s/images_undistorted_left/"%i +\
    " --imagestamp=/media/lfl-data2/MVSEC/%s/raw_tss_imgs_ns_left.txt"%i +\
    " --imupath=/media/lfl-data2/MVSEC/%s/left_imu_data.csv"%i +\
    " --gtpath=/media/lfl-data2/MVSEC/%s/raw_gt_stamped_us.txt"%i +\
    " --calib=/media/lfl-data2/MVSEC/%s/calib_undist_left.txt"%i +\
    " --resultpath=results/mvsec_%s.txt"%i +\
    " --result_pdf=results/mvsec_%s.pdf"%i +\
    " --weights=/home/gwp/DBA-Fusion/droid.pth" +\
    " --stride=2" +\
    " --max_factors=48" +\
    " --active_window=12" +\
    " --frontend_window=5" +\
    " --frontend_radius=2" +\
    " --frontend_nms=1" +\
    " --far_threshold=0.02" +\
    " --inac_range=3" +\
    " --visual_only=0" +\
    " --translation_threshold=0.2" +\
    " --evaluate_flag" +\
    " --mask_threshold=-1.0",shell=True)
    # " --skip_edge=[-4,-5,-6]" +\
    # " --save_pkl" +\
    # " --pklpath=results/%s.pkl"%i +\
    # " --show_plot",shell=True)
    p.wait()
