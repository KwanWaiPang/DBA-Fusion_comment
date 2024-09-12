import os
import subprocess

# 遍历所有的序列
for i in [\
    'vicon_aggressive_hdr',\
    'vicon_dark1',\
    'vicon_dark2',\
    'vicon_darktolight1',\
    'vicon_darktolight2',\
    'vicon_hdr1',\
    'vicon_hdr2',\
    'vicon_hdr3',\
    'vicon_hdr4',\
    'vicon_lighttodark1',\
    'vicon_lighttodark2'
      ]:
    p = subprocess.Popen("python demo_vio_davis240c.py" +\
    " --imagedir=/media/lfl-data2/davis240c/%s/images_undistorted_left/"%i +\
    " --imagestamp=/media/lfl-data2/davis240c/%s/raw_tss_imgs_ns_left.txt"%i +\
    " --imupath=/media/lfl-data2/davis240c/%s/imu_data.csv"%i +\
    " --gtpath=/media/lfl-data2/davis240c/%s/raw_gt_stamped_ns_left.txt"%i +\
    " --resultpath=results/davis240c_%s.txt"%i +\
    " --result_pdf=results/davis240c_%s.pdf"%i +\
    " --weights=/home/gwp/DBA-Fusion/droid.pth" +\
    " --calib=/media/lfl-data2/davis240c/%s/calib_undist_left.txt"%i +\
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
