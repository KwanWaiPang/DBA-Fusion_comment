import os
import subprocess

# 遍历所有的序列
for i in [\
    'vicon_hdr1',\
    'vicon_hdr2',\
    'vicon_hdr3',\
    'vicon_hdr4',\
    'vicon_darktolight1',\
    'vicon_darktolight2',\
    'vicon_lighttodark1',\
    'vicon_lighttodark2',\
    'vicon_dark1',\
    'vicon_dark2',\
    'vicon_aggressive_hdr'
      ]:
    p = subprocess.Popen("python demo_vio_mono_hku.py" +\
    " --imagedir=/media/lfl-data2/Mono_HKU/%s/images_undistorted_davis346/"%i +\
    " --imagestamp=/media/lfl-data2/Mono_HKU/%s/raw_tss_imgs_ns_davis346.txt"%i +\
    " --imupath=/media/lfl-data2/Mono_HKU/%s/davis346_imu_data.csv"%i +\
    " --gtpath=/media/lfl-data2/Mono_HKU/%s/raw_gt_stamped_us.txt"%i +\
    " --resultpath=results/mono_hku_%s.txt"%i +\
    " --result_pdf=results/mono_hku_%s.pdf"%i +\
    " --weights=/home/gwp/DBA-Fusion/droid.pth" +\
    " --calib=/media/lfl-data2/Mono_HKU/%s/calib_undist_davis346.txt"%i +\
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
