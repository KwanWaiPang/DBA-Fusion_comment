import os
import subprocess

# 遍历所有的序列
for i in [\
    'indoor_45_2_davis_with_gt',\
    'indoor_45_4_davis_with_gt',\
    'indoor_45_9_davis_with_gt',\
      ]:
    p = subprocess.Popen("python demo_vio_fpv_indoor_45.py" +\
    " --imagedir=/media/lfl-data2/UZH-FPV/%s/images_undistorted/"%i +\
    " --imagestamp=/media/lfl-data2/UZH-FPV/%s/raw_images_timestamps_us.txt"%i +\
    " --imupath=/media/lfl-data2/UZH-FPV/%s/imu.txt"%i +\
    " --gtpath=/media/lfl-data2/UZH-FPV/%s/raw_stamped_groundtruth_us.txt"%i +\
    " --calib=/media/lfl-data2/UZH-FPV/%s/calib_undist.txt"%i +\
    " --resultpath=results/fpv_indoor_%s.txt"%i +\
    " --result_pdf=results/fpv_indoor_%s.pdf"%i +\
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
