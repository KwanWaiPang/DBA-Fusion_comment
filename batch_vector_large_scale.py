import os
import subprocess

# 遍历所有的序列
for i in [\
    'corridors_dolly1',\
    'corridors_walk1',\
    'school_dolly1',\
    'school_scooter1',\
    'units_dolly1',\
    'units_scooter1',\
      ]:
    p = subprocess.Popen("python demo_vio_vector_large_scale.py" +\
    " --imagedir=/media/lfl-data2/VECtor/%s/images_undistorted_left/"%i +\
    " --imagestamp=/media/lfl-data2/VECtor/%s/raw_tss_imgs_ns_left.txt"%i +\
    " --imupath=/media/lfl-data2/VECtor/%s/imu_data.csv"%i +\
    " --gtpath=/media/lfl-data2/VECtor/%s/%s.synced.gt.txt"%(i, i) +\
    " --calib=/media/lfl-data2/VECtor/%s/calib_undist_left.txt"%i +\
    " --resultpath=results/vector_%s.txt"%i +\
    " --result_pdf=results/vector_%s.pdf"%i +\
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
