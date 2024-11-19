import os
import subprocess

# 遍历所有的序列
for i in [\
    '00_peanuts_dark',\
    '01_peanuts_light',\
    '02_rocket_earth_light',\
    '03_rocket_earth_dark',\
    '06_ziggy_and_fuzz',\
    '07_ziggy_and_fuzz_hdr',\
    '08_peanuts_running',\
    '09_ziggy_flying_pieces',\
    '11_all_characters'
      ]:
    p = subprocess.Popen("python demo_vio_eds.py" +\
    " --imagedir=/media/lfl-data2/EDS/%s/images_undistorted_calib1/"%i +\
    " --imagestamp=/media/lfl-data2/EDS/%s/images_timestamps.txt"%i +\
    " --imupath=/media/lfl-data2/EDS/%s/imu.csv"%i +\
    " --gtpath=/media/lfl-data2/EDS/%s/stamped_groundtruth.txt"%i +\
    " --resultpath=results/EDS%s.txt"%i +\
    " --result_pdf=results/EDS%s.pdf"%i +\
    " --weights=/home/gwp/DBA-Fusion/droid.pth" +\
    " --calib=/media/lfl-data2/EDS/%s/calib_undist_calib1_rgb.txt"%i +\
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
