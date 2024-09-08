import sys
sys.path.append('dbaf')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import argparse
from dbaf import DBAFusion

import h5py
import pickle
import re
import math
import quaternion
import gtsam

import matplotlib.pyplot as plt
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
import evo.main_ape as main_ape

# for the sever useer, we need to set the backend from "TkAgg" to "Agg"
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'
from evo.tools import plot

def plot_trajectory(pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True):
    assert isinstance(pred_traj, PoseTrajectory3D)

    if gt_traj is not None:
        assert isinstance(gt_traj, PoseTrajectory3D)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xz # ideal for planar movement
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

# 打包图像数据：时间戳、图像、内参
def image_stream(imagedir, imagestamp, enable_h5, h5path, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    # 获取内参矩阵
    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    Kn = np.eye(3)
    Kn[0,0] = fx 
    Kn[0,2] = cx 
    Kn[1,1] = fy 
    Kn[1,2] = cy

    if not enable_h5:#如果不是h5文件
        image_list = sorted(os.listdir(imagedir))[::stride]#获取图像列表，并按名字排序，每隔stride取一个
        # image_stamps = np.loadtxt(imagestamp,str,delimiter=',')#读取时间戳
        image_stamps= np.loadtxt(imagestamp)[::stride]#读取时间戳（注意要跟上面一样跳时间戳）
        # image_dict = dict(zip(image_stamps[:,1],image_stamps[:,0]))
        for t, imfile in enumerate(image_list):
            image = cv2.imread(os.path.join(imagedir, imfile))

            if len(calib) > 4:
                m1, m2 = cv2.fisheye.initUndistortRectifyMap(K,calib[4:],np.eye(3),Kn,(512,512),cv2.CV_32FC1)
                image = cv2.remap(image, m1, m2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # tt = float(image_dict[imfile]) /1e6 #时间戳,转换为秒，读入的时间戳是微秒
            tt = image_stamps[t] /1e9 #时间戳,转换为秒，读入的时间戳是纳秒，t是索引

            h0, w0, _ = image.shape
            # h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))#按比例缩放（此处应该就是统一了尺寸图像为384*512了？）
            # w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            # image = cv2.resize(image, (w1, h1))
            # image = image[:h1-h1%8, :w1-w1%8]
            image=image[:h0-h0%8, :w0-w0%8]#裁剪图像，使得图像的长宽都是8的倍数
            image = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy ])
            # intrinsics[0::2] *= (w1 / w0)
            # intrinsics[1::2] *= (h1 / h0)

            yield tt, image[None], intrinsics
    else:
        ccount = 0
        h5_f = h5py.File(h5path,'r')
        all_keys = sorted(list(h5_f.keys()))
        for key in all_keys:
            ccount += 1
            yield pickle.loads(np.array(h5_f[key]))

# 主函数
if __name__ == '__main__':

    print(f'\033[0;31;42m testing davis240c!!! \033[0m')

    # 检查GPU是否可用，并打印GPU信息
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    # 设置参数，原文通过执行batch_tumvi.py文件来设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory") #图像数据的路径
    parser.add_argument("--imagestamp", type=str, help="") #图像时间戳
    parser.add_argument("--imupath", type=str, help="")# imu数据
    parser.add_argument("--gtpath", type=str, help="") #参考轨迹（真值）

    #当数据集较大时，可以将数据集存储为h5文件，加快读取速度？
    parser.add_argument("--enable_h5", action="store_true", help="") #存在的时候就是真
    parser.add_argument("--h5path", type=str, help="")
    parser.add_argument("--resultpath", type=str, default="result.txt", help="")

    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth") #权重自动导入（直接用DROID-SLAM训练好的权重）
    parser.add_argument("--buffer", type=int, default=80)
    parser.add_argument("--image_size", default=[240, 320])

    # 在CovisibleGraph中会用到参数。定义为48，即构建的最多的约束的边数
    parser.add_argument("--max_factors", type=int, default=48, help="maximum active edges (which determines the GPU memory usage)")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    
    # 在MotionFilter会用到这个参数，用于判断是否添加关键帧（检查运动是否大于阈值）
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")

    # 在DBAFusionFrontend会用到这个参数，要等待多少帧才开始优化
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=3.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--active_window", type=int, default=8, help="maximum frames involved in DBA")
    parser.add_argument("--inac_range", type=int, default=3, help="maximum inactive frames (whose flow wouldn't be updated) involved in DBA")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    # CovisibleGraph中会用到这个参数
    parser.add_argument("--upsample", action="store_true")

    parser.add_argument("--visual_only", type=int,default=0, help="wheter to disbale the IMU")
    parser.add_argument("--far_threshold", type=float, default=0.02, help="far pixels would be downweighted (unit: m^-1)")

    # 用于frondend的参数
    parser.add_argument("--translation_threshold", type=float, default=0.2, help="avoid the insertion of too close keyframes (unit: m)")
    # 用于CovisibleGraph的参数
    parser.add_argument("--mask_threshold", type=float, default=-1, help="downweight too close edges (unit: m)")

    parser.add_argument("--skip_edge", type = str, default ="[]", help="whether to add 'skip' edges in the graph (for example, [-4,-5,-6] relative to the oldest active frame)")
    parser.add_argument("--save_pkl", action="store_true")
    parser.add_argument("--pklpath", default="result.pkl", help="path to saved reconstruction")
    parser.add_argument("--show_plot", action="store_true", help="plot the trajectory during running")

    parser.add_argument("--evaluate_flag", action="store_true", help="plot the trajectory during running")#是否进行评估
    parser.add_argument("--result_pdf", type=str, default="result.pdf", help="")#结果pdf文件
    
    args = parser.parse_args()
    args.skip_edge = eval(args.skip_edge)

    args.stereo = False
    dbaf = None
    torch.multiprocessing.set_start_method('spawn')

    # 使用三重引号 """ 来表示多行字符串
    print(f"""imagedir: {args.imagedir} \n
            imagestamp: {args.imagestamp} \n
            imupath: {args.imupath} \n
            gtpath: {args.gtpath} \n
            resultpath: {args.resultpath} \n
            calib: {args.calib} \n
            result_pdf: {args.result_pdf} \n""")


    """ Load reference trajectory (for visualization) """
    all_gt ={}
    try:
        fp = open(args.gtpath,'rt') #打开真值轨迹文件
        while True:
            line = fp.readline().strip() # 读取文件中的一行并去掉首尾的空白字符
            if line == '':break  # 如果读到的是空行，表示文件结束，跳出循环
            if line[0] == '#' : continue # 如果行的第一个字符是 #，表示这是注释行，跳过
            line = re.sub('\s\s+',' ',line) # 用单个空格替换多个连续的空白字符
            elem = line.split(',')  # 按逗号分隔行中的各个元素，得到一个列表
            # 第一个元素是时间戳
            sod = float(elem[0])/1e6 #时间戳（转换为秒，读入的时候是微妙）
            if sod not in all_gt.keys(): # 如果时间戳还没有记录在字典中
                all_gt[sod] ={} # 在字典中创建一个新的条目

            # 先获取四元数，然后把四元数转换为旋转矩阵
            R = quaternion.as_rotation_matrix(quaternion.from_float_array([float(elem[4]),\
                                                                           float(elem[5]),\
                                                                           float(elem[6]),\
                                                                           float(elem[7])]))
            TTT = np.eye(4,4)  # 创建一个4x4的单位矩阵
            TTT[0:3,0:3] = R  # 将旋转矩阵赋值给单位矩阵的左上角3x3子矩阵
            # 第四列的前3行，分别是x,y,z（ 将位置数据赋值给单位矩阵的前三行的第四列）
            TTT[0:3,3] = np.array([ float(elem[1]), float(elem[2]), float(elem[3])])
            # 将构建的转换矩阵存储在字典对应的时间戳条目中
            all_gt[sod]['T'] = TTT
        all_gt_keys =sorted(all_gt.keys()) # 获取所有的时间戳，并按升序排序
        fp.close() # 关闭文件
    except:
        pass

    """ Load IMU data """
    all_imu = np.loadtxt(args.imupath,delimiter=',')
    all_imu[:,0] /= 1e9 #时间戳，转换为秒
    all_imu[:,1:4] *= 180/math.pi #角速度，转换为度
    # 线加速度不变
    #     
    tstamps = []

    """ Load images """
    clahe = cv2.createCLAHE(2.0,tileGridSize=(8, 8)) #读入图片时先进行直方图均衡化（初始化直方图均衡化的类）
    # 通过image_stream函数获取数据，并通过tpdm来显示进度
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.imagestamp, args.enable_h5,\
                                                     args.h5path, args.calib, args.stride)):
        # 对图像进行直方图均衡化
        mm = clahe.apply(image[0][0].numpy())
        # mm[None]：假设 mm 是一个 2D 张量，形状为 (H, W)，那么 mm[None] 的形状将变为 (1, H, W)
        image[0] = torch.tensor(mm[None].repeat(3,0))#.repeat(3,0)， 将形状从 (1, H, W) 变为 (3, H, W)，
        if args.show_plot:
            show_image(image[0]) #将图像通过cv2.imshow显示出来
        if dbaf is None:
            args.image_size = [image.shape[2], image.shape[3]]
            # 根据参数初始化DBAFusion类
            dbaf = DBAFusion(args)
            # 设置前端中的数据
            dbaf.frontend.all_imu = all_imu #设置IMU数据
            dbaf.frontend.all_gnss = [] #设置GNSS数据为空
            dbaf.frontend.all_odo = [] #设置里程计数据为空
            # 设置图像时间戳
            # dbaf.frontend.all_stamp  = np.loadtxt(args.imagestamp,str,delimiter=',')
            dbaf.frontend.all_stamp  = np.loadtxt(args.imagestamp)  
            # dbaf.frontend.all_stamp = dbaf.frontend.all_stamp[:,0].astype(np.float64)[None].transpose(1,0)/1e9 #转换为秒
            dbaf.frontend.all_stamp = dbaf.frontend.all_stamp[None].transpose(1,0)/1e9 #提取第一列转换为秒（纳秒）
            if len(all_gt) > 0:
                # 设置真值数据
                dbaf.frontend.all_gt = all_gt #时间戳+位姿
                # 将all_gt排序后的
                dbaf.frontend.all_gt_keys = all_gt_keys #时间戳
            
            # 设置video数据包中的数据
            # IMU-Camera Extrinsics（设置IMU与camera的外参）
            dbaf.video.Ti1c = np.array(
            [0.9999, -0.0122,  0.0063, 0.0067,      
             0.0121,  0.9998,  0.0093, 0.0007,
            -0.0064, -0.0092,  0.9999, 0.0342,
             0.0, 0.0, 0.0, 1.0]).reshape([4,4])
            # dbaf.video.Ti1c = np.linalg.inv(dbaf.video.Ti1c) #矩阵求逆
            dbaf.video.Tbc = gtsam.Pose3(dbaf.video.Ti1c) #将矩阵转换为gtsam.Pose3类型
            
            # IMU parameters（IMU一些参数的设置）
            # 通过state（MultiSensorState类）设置优化器中的状态
            dbaf.video.state.set_imu_params((np.array([ 0.0003924 * 25,0.000205689024915 * 25, 0.004905 * 10, 0.000001454441043 * 5000])*1.0).tolist())
            # 设置video数据包中初始化的pose以及bias
            dbaf.video.init_pose_sigma = np.array([0.1, 0.1, 0.0001, 0.0001,0.0001,0.0001])
            dbaf.video.init_bias_sigma = np.array([1.0,1.0,1.0, 1.0,1.0,1.0])
            # 设置前端中的参数
            dbaf.frontend.translation_threshold = args.translation_threshold #避免插入太近的关键帧
            dbaf.frontend.graph.mask_threshold  = args.mask_threshold #downweight too close edges，太靠近的边的权重减小

        # 进行tracking（传入的为时间、图片、内参）
        dbaf.track(t, image, intrinsics=intrinsics)

    if args.save_pkl:#保存结果
        dbaf.save_vis_easy()

    dbaf.terminate()

    if args.evaluate_flag:
        #获取真值轨迹

        traj_ref = file_interface.read_tum_trajectory_file(args.gtpath)
        gtlentraj = traj_ref.get_infos()["path length (m)"]#获取轨迹长度

        #轨迹的时间戳需要以秒为单位(原本是微妙)
        traj_ref.timestamps = traj_ref.timestamps / 1e6

        #进行验证
        est_file=args.resultpath#获取结果文件（注意保存的时间应该已经转换为秒了~）
        traj_est = file_interface.read_tum_trajectory_file(est_file)

        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)#同步轨迹

        result = main_ape.ape(traj_ref, traj_est, 
                pose_relation=PoseRelation.translation_part, align=True,n_to_align=1000, correct_scale=True)#注意n_to_align=1000
        
        print(f"\033[31m EVO结果：{result}\033[0m");
        MPE = result.stats["mean"] / gtlentraj * 100 #注意单位为%
        print(f"MPE is {MPE:.02f}")    
        ate_score = result.stats["rmse"] #注意单位为m

        res_str = f"\nATE[m]: {ate_score:.03f} | MPE[%/m]: {MPE:.02f}"

        pdfname = args.result_pdf
        plot_trajectory(traj_est, traj_ref, f"{res_str})",
                        pdfname, align=True, correct_scale=True)
        gwp_debug=1;
