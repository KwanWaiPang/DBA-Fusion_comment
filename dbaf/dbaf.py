import torch
import lietorch
import numpy as np
from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from dbaf_frontend import DBAFusionFrontend
from collections import OrderedDict
from torch.multiprocessing import Process

from lietorch import SE3
import geom.projective_ops as pops
import droid_backends
import pickle

# 处理的整个类
class DBAFusion:
    def __init__(self, args):
        super(DBAFusion, self).__init__()
        self.load_weights(args.weights) # （导入网络的权重，并初始化DroidNet）load DroidNet weights
        self.args = args

        # （此类应该是用于存放所有的数据的）store images, depth, poses, intrinsics (shared between processes)
        # args.stereo设置为false的，args.upsample也是false。其余的就是图像的尺寸，缓冲区大小，是否保存pkl文件
        self.video = DepthVideo(args.image_size, args.buffer, save_pkl = args.save_pkl, stereo=args.stereo, upsample=args.upsample)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)#传入的参数为网络、数据、添加关键帧的阈值

        # frontend process（视觉的local BA，应该可以理解为用网络构建残差约束）
        self.frontend = DBAFusionFrontend(self.net, self.video, self.args)#传入的参数为网络、数据、args（参数）

        self.pklpath = args.pklpath #保存的pkl文件的路径
        self.upsample = args.upsample #参数参数是否进行上采样

    # 导入网络的权重（gwp_TODO 这部分后续修改是关键）
    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        # 导入权重前先初始化了Droid-SLAM的网络
        self.net = DroidNet()#初始化网络
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map （进行tracking） """

        with torch.no_grad():# 不进行梯度计算
            # check there is enough motion（传入的为图片、时间和内参，计算是否有足够的motion，同时做了特征的提取并更新self.video中数据）
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()#应该是调用__call__函数

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """
        del self.frontend

    # 保存结果
    def save_vis_easy(self):
        mcameras = {} # 保存的相机位姿
        mpoints = {} # 保存的点云
        mstamps = {} # 保存的时间戳
        with torch.no_grad(): # 不进行梯度计算
            dirty_index = torch.arange(0,self.video.count_save,device='cuda')

            stamps= torch.index_select(self.video.tstamp_save, 0 ,dirty_index)
            poses=  torch.index_select( self.video.poses_save, 0 ,dirty_index)
            disps=  torch.index_select( self.video.disps_save, 0 ,dirty_index)
            images = torch.index_select( self.video.images_save, 0 ,dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()
            points = droid_backends.iproj(SE3(poses).inv().data, disps, self.video.intrinsics[0]).cpu()
            thresh = 0.4 * torch.ones_like(disps.mean(dim=[1,2])) / 4.0  * (1.0 / torch.median(disps.mean(dim=[1,2])))
            # thresh = 0.4 * torch.ones_like(disps.mean(dim=[1,2])) 
            count = droid_backends.depth_filter(
                self.video.poses_save, self.video.disps_save, self.video.intrinsics[0], dirty_index, thresh)

            count = count.cpu()
            disps = disps.cpu()

            if self.upsample:
                disps_up=  torch.index_select( self.video.disps_up_save, 0 ,dirty_index)
                disps_up = disps_up.cpu()

            masks = ((count >= 1) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()
                mcameras[ix] = pose
                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                stamp = stamps[i].cpu()
                if self.upsample:
                    mpoints[ix] = {'pts':pts,'clr':clr,'disp':disps[i].cpu().numpy(),'disps_up':disps_up[i].cpu().numpy(),'rgb':images[i].cpu().numpy()}
                else:
                    mpoints[ix] = {'pts':pts,'clr':clr,'disp':disps[i].cpu().numpy(),'rgb':images[i].cpu().numpy()}
                mstamps[ix] = stamp
        ddict = {'points':mpoints,'cameras':mcameras,'stamps':mstamps}
        f_save = open(self.pklpath, 'wb')
        pickle.dump(ddict,f_save) 

        mcameras = {}
        mpoints = {}
        mstamps = {}
        with torch.no_grad():
            dirty_index = torch.arange(0,self.video.count_save,device='cuda')

            stamps= torch.index_select(self.video.tstamp_save, 0 ,dirty_index)
            poses=  torch.index_select( self.video.poses_save, 0 ,dirty_index)
            disps=  torch.index_select( self.video.disps_save, 0 ,dirty_index)
            images = torch.index_select( self.video.images_save, 0 ,dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()
            points = droid_backends.iproj(SE3(poses).inv().data, disps, self.video.intrinsics[0]).cpu()
            thresh = 0.4 * torch.ones_like(disps.mean(dim=[1,2]))
            count = droid_backends.depth_filter(
                self.video.poses_save, self.video.disps_save, self.video.intrinsics[0], dirty_index, thresh)

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 0) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()
                mcameras[ix] = pose
                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                stamp = stamps[i].cpu()
                mpoints[ix] = {'pts':pts,'clr':clr,'disp':disps[i].cpu().numpy(),'rgb':images[i].cpu().numpy()}
                mstamps[ix] = stamp
        ddict = {'points':mpoints,'cameras':mcameras,'stamps':mstamps}
        f_save = open(self.pklpath.split('.')[0] + '_raw.pkl', 'wb')
        pickle.dump(ddict,f_save) 
