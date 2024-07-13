import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock
import numpy as np

class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet #调用context network
        self.fnet = net.fnet #调用feature network
        self.update = net.update #调用update network

        self.video = video #所有的数据都存放在video中
        self.thresh = thresh #阈值，超过这个阈值就会添加关键帧
        self.device = device #默认为cuda:0

        self.count = 0 #计数器，应该是用于统计运动幅度小于阈值的帧的数目，目前好像没有用到

        # mean, std for image normalization（定义两个张量，分别表示均值和标准差，用于对图片进行归一化）
        # 而[:, None, None]就是将其扩展到三维。: 表示保持原有的第一个维度不变（即保留所有元素）。None 是新添加的两个维度。这相当于对原始张量进行升维操作。也就是每个通道的均值和标准差
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)#返回的context features分为两部分，一部分为tanh激活和另外一部分为relu激活
    
    @torch.cuda.amp.autocast(enabled=True)
    def context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)
    
    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)#去除第一个维度

    @torch.cuda.amp.autocast(enabled=True)
    def feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)
    
    @torch.cuda.amp.autocast(enabled=True)#自动混合精度
    @torch.no_grad()#不进行梯度计算
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main update operation - run on every frame in video """

        # 单位矩阵 pose
        Id = lietorch.SE3.Identity(1,).data.squeeze()
        # 图像的高和宽的8分之一（“//”为整除）gwp_TODO：注意后续可能要修改对应网络的维度
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images（将图像进行归一化）
        # None 在最前面增加一个新的维度，使得形状从 (C, H, W) 变为 (1, C, H, W)
        # [:, [2, 1, 0]] 通过索引操作重新排列通道顺序。原始顺序是 RGB，索引 [2, 1, 0] 将其变为 BGR。
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0 #除以255.0，将像素值归一化到 [0, 1] 之间
        inputs = inputs.sub_(self.MEAN).div_(self.STDV) #减去均值，除以标准差，进一步归一化。

        # extract features
        gmap = self.__feature_encoder(inputs) #提取matching feature, fnet(获取feature matching，另外一个为context network)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            # 返回的context features分为两部分，一部分为tanh激活（net）和另外一部分为relu激活（inp）
            net, inp = self.__context_encoder(inputs[:,[0]]) 
            self.net, self.inp, self.fmap = net, inp, gmap # [1,128,H//8,W//8], [1,128,H//8,W//8], [1,128,H//8,W//8]
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0])

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]#生成坐标网格，在前两个维度上增加两个维度。变为(1, 1, ht, wd, 2)

            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0) #关键帧和当前帧之间的相关运算 [None,[0]]即保留第一行之后进行unsqueeze(0)，

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:#如果运动的幅度大于阈值
                self.count = 0 #计数器清零
                net, inp = self.__context_encoder(inputs[:,[0]]) 
                self.net, self.inp, self.fmap = net, inp, gmap 
                # 打包数据，时间错，图像，内参，特征，context features，input features
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0])
            else:
                self.count += 1 #否则一直累加计数器
