import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from lietorch import SO3, SE3, Sim3
from scipy.spatial.transform import Rotation as R
import copy
import pickle
import re
import math

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # if colors != None:
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

    rotating = False

def str2array(ss):
    elem = re.sub('\s\s+',' ',ss).split(' ')
    num=[]
    for e in elem:
        num.append(float(e))
    return np.array(num)


f = open(r'./results/outdoors6.pkl','rb')
dump_data= pickle.load(f)
print(dump_data.keys())

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name='123')
vis.get_render_option().point_size = 4
opt = vis.get_render_option()
opt.background_color = np.asarray([0,0,0])

def key_action_callback(vis, action, mods):
    print(action)
    if action == 1:  # key down
        ctr = vis.get_view_control()
        view_params = ctr.convert_to_view_parameters()
        print(view_params)
    return True

vis.register_key_action_callback(32, key_action_callback)  # space

for ix in sorted(dump_data['points'].keys()):
    if ix < 520 : continue
    if ix > 920 : continue

    # if ix < 800 : continue
    # if ix > 1800 : continue

    dd=dump_data['points'][ix]
    pts = dd['pts'] # * 17.0
    clr = dd['clr']
    pose = dump_data['cameras'][ix]
    pose[0:3,3] = pose[0:3,3]
    npts_c = np.matmul(pose[0:3,0:3].T,(pts-pose[0:3,3]).T).T
    npts = np.asarray(pts)
    nclr = np.asarray(clr)
    mask0 = npts_c[:,1]> -5.0
    mask1 = np.logical_or(npts_c[:,1]> -0.0,nclr[:,0]<0.4)
    mask2 = npts_c[:,2] < 10.0
    mask = np.logical_and(np.logical_and(mask0,mask1),mask2)
    point_actor = create_point_actor(pts[mask], clr[mask])
    
    vis.add_geometry(point_actor)
    cam_actor = create_camera_actor(1.0,0.3)
    cam_actor.transform(pose)
    vis.add_geometry(cam_actor)

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters() 
    pose = dump_data['cameras'][ix]
    theta = 30/180.0*math.pi
    view_pose = np.array([[1.0,0.0,0.0,0.0],
                          [0.0,math.cos(theta),math.sin(theta),-15.0 * 0.7],
                          [0.0,-math.sin(theta),math.cos(theta),-30.0 * 0.7],
                          [0.0,0.0,0.0,1.0]])
    if ix > 5:
        rotvec_sum = np.array([0.0,0.0,0.0])
        t_sum = np.array([0.0,0.0,0.0])
        for ii in range(5):
            RR = dump_data['cameras'][ix-4+ii][0:3,0:3]
            rotvec_sum += R.as_rotvec(R.from_matrix(RR))
            t_sum += dump_data['cameras'][ix-4+ii][0:3,3]
        rotvec = rotvec_sum/5.0
        pose[0:3,0:3] = R.from_rotvec(rotvec).as_matrix()
        pose[0:3,3] = t_sum/5.0

    pose = np.matmul(pose,view_pose)
    camera_params.extrinsic = np.linalg.inv(pose)
    print(camera_params.intrinsic)
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    vis.poll_events()
    vis.update_renderer()
    # vis.capture_screen_image("tum_gif/%010d.jpg"%ix)


vis.destroy_window()
quit()

