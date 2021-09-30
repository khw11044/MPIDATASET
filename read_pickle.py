import pickle
import matplotlib.pyplot as plt 
from vis import show3Dpose

json_file_path = "./test_data.pickle"
with open(json_file_path, 'rb') as f:
    data = pickle.load(f)

cams = ['cam0', 'cam2', 'cam7', 'cam8']
all_cams = ['cam0', 'cam1', 'cam2', 'cam3']

sample = dict()
anno3d = dict()

cam_view = 'cam1'


plt.ion()
fig = plt.figure(1,figsize=(12,12))
for idx in range(data['poses_2d_pred']['cam0'].shape[0]):       # 18432개의 frame 
    for c_idx, cam in enumerate(cams):
        p2d = data['poses_2d_pred'][cam][idx]
        p3d = data['poses_3d_pred'][cam][idx]

        sample['cam' + str(c_idx)] = p2d
        anno3d['cam' + str(c_idx)] = p3d
        print(cam)
        print(c_idx)

    poses_2d = {key:sample[key] for key in all_cams} 
    poses_3d = {key:anno3d[key] for key in all_cams}    # 카메라0,1,2,3 순으로 


    vis_3d_cam0 = poses_3d['cam0'].reshape(3,16)
    vis_3d_cam1 = poses_3d['cam1'].reshape(3,16)
    vis_3d_cam2 = poses_3d['cam2'].reshape(3,16)
    vis_3d_cam3 = poses_3d['cam3'].reshape(3,16)

    ax = fig.add_subplot('221', projection='3d', aspect='auto')
    show3Dpose(vis_3d_cam0.T, ax, radius=1000, lcolor='blue')

    ax = fig.add_subplot('222', projection='3d', aspect='auto')
    show3Dpose(vis_3d_cam1.T, ax, radius=1000, lcolor='blue')

    ax = fig.add_subplot('223', projection='3d', aspect='auto')
    show3Dpose(vis_3d_cam2.T, ax, radius=1000, lcolor='blue')

    ax = fig.add_subplot('224', projection='3d', aspect='auto')
    show3Dpose(vis_3d_cam3.T, ax, radius=1000, lcolor='blue')
    plt.draw()
    plt.pause(0.01)
    fig.clear()
    
