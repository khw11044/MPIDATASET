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

cam_view = 'cam7'


plt.ion()
fig = plt.figure(1,figsize=(12,12))

for idx in range(0, data['poses_2d_pred']['cam0'].shape[0], 4):       # 18432개의 frame 
    
    p2d = data['poses_2d_pred'][cam_view][idx]
    p3d = data['poses_3d_pred'][cam_view][idx]

   
    vis_3d_cam0 = p3d.reshape(3,16)

    ax = fig.add_subplot('221', projection='3d', aspect='auto')
    show3Dpose(vis_3d_cam0.T, ax, radius=1000, lcolor='blue')

    
    plt.draw()
    plt.pause(0.01)
    fig.clear()
    
