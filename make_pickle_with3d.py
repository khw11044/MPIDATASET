from scipy import io
import json
import os 
import glob
import numpy as np
import pickle
# import matplotlib.pyplot as plt 
# from vis import show3Dpose
# 3dhp --> h36m 폼으로 바꾸기 

# root = 'train_data/annotation'
root = 'test_data/annotation'

matfile_list = glob.glob(root + '/*mat')


train_json = {}
train_json["poses_2d_pred"] = {}
train_json["poses_2d_pred"]["cam0"] = []
train_json["poses_2d_pred"]["cam2"] = []
train_json["poses_2d_pred"]["cam7"] = []
train_json["poses_2d_pred"]["cam8"] = []

train_json["confidences"] = {}
train_json["confidences"]["cam0"] = []
train_json["confidences"]["cam2"] = []
train_json["confidences"]["cam7"] = []
train_json["confidences"]["cam8"] = []

train_json["poses_3d_pred"] = {}
train_json["poses_3d_pred"]["cam0"] = []
train_json["poses_3d_pred"]["cam2"] = []
train_json["poses_3d_pred"]["cam7"] = []
train_json["poses_3d_pred"]["cam8"] = []


train_json["subjects"] = []

def convert_form(full_2Dx):
    h36m_form = np.append(full_2Dx[23:26],full_2Dx[18:21])
    h36m_form = np.append(h36m_form,full_2Dx[4:8])
    h36m_form = np.append(h36m_form,full_2Dx[9:12])
    h36m_form = np.append(h36m_form,full_2Dx[14:17])

    return h36m_form

for i, mat_path in enumerate(matfile_list):           # 7개의 subject      S1, S2, S3, S4, S5, S6, S7
    s = int(matfile_list[i].split('/')[-1].split('_')[1][-1])
    mat_file = io.loadmat(mat_path)
    annot2 = mat_file['annot2']         # 파일 하나당 14개의 영상 == 14개의 view --> 0 2 7 8 view 선택 
    annot3 = mat_file['annot3'] 
    for i, video in enumerate(annot2):  # i는 view
        if i == 0:  
            for j,point in enumerate(video[0]):
                full_2Dx = convert_form(point[0::2])         
                full_2Dy = convert_form(point[1::2])
                point = np.concatenate((full_2Dx,full_2Dy),axis=0)
                train_json["poses_2d_pred"]["cam0"].append(point)
                train_json["confidences"]["cam0"].append(np.ones(point.shape[0]//2))
                train_json["subjects"].append(s)

                point3d = annot3[i][0][j]
                full_3Dx = convert_form(point3d[0::3])   
                full_3Dz = convert_form(-point3d[1::3])      
                full_3Dy = convert_form(point3d[2::3])       
                point3d = np.concatenate((full_3Dx,full_3Dy,full_3Dz),axis=0)
                train_json["poses_3d_pred"]["cam0"].append(point3d)

        elif i == 2:
            for j,point in enumerate(video[0]):
                full_2Dx = convert_form(point[0::2])         
                full_2Dy = convert_form(point[1::2])
                point = np.concatenate((full_2Dx,full_2Dy),axis=0)
                train_json["poses_2d_pred"]["cam2"].append(point)
                train_json["confidences"]["cam2"].append(np.ones(point.shape[0]//2))

                point3d = annot3[i][0][j]
                full_3Dx = convert_form(point3d[0::3])   
                full_3Dz = convert_form(-point3d[1::3])      
                full_3Dy = convert_form(point3d[2::3])
                point3d = np.concatenate((full_3Dx,full_3Dy,full_3Dz),axis=0)
                train_json["poses_3d_pred"]["cam2"].append(point3d)

        elif i == 7:
            for j,point in enumerate(video[0]):
                full_2Dx = convert_form(point[0::2])         
                full_2Dy = convert_form(point[1::2])
                point = np.concatenate((full_2Dx,full_2Dy),axis=0)
                train_json["poses_2d_pred"]["cam7"].append(point)
                train_json["confidences"]["cam7"].append(np.ones(point.shape[0]//2))

                point3d = annot3[i][0][j]
                full_3Dx = convert_form(point3d[0::3])   
                full_3Dz = convert_form(-point3d[1::3])      
                full_3Dy = convert_form(point3d[2::3])
                point3d = np.concatenate((full_3Dx,full_3Dy,full_3Dz),axis=0)
                train_json["poses_3d_pred"]["cam7"].append(point3d)

        elif i == 8:
            for j,point in enumerate(video[0]):
                full_2Dx = convert_form(point[0::2])         
                full_2Dy = convert_form(point[1::2])
                point = np.concatenate((full_2Dx,full_2Dy),axis=0)
                train_json["poses_2d_pred"]["cam8"].append(point)
                train_json["confidences"]["cam8"].append(np.ones(point.shape[0]//2))

                point3d = annot3[i][0][j]
                full_3Dx = convert_form(point3d[0::3])   
                full_3Dz = convert_form(-point3d[1::3])      
                full_3Dy = convert_form(point3d[2::3])
                point3d = np.concatenate((full_3Dx,full_3Dy,full_3Dz),axis=0)
                train_json["poses_3d_pred"]["cam8"].append(point3d)
        

    print('done mat file')        

train_json["poses_2d_pred"]["cam0"] = np.array(train_json["poses_2d_pred"]["cam0"])
train_json["poses_2d_pred"]["cam2"] = np.array(train_json["poses_2d_pred"]["cam2"])
train_json["poses_2d_pred"]["cam7"] = np.array(train_json["poses_2d_pred"]["cam7"])
train_json["poses_2d_pred"]["cam8"] = np.array(train_json["poses_2d_pred"]["cam8"])

train_json["confidences"]["cam0"] = np.array(train_json["confidences"]["cam0"])
train_json["confidences"]["cam2"] = np.array(train_json["confidences"]["cam2"])
train_json["confidences"]["cam7"] = np.array(train_json["confidences"]["cam7"])
train_json["confidences"]["cam8"] = np.array(train_json["confidences"]["cam8"])

train_json["poses_3d_pred"]["cam0"] = np.array(train_json["poses_3d_pred"]["cam0"])
train_json["poses_3d_pred"]["cam2"] = np.array(train_json["poses_3d_pred"]["cam2"])
train_json["poses_3d_pred"]["cam7"] = np.array(train_json["poses_3d_pred"]["cam7"])
train_json["poses_3d_pred"]["cam8"] = np.array(train_json["poses_3d_pred"]["cam8"])



print(len(train_json))



json_file_path = root.split('/')[0] + ".pickle"

with open(json_file_path, "wb") as outfile:
    pickle.dump(train_json, outfile, pickle.HIGHEST_PROTOCOL)



