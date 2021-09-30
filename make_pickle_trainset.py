from scipy import io
import json
import os 
import glob
import numpy as np
import pickle
# 3dhp --> h36m 폼으로 바꾸기 

root = 'train_data/annotation/seq1'

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
    for i, video in enumerate(annot2):
        if i == 0:
            for point in video[0]:
                full_2Dx = convert_form(point[0::2])         
                full_2Dy = convert_form(point[1::2])
                point = np.concatenate((full_2Dx,full_2Dy),axis=0)
                train_json["poses_2d_pred"]["cam0"].append(point)
                train_json["confidences"]["cam0"].append(np.ones(point.shape[0]//2))
                train_json["subjects"].append(s)
        elif i == 2:
            for point in video[0]:
                full_2Dx = convert_form(point[0::2])         
                full_2Dy = convert_form(point[1::2])
                point = np.concatenate((full_2Dx,full_2Dy),axis=0)
                train_json["poses_2d_pred"]["cam2"].append(point)
                train_json["confidences"]["cam2"].append(np.ones(point.shape[0]//2))
        elif i == 7:
            for point in video[0]:
                full_2Dx = convert_form(point[0::2])         
                full_2Dy = convert_form(point[1::2])
                point = np.concatenate((full_2Dx,full_2Dy),axis=0)
                train_json["poses_2d_pred"]["cam7"].append(point)
                train_json["confidences"]["cam7"].append(np.ones(point.shape[0]//2))
        elif i == 8:
            for point in video[0]:
                full_2Dx = convert_form(point[0::2])         
                full_2Dy = convert_form(point[1::2])
                point = np.concatenate((full_2Dx,full_2Dy),axis=0)
                train_json["poses_2d_pred"]["cam8"].append(point)
                train_json["confidences"]["cam8"].append(np.ones(point.shape[0]//2))

    
    print('done mat file')        

train_json["poses_2d_pred"]["cam0"] = np.array(train_json["poses_2d_pred"]["cam0"])
train_json["poses_2d_pred"]["cam2"] = np.array(train_json["poses_2d_pred"]["cam2"])
train_json["poses_2d_pred"]["cam7"] = np.array(train_json["poses_2d_pred"]["cam7"])
train_json["poses_2d_pred"]["cam8"] = np.array(train_json["poses_2d_pred"]["cam8"])

train_json["confidences"]["cam0"] = np.array(train_json["confidences"]["cam0"])
train_json["confidences"]["cam2"] = np.array(train_json["confidences"]["cam2"])
train_json["confidences"]["cam7"] = np.array(train_json["confidences"]["cam7"])
train_json["confidences"]["cam8"] = np.array(train_json["confidences"]["cam8"])

print(len(train_json))

json_file_path = "./train_mpi.pickle"

with open(json_file_path, "wb") as outfile:
    pickle.dump(train_json, outfile, pickle.HIGHEST_PROTOCOL)

# annot2 : 14개의 동영상
# annot2[0] : 

# # 특정 변수 읽기
# annot2 = mat_file['annot2']
# annot3 = mat_file['annot3']
# cameras = mat_file['cameras']
# frames = mat_file['frames']
# univ_annot3 = mat_file['univ_annot3']

