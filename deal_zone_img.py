import json
import os
import numpy as np
import cv2
from utils.zone_json import get_cams_info_from_json
from utils.cam import cap_vr_cam
from vrtech.blender_utils.slam_utils.matrix_utils import pose_to_4x4, rot_vec_to_4x4, mat_to_pose, cam_mat_to_blender_mat
import subprocess
# def cap_vr_cam(ori_img, ori_K, new_K, new_img_size):
#     w_ori, h_ori = ori_img.shape[1], ori_img.shape[0]
#     w_new, h_new = new_img_size
#     map_x_new, map_y_new = np.meshgrid(np.arange(h_new), np.arange(w_new))
#     map_x_old =(map_x_new.copy() - new_K[0,2])/new_K[0,0]*ori_K[0,0] + ori_K[0,2]
#     map_y_old =(map_y_new.copy() - new_K[1,2])/new_K[1,1]*ori_K[1,1] + ori_K[1,2]
#     map_x_old = map_x_old.astype(np.float32)
#     map_y_old = map_y_old.astype(np.float32)
#     new_img = cv2.remap(ori_img, map_x_old, map_y_old, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
#     return new_img

json_file = r'cam_zone.json'
apollo_cam_intri = np.array([
            [2304.54786556982, 0, 1686.23787612802],
            [0, 2305.875668062, 1354.98486439791],
            [0,0,1]
        ])
apollo_cam_extri = np.linalg.inv(pose_to_4x4(np.array([0.1863706260919571, -0.04961742088198662, 3.182359457015991, 4.284457683563232, 3.4337785243988037, 10.721973419189453])))
# apollo_cam_extri = pose_to_4x4(np.array([0.1863706260919571, -0.04961742088198662, 3.182359457015991, 4.284457683563232, 3.4337785243988037, 10.721973419189453]))

apollo_cam_extri[:3, 3] = 0
test_img_path = r'zone_raw_sample_data\n000009_2023-05-21-11-17-14-101116_CAM_FRONT_120.distorted.jpeg'
cams_info = get_cams_info_from_json(json_file)
zone_cam_intri = cams_info[0]['cam_intri']
zone_cam_extri = rot_vec_to_4x4(cams_info[0]['cam2ego_rotation'])
blr_zone_cam_extri = cam_mat_to_blender_mat(zone_cam_extri)
blr_zone_cam_extri = rot_vec_to_4x4(np.array([0, 0, -0])/180*np.pi)
blr_apollo_cam_extri = rot_vec_to_4x4(np.array([-12, -8, 4])/180*np.pi)
ori_img = cv2.imread(test_img_path)
new_img = cap_vr_cam(ori_img, zone_cam_intri, apollo_cam_intri, (2710, 3384))
cv2.imwrite(r'data\apollo\val\images\zone_img2.jpg', new_img)
subprocess.run(r'c:\Users\46501\.conda\envs\baamv2\python.exe main.py', shell=True, check=True)
subprocess.run(r'D: && cd D:\Code\3DReconstruction\BAAM/vis && cmd /C "c:\Users\46501\.conda\envs\baamv2\python.exe vis_apollo.py', shell=True, check=True)