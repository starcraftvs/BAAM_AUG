import cv2
import numpy as np

def cap_vr_cam(ori_img, ori_K, new_K, new_img_size, ori_T=np.eye(4, 4), new_T=np.eye(4, 4)):
    w_ori, h_ori = ori_img.shape[1], ori_img.shape[0]
    w_new, h_new = new_img_size
    map_x_new, map_y_new = np.meshgrid(np.arange(h_new), np.arange(w_new))
    map_new_3d = np.stack([map_x_new, map_y_new, np.ones_like(map_x_new)], axis=-1)
    map_new_3d = map_new_3d.reshape(-1, 3)
    map_world_3d = np.dot(np.linalg.inv(new_K), map_new_3d.T)
    map_world_3d = np.dot(np.linalg.inv(new_T[:3, :3]), map_world_3d)
    map_world_3d = np.dot(ori_T[:3, :3], map_world_3d)
    map_old_3d = np.dot(ori_K, map_world_3d)
    map_x_old = map_old_3d[0, :]/map_old_3d[2, :]
    map_y_old = map_old_3d[1, :]/map_old_3d[2, :]
    map_x_old = map_x_old.reshape(w_new, h_new)
    map_y_old = map_y_old.reshape(w_new, h_new)
    # map_x_old =(map_x_new.copy() - new_K[0,2])/new_K[0,0]*ori_K[0,0] + ori_K[0,2]
    # map_y_old =(map_y_new.copy() - new_K[1,2])/new_K[1,1]*ori_K[1,1] + ori_K[1,2]
    map_x_old = map_x_old.astype(np.float32)
    map_y_old = map_y_old.astype(np.float32)
    new_img = cv2.remap(ori_img, map_x_old, map_y_old, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return new_img

def old_3d_to_new_3d(ori_3d_points, ori_T, new_T):
    new_3d_points = np.dot(new_T, np.dot(np.linalg.inv(ori_T), np.vstack([ori_3d_points, np.ones((1, ori_3d_points.shape[1]))])))
    new_3d_points = new_3d_points[:3, :]/new_3d_points[3, :]
    return new_3d_points

def old_2d_to_new_2d(ori_2d_points, ori_K, new_K, ori_T, new_T):
    ori_3d_points = np.dot(np.linalg.inv(ori_K), np.vstack([ori_2d_points, np.ones((1, ori_2d_points.shape[1]))]))
    new_3d_points = old_3d_to_new_3d(ori_3d_points, ori_T, new_T)
    new_2d_points = np.dot(new_K, new_3d_points)
    # filtered_new_2d_points = new_2d_points[new_2d_points[2, :] > 0]
    new_2d_points[0, new_2d_points[2, :] < 0] = 0
    new_2d_points[1, new_2d_points[2, :] < 0] = 0
    new_2d_points = new_2d_points[:2, :]/new_2d_points[2, :]
    return new_2d_points