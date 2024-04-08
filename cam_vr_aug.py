import os
import json
import cv2
import numpy as np
from utils.cam import cap_vr_cam, old_2d_to_new_2d
from vrtech.blender_utils.slam_utils.matrix_utils import pose_to_4x4, rot_vec_to_4x4, mat_to_pose, cam_mat_to_blender_mat
import uuid
ori_image_folder_path = r'D:\DataSets\Apollo\ApolloForBAAM\train\images'
ori_json_folder_path = r'D:\DataSets\Apollo\ApolloForBAAM\train\apollo_annot'
ori_imgs_path = [os.path.join(ori_image_folder_path, i) for i in os.listdir(ori_image_folder_path)]
blr_apollo_cam_extri = rot_vec_to_4x4(np.array([-12, -8, 4])/180*np.pi)
K  =  np.array([
            [2304.54786556982, 0, 1686.23787612802],
            [0, 2305.875668062, 1354.98486439791],
            [0,0,1]
        ])
for a in range(20):
    for i in range(len(ori_imgs_path)):
        ori_img_path = ori_imgs_path[i]
        ori_json_path = os.path.join(ori_json_folder_path, os.path.basename(ori_img_path).split('.')[0]+'.json')
        if not (os.path.exists(ori_img_path) and os.path.exists(ori_json_path)): continue
        ori_dict = json.load(open(ori_json_path))
        ori_img = cv2.imread(ori_img_path)
        new_dict = ori_dict.copy()
        random_rot_vec = (np.random.random((3,)) - 0.5) * 20 / 180 * np.pi
        ori_T = np.eye(4)
        target_T = rot_vec_to_4x4(random_rot_vec)
        # target_T = np.eye(4)
        new_img = cap_vr_cam(ori_img, K, K, (2710, 3384), ori_T=ori_T, new_T=target_T)
        for j in range(len(new_dict)):
            # repro box
            old_bbox = new_dict[j]['bbox']
            old_bbox_points = np.array([[old_bbox[0], old_bbox[1]], [old_bbox[2], old_bbox[3]], [old_bbox[0], old_bbox[3]], [old_bbox[2], old_bbox[1]]]).T
            new_bbox_points = old_2d_to_new_2d(old_bbox_points, K, K, ori_T, target_T)
            new_bbox = [np.min(new_bbox_points[0]), np.min(new_bbox_points[1]), np.max(new_bbox_points[0]), np.max(new_bbox_points[1])]
            new_bbox = [int(i) for i in new_bbox]
            new_dict[j]['bbox'] = new_bbox
            # old_bbox_points = np.array([[old_bbox[0], old_bbox[1]], [old_bbox[2], old_bbox[3]]]).T
            # new_bbox_points = old_2d_to_new_2d(old_bbox_points, K, K, ori_T, target_T)
            # new_bbox = [new_bbox_points[0, 0], new_bbox_points[1, 0], new_bbox_points[0, 1], new_bbox_points[1, 1]]
            # new_img = cv2.rectangle(new_img, (int(new_bbox[0]), int(new_bbox[1])), (int(new_bbox[2]), int(new_bbox[3])), (0, 255, 0), 2)
            # repro keypoints
            old_keypoints_2d_with_visible = new_dict[j]['keypoints'].copy()
            old_keypoints_2d = np.array(old_keypoints_2d_with_visible).reshape(-1, 3)[:, :2].T
            old_keypoints_2d_filtered = old_keypoints_2d.copy()
            old_keypoints_2d_filtered = old_keypoints_2d_filtered.reshape(2, -1)
            # if x == 0 and y == 0: filter these points
            old_keypoints_2d_filtered_x = old_keypoints_2d_filtered[0]
            old_keypoints_2d_filtered_y = old_keypoints_2d_filtered[1]
            old_keypoints_2d_filtered_mask = np.logical_or(old_keypoints_2d_filtered_x > 0, old_keypoints_2d_filtered_y > 0)
            old_keypoints_2d_filtered = old_keypoints_2d_filtered[:, old_keypoints_2d_filtered_mask]
            # old_keypoints_2d_filtered = old_keypoints_2d_filtered[old_keypoints_2d_filtered>0].reshape(2, -1)
            new_keypoints_2d_filtered = old_2d_to_new_2d(old_keypoints_2d_filtered, K, K, ori_T, target_T)
            # filter out points that are out of the image
            new_keypoints_2d_filtered_x = new_keypoints_2d_filtered[0]
            new_keypoints_2d_filtered_y = new_keypoints_2d_filtered[1]
            new_keypoints_2d_filtered_mask = np.logical_and(np.logical_and(new_keypoints_2d_filtered_x > 0, new_keypoints_2d_filtered_x < 3384), np.logical_and(new_keypoints_2d_filtered_y > 0, new_keypoints_2d_filtered_y < 2710))
            new_keypoints_2d_filtered[0][np.logical_not(new_keypoints_2d_filtered_mask)] = 0
            new_keypoints_2d_filtered[1][np.logical_not(new_keypoints_2d_filtered_mask)] = 0
            new_keypoints_2d_filtered_visible = np.zeros_like(old_keypoints_2d_filtered[0])
            new_keypoints_2d_filtered_visible[new_keypoints_2d_filtered_mask] = 2.0
            new_keypoints_2d_with_visible = np.zeros_like(old_keypoints_2d_with_visible)
            new_keypoints_2d_with_visible[::3][old_keypoints_2d_filtered_mask] = new_keypoints_2d_filtered[0]
            new_keypoints_2d_with_visible[1::3][old_keypoints_2d_filtered_mask] = new_keypoints_2d_filtered[1]
            new_keypoints_2d_with_visible[2::3][old_keypoints_2d_filtered_mask] = new_keypoints_2d_filtered_visible
            new_dict[j]['keypoints'] = new_keypoints_2d_with_visible.tolist()
            # new_keypoints_2d = np.zeros_like(old_keypoints_2d)
            # new_keypoints_2d_visibility = np.zeros_like(new_keypoints_2d[0, :])
            # new_keypoints_2d_visibility[new_keypoints_2d[0, :]>0] = 2.0
            # new_keypoints_2d_with_visible = np.zeros_like(old_keypoints_2d_with_visible)
            # new_keypoints_2d_with_visible[::3] = new_keypoints_2d[0]
            # new_keypoints_2d_with_visible[1::3] = new_keypoints_2d[1]
            # new_keypoints_2d_with_visible[2::3] = new_keypoints_2d_visibility

            # repro segmenation
            old_seg = new_dict[j]['segmentation']
            new_seg = []
            for seg_part in old_seg:
                seg_part = old_seg[0]
                new_seg_part = old_2d_to_new_2d(np.array(seg_part).reshape(-1, 2).T, K, K, ori_T, target_T)
                new_seg_part = new_seg_part.T.reshape(-1).tolist()
                new_seg.append(new_seg_part)   
                new_dict[j]['segmentation'] = new_seg

            # repro pose
            old_pose = new_dict[j]['pose']
            old_pose_mat = pose_to_4x4(np.array(old_pose))
            new_pose_mat = np.dot(target_T, np.dot(np.linalg.inv(ori_T), old_pose_mat))
            new_pose = mat_to_pose(new_pose_mat).tolist()
            new_dict[j]['pose'] = new_pose

        # save new image and json
        uuid_str = str(uuid.uuid4())
        new_img_path = os.path.join(r'D:\DataSets\Apollo\ApolloForBAAM\train_augged\images', os.path.basename(ori_img_path).split('.')[0]+'_'+uuid_str+'.png')
        new_json_path = os.path.join(r'D:\DataSets\Apollo\ApolloForBAAM\train_augged\apollo_annot', os.path.basename(ori_img_path).split('.')[0]+'_'+uuid_str+'.json')
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(new_json_path), exist_ok=True)
        cv2.imwrite(new_img_path, new_img)
        with open(new_json_path, 'w') as f:
            json.dump(new_dict, f)
            # new_dict[j]['keypoints'] = new_keypoints_2d_with_visible.tolist()
            # for k in range(0, len(new_keypoints_2d_with_visible), 3):
            #     if new_keypoints_2d_with_visible[k+2] < 0.2: continue
            #     new_img = cv2.circle(new_img, (int(new_keypoints_2d_with_visible[k]), int(new_keypoints_2d_with_visible[k + 1])), 3, (0, 0, 255), -1)
            #     if old_keypoints_2d_with_visible[k+2] < 0.2: continue
            #     ori_img = cv2.circle(ori_img, (int(old_keypoints_2d_with_visible[k]), int(old_keypoints_2d_with_visible[k + 1])), 3, (0, 0, 255), -1)
            # new_keypoints_2d_with_visible[:len(new_keypoints_2d[0])] = new_keypoints_2d.reshape(-1)