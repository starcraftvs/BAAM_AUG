from typing import Union
import json
import numpy as np

def read_json_file(json_file: Union[str, dict]):
    if isinstance(json_file, str):
        with open(json_file) as f:
            info = json.load(f)
    else:
        info = json_file
    return info

def get_cams_info_from_json(json_file: Union[str, dict]):
    cam_info_list = []
    json_info = read_json_file(json_file)
    for sensor_info in json_info['meta']['sensor']:
        if 'CAM' not in sensor_info['sensor_id']:
            continue
        cam_info = {}
        cam_info['sensor_id'] = sensor_info['sensor_id']
        cam_info['img_size'] = (sensor_info['width'], sensor_info['height'])
        sensor_param = sensor_info['sensor_param']
        cam_info['cam2ego_rotation'] = np.array(sensor_param['sensor2ego_rotation'])
        cam_info['cam2ego_translation'] = np.array(sensor_param['sensor2ego_translation'])
        if cam_info['cam2ego_rotation'].shape == 3:
            temp = cam_info['cam2ego_rotation'].copy()
            cam_info['cam2ego_rotation'] = cam_info['cam2ego_translation']
            cam_info['cam2ego_translation'] = temp
        cam_info['cam_intri'] = np.array(sensor_param['intrinsic'])
        cam_info['cam_model'] = sensor_param['camera_model']
        if 'FISHEYE' in cam_info['sensor_id']:
            cam_info['distort'] = np.array(sensor_param['distort'])
            cam_info['cam_model'] = 'opencv_omini'
        if 'sensor2lidar_rotation'in sensor_param:
            cam_info['cam2lidar_rotation'] = np.array(sensor_param['sensor2lidar_rotation'])
            cam_info['cam2lidar_translation'] = np.array(sensor_param['sensor2lidar_translation'])
            if cam_info['cam2lidar_rotation'].shape == 3:
                temp = cam_info['cam2lidar_rotation'].copy()
                cam_info['cam2lidar_rotation'] = cam_info['cam2lidar_translation']
                cam_info['cam2lidar_translation'] = temp
        if 'cam_extri' in cam_info:
            cam_info['cam_extri'] = np.array(cam_info['cam_extri'])
        cam_info_list.append(cam_info)
    return cam_info_list