import argparse, os, json
import copy
import cv2
from vis import utils as uts

# parser = argparse.ArgumentParser()
# parser.add_argument('-output', '--output', default='../outputs')
# parser.add_argument('-file', '--file', default='zone_img2')
# parser.add_argument('-save', '--save', default='../vis_res')
# args = parser.parse_args()

# import utils as uts
from pytorch3d.io import load_obj
import numpy as np
import open3d as o3d
# path info
# original image path
img_folder_path = r'data/apollo/val/images/'
# output path of inference
output_path = 'outputs/'
# set path to save
save_path = r'vis_res'
os.makedirs(save_path, exist_ok=True)
# path of normalized model
normalized_model_path = r'apollo_deform/0.obj'
# path of normal scaled models
normal_models_path = r'vis/car_models'
# pa
# load vert index
_, face, _ = load_obj(normalized_model_path) # get all the faces (all the vertices in obj are the same)
face = face.verts_idx.numpy()
# load class scales (all scales are from the car_models class (not reseanable)) !!
scales = {}
for i in range(79):
    vert, _, _ = load_obj('vis/car_models/{}.obj'.format(i))
    vert = vert.numpy()
    scales[i] = uts.call_scale(vert)
# load perdiction results
json_files_path = os.path.join(output_path, 'res')
meshes_files_path = os.path.join(output_path, 'mesh')

for file in os.listdir(json_files_path):
    file_name = file.split('.')[0]
    # number of detections * dicts
    res_dict = json.load(open(os.path.join(json_files_path, file)))
    # num of detections * vertices * 3
    res_meshes = np.load(os.path.join(meshes_files_path, file_name+'.npy'))
    # ori_img 
    ori_img = os.path.join(img_folder_path, file_name+'.jpg')
    if not os.path.exists(ori_img):
        ori_img = os.path.join(img_folder_path, file_name+'.jpeg')
    if not os.path.exists(ori_img):
        continue
    # draw 2d results
    res_img_2d = cv2.imread(ori_img)
    colors = np.random.random((len(res_dict), 3)) * 255
    for detect_dict in res_dict:
        if detect_dict['score'] < 0.1: continue
        # 2d bbox
        bbox = detect_dict['bbox']
        # draw 2d bbox
        res_img_2d = cv2.rectangle(res_img_2d, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        # 2d keypoints
        keypoints = detect_dict['keypoints']
        # draw 2d keypoints
        for i in range(0, len(keypoints), 3):
            if keypoints[i+2] < 0.2: continue
            res_img_2d = cv2.circle(res_img_2d, (int(keypoints[i]), int(keypoints[i+1])), 3, (0, 0, 255), -1)
    # save 2d result
    img_2d_res_folder = os.path.join(save_path, 'apollo', 'res_2d')
    os.makedirs(img_2d_res_folder, exist_ok=True)
    cv2.imwrite(os.path.join(img_2d_res_folder, file_name+'_2d.jpg'), res_img_2d)
    print(os.path.join(img_2d_res_folder, file_name+'_2d.jpg'))
    # draw 3d results
    # load original image
    res_img_3d = cv2.imread(ori_img)
    # load 3d model
    car_model = o3d.io.read_triangle_mesh(normalized_model_path)
    # load 3d results
    o3d_list = []
    # revert
    res_meshes[...,[0, 1]] *= -1
    for idx, detect_dict in enumerate(res_dict):
        if detect_dict['score'] < 0.1: continue
        # load vertices
        verts = res_meshes[idx]
        # load scale
        scale = scales[detect_dict['car_id']]
        # get size of predicted model
        detect_scale = uts.call_scale(verts)
        scale = scale / detect_scale
        # project models in opend
        model_mesh = uts.project(np.array(detect_dict['pose']), scale, verts)
        # vertices to mesh in open3d
        # create mesh target
        o3d_model = o3d.geometry.TriangleMesh()
        # set vertices and faces
        o3d_model.vertices = o3d.utility.Vector3dVector(model_mesh)
        o3d_model.triangles = o3d.utility.Vector3iVector(face)
        # set color
        o3d_model.paint_uniform_color(colors[idx]/255)
        # compute normals
        o3d_model.compute_vertex_normals()
        # append to list to render
        o3d_list.append(o3d_model)
    # save render image (directly same scale too large)
    img_scale = 4
    vis = uts.VisOpen3D(width=int(3384/img_scale), height=int(2710/img_scale),visible=False)
    for obj in o3d_list:
        vis.add_geometry(obj)
    vis.mesh_show_back_face()
    vis.to_apollo_plane(scale = img_scale)
    # vis.run()
    img_car_on_plane_path = os.path.join(save_path, 'apollo', 'res_3d', file_name+'_3d.png')
    os.makedirs(os.path.dirname(img_car_on_plane_path), exist_ok=True)
    vis.capture_screen_image(img_car_on_plane_path)
    del vis
    img_car_on_plane = cv2.imread(img_car_on_plane_path)
    # img_car_on_plane to rgba the white part is transparent
    img_car_on_plane = cv2.cvtColor(img_car_on_plane, cv2.COLOR_BGR2BGRA)
    img_car_on_plane[np.all(img_car_on_plane == [255, 255, 255, 255], axis=-1)] = [0, 0, 0, 0]
    # combine img_car_on_plane and the original img
    res_img_3d = cv2.cvtColor(res_img_3d, cv2.COLOR_BGR2BGRA)
    res_img_3d = cv2.resize(res_img_3d, (img_car_on_plane.shape[1], img_car_on_plane.shape[0]))
    res_img_3d = cv2.addWeighted(res_img_3d, 0.5, img_car_on_plane, 0.6, 0)
    cv2.imwrite(img_car_on_plane_path, res_img_3d)

        


print('here')

# vis 3D field to open3D UI
# vis = uts.VisOpen3D(width=1920, height=1080,visible=True)
# for obj in o3d_list:
#     vis.add_geometry(obj)
# vis.mesh_show_back_face()
# vis.load_view_point("view_point_3d.json")
# vis.run()
# vis.capture_screen_image(os.path.join(args.save, f'{args.file}.3d.png'))
