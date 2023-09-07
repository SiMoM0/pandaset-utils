# Pandaset predictions visualizer
# Usage: python3 dataloader.py <sequence> <pandaset_path> <prediction_path>
# Example: python3 pandaset-dataloader/visualizer.py 001 /media/simone/T7/Documents/Uni/MasterThesis/Datasets/PandaSet/ ~/Documents/Methods/Cylinder3D/predictions/

import os
import sys
import json
import gzip
import pickle
import yaml
import quaternion
import numpy as np
import open3d as o3d

# args
sequence = sys.argv[1]
pandaset_path = sys.argv[2]
pred_path = sys.argv[3]

lidar_path = os.path.join(pandaset_path, sequence, 'lidar')

poses_file = open(os.path.join(lidar_path, 'poses.json'))
poses = json.load(poses_file)
poses_file.close()

index = 0

for scan in sorted(os.listdir(lidar_path)):
    
    # skip json file
    if scan[-1] == 'n':
        continue

    #label_path = os.path.join(pred_path, sequence, '{0:06d}.label'.format(index)) # cylinder format
    label_path = os.path.join(pred_path, sequence, '{0:02d}.label'.format(index)) # pvkd format
    labels = np.fromfile(label_path, dtype=np.uint32).reshape(-1, 1)
    labels = labels & 0xFFFF
    #np.savetxt('cyl_predictions.txt', labels, fmt='%d')
    
    with gzip.open(os.path.join(lidar_path, scan), 'rb') as f:
        cloud = pickle.load(f)
        
        # select only 360° lidar (value 0)
        cloud = cloud[cloud['d'] == 0]
        points = cloud.to_numpy(dtype=np.float32)[:, :3]
        
        # get pose and orientation
        pose = poses[index]
        position = np.array(list(pose['position'].values()), dtype=np.float32)
        heading = np.array(list(pose['heading'].values()), dtype=np.float32)
        
        # convert to rotation matrix 3x3
        R = quaternion.as_rotation_matrix(np.quaternion(*heading))
        
        # transformation matrix 4x4
        T = np.zeros(shape=(4, 4), dtype=np.float32)
        T[0:3, 0:3] = R
        T[0:3, 3] = position
        T[3, 3] = 1
        
        # inverse transformation
        T_inv = np.linalg.inv(T)

        # homogeneous coordinates
        hpoints = np.hstack((points, np.ones((points.shape[0], 1))))
        point_cloud = np.matmul(T_inv, hpoints.T).T[:, :3]

        # load colors for labels
        with open('./config/color_map.yaml', 'r') as f:
            color_map = yaml.safe_load(f)
        colors = np.array([color_map[int(i)] for i in labels])
        colors = colors.reshape((-1, 3)) / [255, 255, 255]

        # visualize single point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Create a visualizer and add the point cloud to it
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        # Set the camera position to view the point cloud from the origin
        ctr = vis.get_view_control().convert_to_pinhole_camera_parameters()
        ctr.extrinsic = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
        vis.get_view_control().convert_from_pinhole_camera_parameters(ctr)
        # Run the visualizer
        vis.run()
        vis.destroy_window()

    index += 1