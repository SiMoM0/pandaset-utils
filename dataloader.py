# Pandaset dataloader
# Usage: python3 dataloader.py <pandaset_path>

import os
import sys
import json
import gzip
import yaml
import pickle
import quaternion
import numpy as np
import open3d as o3d

# path to pandaset
pandaset_path = sys.argv[1]

# label mapping from pandaset to kitti format
with open('./config/label_mapping.yaml', 'r') as f:
    label_map = yaml.safe_load(f)

for sequence in sorted(os.listdir(pandaset_path)):
    # get point cloud and labels
    lidar_path = os.path.join(pandaset_path, sequence, 'lidar')
    label_path = os.path.join(pandaset_path, sequence, 'annotations', 'semseg')

    # retrieve poses
    poses_file = open(os.path.join(lidar_path, 'poses.json'))
    poses = json.load(poses_file)
    poses_file.close()

    index = 0

    for scan in sorted(os.listdir(lidar_path)):

        # skip json file
        if scan[-1] == 'n':
            continue
        
        # read point cloud
        with gzip.open(os.path.join(lidar_path, scan), 'rb') as f:
            cloud = pickle.load(f)

        # select only 360° lidar (value 0)
        cloud = cloud[cloud['d'] == 0]
        points = cloud.to_numpy(dtype=np.float32)[:, :3]

        # get position and orientation
        pose = poses[index]
        position = list(pose['position'].values())
        heading = list(pose['heading'].values())

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

        # read labels
        with gzip.open(os.path.join(label_path, scan), 'rb') as f:
            label = pickle.load(f)
            labels = label.to_numpy(dtype=np.uint32)[:point_cloud.shape[0]]
            labels = labels & 0xFFFF
            labels = np.array([label_map[l[0]] for l in labels]).reshape(-1, 1)

        # load colors for labels
        with open('./color_map.yaml', 'r') as f:
            color_map = yaml.safe_load(f)
        colors = np.array([color_map[int(i)] for i in labels])
        colors = colors.reshape((-1, 3)) / [255, 255, 255]

        # create and visualize pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Create a visualizer and add the point cloud to it
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        # Run the visualizer
        vis.run()
        vis.destroy_window()

        index += 1