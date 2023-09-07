# Pandaset dataloader
# Usage: python3 dataloader.py <pandaset_path>

import os
import sys
import json
import gzip
import pickle
import quaternion
import numpy as np
import open3d as o3d

# path to pandaset
pandaset_path = sys.argv[1]
# output path for point cloud
output_path = sys.argv[2]

for sequence in sorted(os.listdir(pandaset_path)):
    lidar_path = os.path.join(pandaset_path, sequence, 'lidar')
    label_path = os.path.join(pandaset_path, sequence, 'annotations', 'semseg')

    if not os.path.exists(label_path):
        continue

    pc = np.zeros((1, 3))

    index = 0

    transformation = None

    poses_file = open(os.path.join(lidar_path, 'poses.json'))
    poses = json.load(poses_file)
    poses_file.close()

    for scan in sorted(os.listdir(lidar_path)):

        # skip json file
        if scan[-1] == 'n':
            continue

        with gzip.open(os.path.join(label_path, scan), 'rb') as f:
            label = pickle.load(f)
            label = label.to_numpy(dtype=np.uint32)

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

            # keep the first transformation for all the other clouds
            if index == 0:
                transformation = T_inv

            # homogeneous coordinates
            hpoints = np.hstack((points, np.ones((points.shape[0], 1))))

            point_cloud = np.matmul(transformation, hpoints.T).T[:, :3]

            #point_cloud = points - position
            #point_cloud = np.matmul(R, point_cloud.T)
            #point_cloud = point_cloud.T + np.matmul(R, position)
            #point_cloud = np.dot(points, R.T)
            #point_cloud = point_cloud.T #- np.matmul(R.T, position)
            #point_cloud = np.column_stack((rotated_pc.T, cloud.to_numpy()[:, 3]))

            #point_cloud = point_cloud.T[:, :3] + np.matmul(R, position)

            #if index in range(0, 21, 5):
            pc = np.append(pc, point_cloud, axis=0)

        index += 1
            
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

    # Create a visualizer and add the point cloud to it
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # Run the visualizer
    vis.run()
    vis.destroy_window()

    # save point cloud as ply
    o3d.io.write_point_cloud(output_path, pcd)