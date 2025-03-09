import open3d as o3d
import queue
import copy
import numpy as np
from pcd import make_point_cloud

# NOTE 1.: this non_blocking_visualizer is a consumer thread that reads from a 
# queue. The producer is the inference loop in visualize_point_cloud_*.py

# NOTE 2.: DPVO does not include all the frames in the graph optimization 
# procedure. Thus, the number of writes to the poseQueue is not the same
# to the number of write to the pcdQueue.

def non_blocking_visualizer(pcdQueue, poseQueue):
    #TODO: Understand why it crashes after rendering all the point clouds.
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    add_geometry = True
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    poseOptimized = False
    savePoses = None

    while True:
        print("Visualizing new pcd...")
        (t, rgb, depth, K) = pcdQueue.get()

        if not poseOptimized:
            (tstamps, poses) = poseQueue.get()
            # Save the last instance of valid optimized poses
            if poses is not None:
                savePoses = poses.copy()
            else:
                # Means that pose optimization has ended, now wait for monocular depth estimation to end.
                poseOptimized = True

        # See NOTE 2.
        if t < 0 and tstamps is None:
            # Now both MDE and VO have ended, so return
            break
        
        # print("####### CHECK SYNC ########")
        # print(t, len(savePoses), savePoses)
        # print(rgb)
        # print(depth)
        # print(K)
    
        # Convert K to a PinholeCameraIntrinsic object in Open3D, which is non-pickable...
        # TODO: Pass K a single time at init. 99.9% of the time we don't to stich clouds coming from different cameras.
        # Camera intrinsics in open3d format for interactive visualizer
        K = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                                height=480, 
                                                fx=K[0, 0], 
                                                fy=K[1, 1], 
                                                cx=K[0, 2], 
                                                cy=K[1, 2])

        if add_geometry: 
            pcd, _ = make_point_cloud(rgb, depth, K, convert_rgb_to_intensity=False)
            pcd.transform(flip_transform)
            # print("Adding geometry")
            vis.add_geometry(pcd)
            add_geometry = False
        else: 
            update, _ = make_point_cloud(rgb, depth, K, convert_rgb_to_intensity=False) 
            update.transform(flip_transform)
            pcd.points = copy.deepcopy(update.points)
            pcd.colors = copy.deepcopy(update.colors)
            # print("Updating geometry")
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

