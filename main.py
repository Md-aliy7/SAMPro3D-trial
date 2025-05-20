import open3d as o3d
import numpy as np
import json
import os
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from scipy.stats import pearsonr
import random
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.spatial import cKDTree

# Define here all used functions

# Function to handle user-selected points and save them
def pick_points_and_save(pcd, output_file="selected_points.txt"):
    """
    Opens an interactive Open3D visualizer for selecting points in a 3D point cloud.
    Saves the selected points to a file.
    """
    print("1) Please pick at least one point using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window and save points")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()

    vis.add_geometry(pcd)

    # Run the visualizer and let the user pick points
    vis.run()

    # Get the indices of the selected points
    picked_points_indices = vis.get_picked_points()

    # Extract the coordinates of the selected points
    selected_points = np.asarray(pcd.points)[picked_points_indices]

    # Destroy visualizer window
    vis.destroy_window()

    # Save selected points
    np.savetxt(output_file, selected_points, fmt="%.7f")
    print(f"Selected points saved to {output_file}")
    return selected_points

# Function to automatically capture images and camera parameters
def auto_capture_images(image_count, rotation_angle):
    """
    Captures RGB images, depth images, and their corresponding intrinsic/extrinsic parameters.
    Ensures a full 360° rotation is achieved correctly.

    :param image_count: Number of images to capture for full rotation.
    :param rotation_angle: Rotation angle per step in degrees.
    """

    # Create necessary folders if they do not exist
    os.makedirs("rgb_images", exist_ok=True)
    os.makedirs("depth_images", exist_ok=True)
    os.makedirs("params", exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)
    view_control = vis.get_view_control()

    # Prompt the user to adjust zoom first
    print("Adjust the zoom position, then press 'Q' to continue.")
    vis.run()  # Let user adjust zoom

    # Save the adjusted zoom parameters
    initial_params = view_control.convert_to_pinhole_camera_parameters()

    # Correct the rotation step to ensure full 360° rotation matches expectations
    corrected_rotation_angle = 6 * rotation_angle  # Adjusted based on real-world effect

    depth_scale = 2500.0  # Define the depth scale to save

    for i in range(image_count):
        # Restore the zoom level before applying rotation
        view_control.convert_from_pinhole_camera_parameters(initial_params)

        # Compute the actual rotation angle that simulates reality
        actual_rotation = corrected_rotation_angle * i

        # Apply horizontal rotation
        view_control.rotate(actual_rotation, 0.0)

        # Capture RGB image
        rgb_filename = f"rgb_images/rgb_img_{i+1}.png"
        vis.capture_screen_image(rgb_filename, do_render=True)
        
        # Capture depth image
        depth_filename = f"depth_images/depth_img_{i+1}.png"
        vis.capture_depth_image(depth_filename, do_render=True, depth_scale=depth_scale)
        
        # Save camera parameters along with depth scale
        params_filename = f"params/params_{i+1}.json"
        # Get the intrinsic parameters of the camera
        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        intrinsic = camera_params.intrinsic

        # Convert camera parameters to a JSON-friendly format
        camera_params_dict = {
            "intrinsic": intrinsic.intrinsic_matrix.tolist(),  # Convert to list
            "extrinsic": camera_params.extrinsic.tolist(),  # Convert to list
            "width": camera_params.intrinsic.width,
            "height": camera_params.intrinsic.height,
            "depth_scale": depth_scale  # Added depth scale
        }

        # Save parameters as JSON
        with open(params_filename, "w") as f:
            json.dump(camera_params_dict, f, indent=4)

        print(f"Captured {rgb_filename}, {depth_filename}, and saved {params_filename}")

    vis.destroy_window()

def projection_3d_2d(points, depth_intrinsic, depth, pose, device):
    """
    :param points: N x 3 format (3D points in world coordinates)
    :param depth: H x W format (depth map)
    :param intrinsic: 3x3 format (camera intrinsic matrix)
    :param pose: 4x4 format (camera extrinsic matrix)
    :return p: N x 2 format (2D pixel coordinates on the image)
    :param corre_ins_idx: indices of the visible points in the frame
    """

    vis_thres = 0.1  # Visibility threshold for occlusion check
    depth_shift = 2500.0  # Depth map scale factor

    # Extract intrinsic parameters
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    bx = 0
    by = 0

    # Convert points to homogeneous coordinates (N x 4)
    points_world = torch.cat([points, torch.ones((points.shape[0], 1), dtype=torch.float64).to(device)], dim=-1).to(torch.float64)
    global points3D; points3D = points_world

    # Apply the inverse of the extrinsic matrix (camera pose) to get the points in camera space (applied in original SAMPro3D code, not here)
    # world_to_camera = torch.inverse(pose)
    world_to_camera = pose
    p = torch.matmul(world_to_camera, points_world.T)  # 4 x N (Xb, Yb, Zb, 1)

    # Project points onto the 2D image plane using the camera intrinsic parameters
    p[0] = (((p[0] - bx) * fx) / p[2] + cx)  # X projection
    p[1] = (((p[1] - by) * fy) / p[2] + cy)  # Y projection

    # plot_2d(trans_points)
    
    try:
        all_idx = torch.arange(0, len(points)).to(device)  # Indices of points
        # Out-of-image check: Find points that are outside the image bounds
        idx = torch.unique(torch.cat([torch.where(p[0] <= 0)[0], torch.where(p[1] <= 0)[0], 
                                      torch.where(p[0] >= depth.shape[1] - 1)[0], 
                                      torch.where(p[1] >= depth.shape[0] - 1)[0]], dim=0), dim=0)
        keep_idx = all_idx[torch.isin(all_idx, idx, invert=True)]  # Keep only points inside the image bounds

        p = p[:, keep_idx]  # Keep only the valid points

        if p.shape[1] == 0:
            print("no 3D prompt is visible in this frame")
            return p, keep_idx  # no 3D prompt is visible in this frame

        # Round the final 2D coordinates to pixel values
        pi = torch.round(p).to(torch.int64)

        # Estimate depth at the projected 2D points
        est_depth = p[2]

        # Get the depth from the depth map at the projected pixel locations
        trans_depth = depth[pi[1], pi[0]] / depth_shift  # Adjust for the depth scale factor

        # Occlusion check: Compare the estimated depth with the actual depth from the depth map
        idx_keep = torch.where(torch.abs(est_depth - trans_depth) <= vis_thres)[0]

        # Final 2D points after occlusion check
        p = p.T[idx_keep, :2]  # Keep the corresponding 2D coordinates
        keep_idx = keep_idx[idx_keep]  # Keep the corresponding indices

        return p, keep_idx

    except:
        raise ValueError

def plot_2d(trans_points):
    # Assuming trans_points is already defined
    points = trans_points  # Replace with your tensor if it's not defined already

    # Extracting X, Y coordinates
    x = points[0, :].numpy()
    y = points[1, :].numpy()

    # Load the image
    img = mpimg.imread('rgb_image.png')

    # Create a folder called 'saved_images' if it doesn't exist
    folder_name = 'saved_images'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Plot the image in its original size (no scaling)
    plt.imshow(img)

    # Overlay the points on the image (adjust coordinates to match image size)
    plt.scatter(x, y, c='r', marker='o', s=2)  # Reduce the size of points with s=2

    # Setting labels
    plt.xlabel('X Label')
    plt.ylabel('Y Label')

    plt.title(f'Overlay Points on 2D Image')

    # Generate a random 9-digit integer for the image filename
    random_number = random.randint(1, 999999999)
    
    # Save the plot as an image with a unique filename
    image_filename = f"{folder_name}/2D_Image_{random_number}.png"
    plt.savefig(image_filename)

    # Clear the plot to avoid overlapping with future plots
    plt.clf()

def filter_masks_pts(masks_pts, min_area=10000, min_distance=2):
    """
    Filters the segmentation masks by removing small regions and simplifying contours.

    :param masks_pts: Segmentation mask points output from SAM2.
    :param min_area: Minimum area required for a valid contour.
    :param min_distance: Minimum distance between consecutive contour points.
    :return: Filtered mask points.
    """
    filtered_masks = []

    for mask in masks_pts:
        # Convert mask to numpy array if it's a tensor
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) >= 3:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue  # Skip small areas

                # Simplify the contour by filtering close points
                filtered_contour = filter_close_points(contour, min_distance)

                if len(filtered_contour) >= 3:  # Ensure valid polygon
                    filtered_masks.append(filtered_contour)

    return filtered_masks

def filter_close_points(contour, min_distance=5):
    """
    Remove points that are closer than min_distance pixels.

    :param contour: Input contour.
    :param min_distance: Minimum distance between points.
    :return: Filtered contour.
    """
    filtered_points = [contour[0]]  # Start with the first point
    last_point = contour[0]

    for point in contour[1:]:
        if np.linalg.norm(point - last_point) >= min_distance:
            filtered_points.append(point)
            last_point = point

    return np.array(filtered_points)

# Function to map 2D points to 3D using the provided formula
def map_2d_to_3d(projected_points, depth_image, intrinsic_matrix, extrinsic_matrix, selected_points, mask):
    # Ensure input u and v are in numpy array format
    u = np.array(projected_points[:, 0]).T
    v = np.array(projected_points[:, 1]).T

    # Get the corresponding depth values for each 2D point (u, v)
    depth_values = np.array(depth_image[v.astype(int), u.astype(int)])  # (N,)

    # Normalize 2D points (convert from pixel to camera coordinates)
    inv_K = np.linalg.inv(intrinsic_matrix[:3, :3])  # Inverse of the intrinsic matrix
    homogeneous_pixels = np.column_stack((u, v, np.ones_like(u)))  # (N, 3)
    
    # Camera coordinates (in the camera frame)
    camera_coords = np.dot(inv_K, homogeneous_pixels.T).T  # (N, 3)
    camera_coords *= depth_values.reshape(-1, 1)  # Scale by depth values

    # Apply the rotation and translation (extrinsics)
    R_inv = extrinsic_matrix[:3, :3].T  # Inverse of rotation matrix
    T = extrinsic_matrix[:3, 3].reshape(3, 1)  # Translation vector (3,1)

    # Map to world coordinates using the formula
    world_coords = np.dot(camera_coords, R_inv.T) - T.T  # (N, 3)

    # Compute the average translation correction
    translation_fix = np.mean(selected_points[idx_in,:].numpy() - world_coords, axis=0)
    reprojected_org_pnts = world_coords + translation_fix

    # Apply the reprojection on the mask pixels
    mask_array = np.array(mask)

    # Get (y, x) coordinates where the mask is nonzero
    mask_coords = np.argwhere(mask_array > 0)  # Shape (N, 2), where [:,0] = y, [:,1] = x

    # Swap columns to match expected format (X first, then Y)
    mask_coords = mask_coords[:, [1, 0]]

    u = np.array(mask_coords[:, 0]).T
    v = np.array(mask_coords[:, 1]).T

    # Get the corresponding depth values for each 2D point (u, v)
    depth_values = np.array(depth_image[v.astype(int), u.astype(int)])  # (N,)

    # Normalize 2D points (convert from pixel to camera coordinates)
    homogeneous_pixels = np.column_stack((u, v, np.ones_like(u)))  # (N, 3)
    
    # Camera coordinates (in the camera frame)
    camera_coords = np.dot(inv_K, homogeneous_pixels.T).T  # (N, 3)
    camera_coords *= depth_values.reshape(-1, 1)  # Scale by depth values

    # Map to world coordinates using the formula
    world_coords = np.dot(camera_coords, R_inv.T) - T.T  # (N, 3)
    
    # Apply the correction to all points
    world_coords_corrected = world_coords + translation_fix

    return world_coords_corrected, reprojected_org_pnts

def match_and_filter_points(scene_pcd, instance_pcd, distance_threshold=0.5, apply_outlier_removal=True):
    """
    Matches each instance point to the nearest scene point while ensuring the distance does not exceed a given threshold.
    Optionally applies outlier removal to the instance point cloud before matching.

    :param scene_pcd: Open3D PointCloud of the original scene.
    :param instance_pcd: Open3D PointCloud of the segmented instance.
    :param distance_threshold: Maximum allowable distance between matched points.
    :param apply_outlier_removal: Whether to apply outlier removal before matching.
    :return: Open3D PointCloud of matched points.
    """

    # Apply outlier removal before matching if enabled
    if apply_outlier_removal:
        # Remove statistical outliers with less aggressive settings
        instance_pcd, inlier_indices = instance_pcd.remove_statistical_outlier(nb_neighbors=300, std_ratio=2.0)

        # Remove radius-based outliers (adjust for better point retention)
        instance_pcd, inlier_indices = instance_pcd.remove_radius_outlier(nb_points=50, radius=0.2)

    # Convert to NumPy arrays
    scene_points = np.asarray(scene_pcd.points)
    instance_points = np.asarray(instance_pcd.points)

    if len(instance_points) == 0 or len(scene_points) == 0:
        print("One or both point clouds are empty after filtering.")
        return 0
    
    # Build KD-Tree for the scene (large dataset)
    kdtree = cKDTree(scene_points)

    # Find the nearest neighbor in the scene for each instance point
    distances, indices = kdtree.query(instance_points, k=1)  # k=1 means the closest point

    # Apply threshold: Keep only points where distance is below the threshold
    valid_mask = distances <= distance_threshold

    # Ensure we are not accessing an empty array after filtering
    if np.sum(valid_mask) == 0:
        raise ValueError("No valid matches found within the distance threshold.")

    matched_original_points = scene_points[indices[valid_mask]]

    # Remove duplicate matches
    matched_original_points = np.unique(matched_original_points, axis=0)

    # Convert matched points back to Open3D point cloud
    matched_pcd = o3d.geometry.PointCloud()
    matched_pcd.points = o3d.utility.Vector3dVector(matched_original_points)
    print(matched_pcd)
    return matched_pcd

def generate_high_accuracy_mesh(matched_pcd, poisson_depth=11, normal_radius=0.01, max_nn=50, density_threshold=10):
    """
    Generates a high-accuracy 3D mesh using Poisson reconstruction on a matched point cloud.

    :param matched_pcd: Open3D PointCloud of matched points.
    :param poisson_depth: Depth parameter for Poisson reconstruction (higher = more details).
    :param normal_radius: Radius for normal estimation (smaller = more local detail).
    :param max_nn: Maximum nearest neighbors for normal estimation.
    :param density_threshold: Percentile threshold for removing low-density faces.
    :return: Open3D TriangleMesh.
    """

    # **Step 1: Estimate normals for better surface detail**
    matched_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn))
    matched_pcd.orient_normals_consistent_tangent_plane(k=10)  # Ensure normal consistency

    # **Step 2: Apply Poisson reconstruction with increased depth for accuracy**
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(matched_pcd, depth=poisson_depth)

    # **Step 3: Remove low-density faces to clean up the mesh**
    density_threshold_value = np.percentile(densities, density_threshold)
    vertices_to_remove = densities < density_threshold_value
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # **Step 4: Refine the mesh (Optional but improves quality)**
    mesh = mesh.filter_smooth_taubin(number_of_iterations=50)  # Apply smoothing
    mesh.remove_degenerate_triangles()  # Remove bad triangles
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    return mesh

def clean_mesh_keep_largest(mesh):
    """
    Removes all small disconnected parts from the mesh and keeps only the largest connected component.

    :param mesh: Open3D TriangleMesh.
    :return: Cleaned Open3D TriangleMesh with only the largest component.
    """

    # Compute connected components in the mesh
    triangle_clusters, cluster_sizes, _ = mesh.cluster_connected_triangles()

    # Find the largest component
    largest_cluster_idx = np.argmax(cluster_sizes)

    # Create a mask to keep only the largest component
    triangles_to_keep = np.array(triangle_clusters) == largest_cluster_idx

    # Remove all other components
    mesh.remove_triangles_by_mask(~triangles_to_keep)

    # Remove any isolated vertices
    mesh.remove_unreferenced_vertices()

    return mesh


#################################################################
############################# START #############################
#################################################################

# Example usage
# Load point cloud
pcd = o3d.io.read_point_cloud("room.ply")

print("Pick some points within the wanted area.")
picked_points_in = pick_points_and_save(pcd = pcd)  # Step 1: Pick points manually
picked_points_in = torch.tensor(picked_points_in, dtype=torch.float64)
print("Pick some points out the wanted area.")
picked_points_out = pick_points_and_save(pcd = pcd)
picked_points_out = torch.tensor(picked_points_out, dtype=torch.float64)

auto_capture_images(image_count=10, rotation_angle=5)  # Step 2: Capture images automatically with rotation adjustment

# Define paths
rgb_folder = "rgb_images"
depth_folder = "depth_images"
params_folder = "params"

# Get list of image files
image_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith(".png")])

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA is available." if torch.cuda.is_available() else "CUDA is not available.")

checkpoint_path = "sam2_hiera_large.pt"
config_path = "sam2_hiera_l.yaml"

# Loop through each image set
for i, image_file in enumerate(image_files):
    # Construct file paths
    rgb_path = os.path.join(rgb_folder, f"rgb_img_{i+1}.png")
    depth_path = os.path.join(depth_folder, f"depth_img_{i+1}.png")
    params_path = os.path.join(params_folder, f"params_{i+1}.json")

    # Load camera parameters
    with open(params_path, "r") as f:
        camera_params = json.load(f)

    # Extract intrinsic and extrinsic matrices
    intrinsic_matrix = np.array(camera_params["intrinsic"])
    extrinsic_matrix = np.array(camera_params["extrinsic"])
    print(intrinsic_matrix)
    print(extrinsic_matrix)
    depth_scale = camera_params["depth_scale"]

    print(f"Processing frame {i+1}...")

    # Load RGB image
    image = cv2.imread(rgb_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load depth image
    depth_image = cv2.imread(depth_path, -1)
    depth_tensor = torch.from_numpy(depth_image.astype(np.float64)) # Convert to float and scale by depth shift

    # Convert matrices to tensors
    pose = torch.tensor(extrinsic_matrix, dtype=torch.float64)
    depth_intrinsic = intrinsic_matrix

    # Project 3D points to 2D
    projected_points_in, idx_in = projection_3d_2d(picked_points_in, depth_intrinsic, depth_tensor, pose, device=torch.device('cpu'))
    projected_points_out, idx_out = projection_3d_2d(picked_points_out, depth_intrinsic, depth_tensor, pose, device=torch.device('cpu'))
    # Correct usage of torch.concatenate
    combined_picked_points = torch.cat((projected_points_in, projected_points_out), dim=0)

    if projected_points_in is None or torch.all(projected_points_in == 0):
        continue
    
    try:
        if checkpoint_path and config_path:
            sam2_model = build_sam2(config_path, checkpoint_path, device=device, apply_postprocessing=True)
            mask_threshold = 0.5
            max_hole_area = 2500
            max_sprinkle_area = 2000

            predictor = SAM2ImagePredictor(
                sam_model=sam2_model,
                mask_threshold=mask_threshold,
                max_hole_area=max_hole_area,
                max_sprinkle_area=max_sprinkle_area,
            )
    except Exception as e:
        raise "SAM2Error"

    # Apply segmentation
    predictor.set_image(Image.open(rgb_path))
    input_labels_in = np.ones((projected_points_in.shape[0],), dtype=np.int64)
    input_labels_out = np.zeros((projected_points_out.shape[0],), dtype=np.int64)
    combined_input_labels = np.concatenate((input_labels_in, input_labels_out))

    masks_pts, scores, logits = predictor.predict(
        point_coords=np.array(combined_picked_points, dtype=np.int64),
        point_labels=combined_input_labels,
        multimask_output=False
    )

    # Apply mask filtering
    filtered_masks_pts = filter_masks_pts(masks_pts)

    # Create a binary mask from the filtered contours
    img_height, img_width, _ = image.shape
    binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for contour in filtered_masks_pts:
        if isinstance(contour, np.ndarray) and len(contour) >= 3:
            cv2.drawContours(binary_mask, [contour.astype(np.int32)], -1, 1, thickness=cv2.FILLED)

    binary_mask = (binary_mask > 0).astype(np.uint8).reshape(img_height, img_width)

    # Modify binary_mask: Set to 0 any point that is 1 in binary_mask and 0 in masks_pts
    if masks_pts.shape == binary_mask.shape:
        binary_mask[(binary_mask == 1) & (masks_pts == 0)] = 0

    # Plot mask overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(binary_mask, cmap="jet", alpha=0.5)
    plt.title(f"Binary Mask Overlay - Frame {i+1}")
    plt.axis("off")
    plt.show()

    # Load depth image
    depth_tensor = depth_tensor / depth_scale  # Convert to float and scale by depth shift

    # Reproject 2D points back to 3D
    reprojected_3d_points, reprojected_org_pnts = map_2d_to_3d(projected_points_in, depth_tensor, intrinsic_matrix, extrinsic_matrix, picked_points_in, binary_mask)

    # Save reprojected points
    np.savetxt(f"reprojected_3d_points_{i+1}.xyz", reprojected_3d_points, fmt="%.8f")

    # Compute Pearson correlation for reprojected points
    try:
        correlations = {
            'X': pearsonr(reprojected_org_pnts[:, 0], picked_points_in[idx_in, 0])[0],
            'Y': pearsonr(reprojected_org_pnts[:, 1], picked_points_in[idx_in, 1])[0],
            'Z': pearsonr(reprojected_org_pnts[:, 2], picked_points_in[idx_in, 2])[0]
        }
        correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Pearson Correlation'])
        print(correlation_df)
    except:
        continue
    
    # Load scene and instance point clouds
    scene_pcd = o3d.io.read_point_cloud("room.xyz")
    instance_pcd = o3d.io.read_point_cloud(f"reprojected_3d_points_{i+1}.xyz")

    # Match points with filtering
    matched_pcd = match_and_filter_points(scene_pcd, instance_pcd, distance_threshold=0.5, apply_outlier_removal=True)
    if matched_pcd == 0:
        continue
    
    o3d.io.write_point_cloud(f"matched_original_pcd_{i+1}.ply", matched_pcd)

    print(f"Processing completed for frame {i+1}.")
    projected_points_in = None
    
# Define folder and output file
output_pcd_file = "combined_matched_original_pcd.ply"

# Get list of all matched point cloud files
matched_pcd_files = sorted([f for f in os.listdir() if f.startswith("matched_original_pcd_") and f.endswith(".ply")])

# Initialize an empty Open3D point cloud
combined_pcd = o3d.geometry.PointCloud()

# Loop through each matched point cloud file and merge unique points
all_points = set()
for pcd_file in matched_pcd_files:
    print(f"Processing {pcd_file}...")

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    # Convert points to tuple format to remove duplicates
    unique_points = {tuple(point) for point in points}
    
    # Merge new unique points into the existing set
    all_points.update(unique_points)

# Convert back to Open3D point cloud format
combined_pcd.points = o3d.utility.Vector3dVector(np.array(list(all_points)))

# Save the combined unique point cloud
o3d.io.write_point_cloud(output_pcd_file, combined_pcd)

# Display completion message
print(f"All matched point clouds combined and saved to {output_pcd_file}.")
