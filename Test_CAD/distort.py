### In this module we simulate distortions in order to create a training data set for further processing. ###

### IMPORTS ###
import os
import sys
import argparse
import numpy as np
import glob
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
import random
import re
import copy


def normalize_point_cloud(pcd, target_scale):
    # Calculate scaling factor
    current_scale = np.max(np.asarray(pcd.points), axis=0) - np.min(np.asarray(pcd.points), axis=0)
    scaling_factor = target_scale / np.max(current_scale)

    # Apply scaling to coordinates
    scaled_points = scaling_factor * np.asarray(pcd.points)
    
    # Create a new PointCloud with scaled coordinates
    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(scaled_points)

    return normalized_pcd


def visualize(*args, normal_view = False, rotate = False, VERTICAL_ROT = -0.5, HORIZONTAL_ROT = 0.5):
    '''
        Visualize PointCloud or TriangleMesh objects in 3D space. No rotation unless rotate is set to True, and VERTICAL_ROT, HORIZONTAL_ROT not 0.0.
    Args:
        *args (PointCloud/TriangleMesh): point cloud or mesh (1+ of same type)
        rotate (bool) : allow rotation, defaults to False
        VERTICAL_ROT (float) :  vertical rotation factor
        HORIZONTAL_ROT (float) : horizontal rotation factor
    '''
    def rotate_view(vis):
        """ Animation rotation controls """
        ctr = vis.get_view_control()
        ctr.rotate(HORIZONTAL_ROT, VERTICAL_ROT)
        return False

    args = list(args)

    if rotate:
        if normal_view:
            o3d.visualization.draw_geometries_with_animation_callback(args, rotate_view, point_show_normal=True)
        else:
            o3d.visualization.draw_geometries_with_animation_callback(args, rotate_view)
    else:
        if normal_view:
            o3d.visualization.draw_geometries(args, point_show_normal=True)
        else:
            o3d.visualization.draw_geometries(args)


def create_uniform_sampled_cloud_from_mesh(mesh, nr_points = 500,  poisson = False,  factor = 5):
    '''
        Takes a mesh, returns sampled model of nr_points points. If Poisson is allowed, evenly distribute the points by removing samples from the initial sampled cloud, with nr_points/factor as new sampling criterion.
    Args:
        mesh (TriangleMesh obj) : mesh object to be sampled
        poisson (bool) : allow poisson sampling: each point has approx. same distance to the neighbouring points, defaults to False
        nr_points (int) : total nr of points of sampling
        factor (int):  used for sample elimination (nr_points/factor as new sampling criterion), defaults to 5

    Returns:
        pcd (PointCloud obj) : uniformly sampled point cloud
    '''
    pcd = mesh.sample_points_uniformly(number_of_points=nr_points) # Sample uniformly
    if poisson:
        pcd = mesh.sample_points_poisson_disk(number_of_points=int(nr_points/factor), pcl=pcd)
    return pcd


def read_mesh(filepath, normals = True):
    '''
        Reads  e.g. .ply, .stl files file paths, returns TriangleMesh object. Calculates normals unless normals is set to False.
    Args:
        filepath (str) : file path
        normals (bool) : calculate normals, defaults to True

    Returns:
        mesh (type: TriangleMesh object)
    '''
    mesh = o3d.io.read_triangle_mesh(filepath)
    if normals:
        mesh.compute_vertex_normals()
    return mesh


def apply_gaussian(ref, printing = False):
    '''
        Creating distorted clouds by applying SNR-based shifts. Threat to accuracy. 
    Args:
        ref (str) : reference as point cloud
        printing (bool): check for printing

    Returns:
        gaussian_pcds (list) : list of distorted pcds
    '''
    print("Apply SNR-based gaussian...")
    ref_points = np.asarray(ref.points)
    ref_num_points = len(ref.points)
    gaussian_pcds = []
    # Define dB (noise power) values - typically between 30-50:
    db_values = [50,45,40,35,30]
    # Calculate the signal power (variance along each dimension)
    signal_power = np.var(np.array(ref.points), axis=0)
    # Calculate the noise power based on SNR
    for db in db_values:
        # Calculate the noise power in linear scale based on the desired SNR (in dB)
        snr_linear = 10 ** (db / 10)
        noise_power_linear = signal_power / snr_linear
        # Generate Gaussian noise with the calculated noise power
        noise = np.random.normal(scale=np.sqrt(noise_power_linear), size=ref_points.shape)
        # Create a distorted point cloud by adding noise to the reference point cloud
        distorted_points = ref_points + noise
        distorted_pcd = o3d.geometry.PointCloud()
        distorted_pcd.points = o3d.utility.Vector3dVector(distorted_points)
        if printing:
            print("Gaussian with dB:"+str(db))
            visualize(distorted_pcd)
        gaussian_pcds.append(distorted_pcd)
    return gaussian_pcds

def apply_outlier_scattering(ref, printing = False):
    '''
    Apply distortion to a point cloud by adding outlier clusters of various sizes at different scattering levels.
    Args:
        ref (o3d.geometry.PointCloud): reference point cloud
        printing (bool): printing check

    Returns:
        distorted_pcds (list): list of distorted point clouds
    '''
    distorted_pcds = []
    # Get the bounding box of the original point cloud
    bbox = ref.get_axis_aligned_bounding_box()
    # Get the dimensions of the bounding box
    x_min, y_min, z_min = bbox.get_min_bound()
    x_max, y_max, z_max = bbox.get_max_bound()
    # Convert Open3D point cloud to numpy array
    original_points = np.asarray(ref.points)
    # Scattering levels for outlier clusters
    scattering_levels = [0.00001, 0.00003, 0.00005, 0.00007, 0.0001]
    # Ratios of big and small clusters
    big_cluster_ratio = 0.5
    small_cluster_ratio = 0.5
    # Initialize distorted_points with the original points
    distorted_points = original_points.copy()
    for scattering_level in scattering_levels:
        num_outliers = int(original_points.shape[0] * scattering_level)
        # Calculate the number of big and small clusters based on the ratios
        num_big_clusters = int(num_outliers * big_cluster_ratio)
        num_small_clusters = num_outliers - num_big_clusters
        # Generate big clusters
        for _ in range(num_big_clusters):
            cluster_size = np.random.randint(500, 1000)  # Big cluster size between 15k and 30k
            cluster_center = np.random.uniform(
                (x_min, y_min, z_min),
                (x_max, y_max, z_max),
                size=(1, 3))
            cluster_points = np.random.normal(
                cluster_center,
                scale=(0.1, 0.1, 0.1),
                size=(cluster_size, 3))
            distorted_points = np.vstack((distorted_points, cluster_points))
        # Generate small clusters
        for _ in range(num_small_clusters):
            cluster_size = np.random.randint(1, 500)  # Small cluster size between 1 and 100
            cluster_center = np.random.uniform(
                (x_min, y_min, z_min),
                (x_max, y_max, z_max),
                size=(1, 3))
            cluster_points = np.random.normal(
                cluster_center,
                scale=(0.1, 0.1, 0.1),
                size=(cluster_size, 3))
            distorted_points = np.vstack((distorted_points, cluster_points))
        # Create a new Open3D point cloud with the distorted points
        distorted_pcd = o3d.geometry.PointCloud()
        distorted_pcd.points = o3d.utility.Vector3dVector(distorted_points.copy())  # Copy to avoid modifying previous clouds
        if printing: 
            print("Outlier scattering with scattering level:", scattering_level)
            visualize(distorted_pcd)
        distorted_pcds.append(distorted_pcd)
    return distorted_pcds

def apply_local_missing(ref, printing = False):
    '''
    Creating distorted clouds by eliminating patches. We define a space anchor of 0.3% size of bounding box. Points in selected anchors are removed. Threat to completeness.
    Args:
        ref (o3d.geometry.PointCloud) : reference as point cloud
        printing (bool): printing check

    Returns:
        local_missing_pcds (list) : list of distorted pcds
    '''
    print("Apply local missing...")
    ref_num_points = len(ref.points)

    # Calculate the bounding box size
    bounding_box = ref.get_axis_aligned_bounding_box()
    bbox_size = np.asarray(bounding_box.get_max_bound()) - np.asarray(bounding_box.get_min_bound())
    bbox_diag_length = np.linalg.norm(bbox_size)
    # Define anchor size and the distortion levels given how many anchors will be added
    anchor_size = 0.15 * bbox_diag_length
    num_anchors_targets = [1, 1, 1, 1, 1]
    # Generate anchors based on the number of anchors per distortion level
    anchors = []
    for i in range(len(num_anchors_targets)):
        num_anchors = num_anchors_targets[i]
        level_anchors = []
        for _ in range(num_anchors):
            # Choose a random point index
            random_point_index = np.random.randint(0, ref_num_points)
            random_point = np.asarray(ref.points[random_point_index])

            # Calculate anchor position
            anchor = o3d.geometry.AxisAlignedBoundingBox()
            anchor.min_bound = random_point - 0.5 * anchor_size
            anchor.max_bound = random_point + 0.5 * anchor_size
            level_anchors.append(anchor)
        anchors.append(level_anchors)
    # Initialize a copy of the reference point cloud for distortion
    distorted_ref = copy.deepcopy(ref)
    # Apply local missing distortion progressively
    local_missing_pcds = []
    for anchor_list in anchors:
        for anchor in anchor_list:
            anchor_min = np.asarray(anchor.get_min_bound())
            anchor_max = np.asarray(anchor.get_max_bound())
            # Remove anchor bounded points:
            indices_to_remove = []
            for index, point in enumerate(distorted_ref.points):
                if np.all(point >= anchor_min) and np.all(point <= anchor_max):
                    indices_to_remove.append(index)
            distorted_ref.points = o3d.utility.Vector3dVector(
                np.delete(np.asarray(distorted_ref.points), indices_to_remove, axis=0))
        # Append the current state of the distorted point cloud
        local_missing_pcds.append(copy.deepcopy(distorted_ref))
        if printing:
            print("Local missing with anchor number:"+str(len(anchor_list)))
            visualize(distorted_ref)
    return local_missing_pcds

def ignore_apply_uniform_shift(ref):
    '''
        Creating distorted clouds by shifting. Shifting is applied to 10-20-30-40-50% of randomly selected points, with shifting ranges of -/+ 0.5-1-2-3-4% respectively of bounding box.  Threat to accuracy. 
    Args:
        ref (str) : reference as point cloud

    Returns:
        uniform_shift_pcds (list) : list of distorted pcds
    '''
    print("Apply uniform shift...")
    ref_num_points = len(ref.points)
    # Define the random selection proportions:
    proportions = [0.1, 0.2, 0.3, 0.4, 0.5]
    size_targets =  [int(ref_num_points*p) for p in proportions]
    shifting_targets = [0.005, 0.01, 0.02, 0.03, 0.04]
    # Calculate the bounding box size
    bounding_box = ref.get_axis_aligned_bounding_box()
    bbox_size = np.asarray(bounding_box.get_max_bound()) - np.asarray(bounding_box.get_min_bound())
    bbox_diag_length = np.linalg.norm(bbox_size)
    shift_targets = [shift * bbox_diag_length for shift in shifting_targets]
    uniform_shift_pcds = []
    i = 0
    for target in size_targets:
        ind = random.sample(range(ref_num_points), target)
        pos_distorted_ref = ref.select_by_index(ind)
        neg_distorted_ref = ref.select_by_index(ind)
        noise = np.random.normal(0, shift_targets[i], size=(ref_num_points, 3))
        pos_distorted_ref.points = o3d.utility.Vector3dVector(np.asarray(ref.points) + noise)
        neg_distorted_ref.points = o3d.utility.Vector3dVector(np.asarray(ref.points) - noise)
        uniform_shift_pcds.append(pos_distorted_ref)
        uniform_shift_pcds.append(neg_distorted_ref)
        i += 1
        visualize(pos_distorted_ref)
        visualize(neg_distorted_ref)
    return uniform_shift_pcds

def ignore_apply_reconstruction(ref, sampling_rate):
    '''
        Creating distorted clouds by adding outliers by reconstruction of downsampled point clouds. Varies over 3 depth levels. Threat to accuracy. 
    Args:
        ref (str) : reference as point cloud

    Returns:
        reconstructed_pcds (list) : list of distorted pcds
    '''
    print("Apply reconstruction...")
    depth_levels = [3,5,8]
    # Fetch downsampled pcds:
    downsampled_pcds = apply_downsampling(ref)
    # Reconstruct using poisson surface reconstruction:
    reconstructed_pcds = []
    for pcd in downsampled_pcds:
        for depth in depth_levels:
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                reconstructed, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
            # Sample CAD, turning mesh into point cloud:
            ref_reconstructed = create_uniform_sampled_cloud_from_mesh(reconstructed, nr_points = sampling_rate, poisson = True, factor = 1)
            reconstructed_pcds.append(ref_reconstructed)
            visualize(ref_reconstructed)
    return reconstructed_pcds

def ignore_apply_downsampling(ref):
    '''
        Creating distorted clouds by randomly downsampling by removing 10%-30%-50%-70%-90% points from point cloud. This induces missing values.  Threat to completeness. 
    Args:
        ref (str) : reference as point cloud

    Returns:
        downsampled_pcds (list) : list of distorted pcds
    '''
    print("Apply downsampling...")
    ref_num_points = len(ref.points)
    # Define the downsampling percentages
    percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    downsampling_targets = [int(ref_num_points*p) for p in percentages]
    downsampled_pcds = []
    for target in downsampling_targets:
        # Select -target- indices from  reference:
        ind = random.sample(range(ref_num_points), target)
        distorted_ref = ref.select_by_index(ind)
        # Visualize the downsampled point cloud
        downsampled_pcds.append(distorted_ref)
        visualize(distorted_ref)
    return downsampled_pcds

def distort(cad_path, output):
    '''
        Takes a CAD path and creates a set of distorted variations using multiple geometry-based methods. Save to output folder.
    Args:
        cad_path (str) : path to CAD mesh
        output (str) : name of output folder
    '''
    print("Action in progress: reading and sampling reference...")
    # Read reference:
    cad = read_mesh(cad_path)
    # Pick sampling rate:
    sampling_rate = 100_000 # Default
    sampling_rate = int(sampling_rate)
    # Sample CAD, turning mesh into point cloud:
    ref_pcd = create_uniform_sampled_cloud_from_mesh(cad, nr_points = sampling_rate, poisson = True, factor = 1)
    # Reference to be add distortions on:
    ref = copy.deepcopy(ref_pcd)
    # For experiments - allow normalization of reference: 
    # target_scale = 10  # Set your desired target scale
    # ref = normalize_point_cloud(ref, target_scale)
    # ref_pcd = normalize_point_cloud(ref_pcd, target_scale)
    print("Action in progress: applying distortions...")
    # ______________GEOMETRY DISTORTIONS______________
    # Gaussian geometry shifting - Threat to accuracy
    gaussian_pcds = apply_gaussian(ref, printing = False) 
    # Outlier scattering - Threat to validity.
    outlier_pcds = apply_outlier_scattering(ref, printing = False)
    # Local missing - Threat to completeness.
    local_missing_pcds = apply_local_missing(ref, printing  = False)
    distortion_data = {}
    distortion_data["Raw"] = [ref_pcd]
    distortion_data["Accuracy"] = gaussian_pcds
    distortion_data["Validity"] = outlier_pcds
    distortion_data["Completeness"] = local_missing_pcds
    print(f"Action in progress: saving distorted clouds to {output} folder, with subfolders belonging to each distortion...")
    # Create the parent folder "pcds" if it doesn't exist
    os.makedirs(output, exist_ok=True)
    # Iterate through the distortion types and their corresponding point clouds
    for distortion_type, point_clouds in distortion_data.items():
        distortion_folder = os.path.join(output, distortion_type)
        os.makedirs(distortion_folder, exist_ok=True)
        for idx, point_cloud in enumerate(point_clouds):
            point_cloud_path = os.path.join(distortion_folder, f"point_cloud_{idx}.pcd")
            o3d.io.write_point_cloud(point_cloud_path, point_cloud)
    print(f"Distortion process finished.")

def main(argv = None):
    ''' Make sure that the python file is in the same folder as the .stl models.
    Args:
        argv (list): list of arguments
    '''
    stl_files = glob.glob("*.stl")
    # Distort .stl models in the current folder:
    for stl_filename in stl_files:
        cad_path = stl_filename
        output_name = stl_filename[:-4] # Remove .stl ending
        distort(cad_path, output_name)

if __name__ == '__main__':
    main()
