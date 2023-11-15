# -*- coding: utf-8 -*-
'''This file should be invoked as python -m 3D_DaVa'''

### IMPORTS ###
import os
import glob
from . import processing as proc
from . import alignment as ali
from . import inout
from functools import lru_cache
import numpy as np
import sys
import argparse
import open3d as o3d
import copy
import time
import matplotlib.pyplot as plt
import datetime
import random
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew
from scipy.stats import mode
import re

### GLOBAL VARIABLES ###
original = None
point_scores = None
large = False
downsized = None
downsized_links = None
downsized_voxel_size = None
mean_nb = None
automatic = False 
snapshotting = False    
visualization = False
saving = False

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

@lru_cache
def process_with_reference(input_file:str, reference_file:str, *args):
    '''
        Point Cloud Data Validation - with reference
    Args:
        input_file (str) : path of point cloud to analyse
        reference_file (str) : path of reference point cloud
        *args (str) : name of output file - optional
    Returns:
        metric (list) : list containing accuracy, validity and completeness of point cloud
    '''
    # TODO: For intermediary metric: turn point cloud (not downsized) to mesh using voxel modelling
    # TODO: Try other R-PCQA approaches (see literature-Graph Similarity) ...
    # TODO: Try other registration methods (ISS/Curvature/Graph)

    global original
    global large
    global snapshotting
    global visualization
    global automatic
    global downsized
    global downsized_links
    global downsized_voxel_size
   
    # Keep track of processing time:
    start_time = time.time()

    print("Action in progress: process cloud without reference for initial denoisal...")
    _ = process_without_reference(input_file)
    # Cloud to be worked on:
    pcd = copy.deepcopy(original)
    # Get partly denoised cloud:
    scaled_probs = proc.minmax_scale(point_scores)
    # Use IQR rule as threshold:
    Q1 = np.percentile(scaled_probs, 25) # First quartile
    Q3 = np.percentile(scaled_probs, 75) # Third quartile
    IQR = Q3 - Q1
    noise_threshold = Q3 + 1.5 * IQR
    noise_ind = np.where(scaled_probs >= noise_threshold)[0]
    denoised_pcd = proc.get_cloud_by_index(original, noise_ind, invert =True)
    if visualization:
        print("Denoised:")
        inout.visualize(denoised_pcd) # Test visual
    #print("Denoised:")
    #inout.visualize(denoised_pcd) # Test visual
    #inout.visualize_differences(pcd, denoised_pcd) # Test visual
    

    print("Action in progress: read and sample reference...")
    # Read and sample reference:
    original_ref = inout.read_mesh(reference_file)
    SAMPLING_RATE = len(original.points) * 2 # 200_000
    ref_pcd = proc.create_uniform_sampled_cloud_from_mesh(original_ref, nr_points = SAMPLING_RATE)
    # Reference to be worked on:
    ref = copy.deepcopy(ref_pcd)
    NUM_RAW_POINTS_REF = len(ref.points)
    if snapshotting:
        inout.snapshot(ref, name = "sampled_ref")
    if visualization:
        inout.visualize(ref)

    # Normalize size - for experiments only!
    # target_scale = 10  # Set your desired target scale
    # ref_pcd = normalize_point_cloud(ref_pcd, target_scale)
    # ref = normalize_point_cloud(ref, target_scale)

    # '''
    print("Action in progress: scaling point clouds to same size...")
    # Rescale pcd to mesh size. Use denoised cloud for this for better results.
    pcd_max_bound = denoised_pcd.get_max_bound()
    pcd_min_bound = denoised_pcd.get_min_bound()
    pcd_dims = pcd_max_bound - pcd_min_bound
    ref_max_bound = ref.get_max_bound()
    ref_min_bound = ref.get_min_bound()
    ref_dims = ref_max_bound - ref_min_bound
    # Check which boundary box is bigger by volume:
    vol1 = np.prod(pcd_dims)
    vol2 = np.prod(ref_dims)
    # Scale only point cloud:
    scaling_factor = max(ref_dims) / max(pcd_dims)
    # Scale point cloud and denoised point cloud:
    pcd.scale(scaling_factor, center=pcd.get_center())
    denoised_pcd.scale(scaling_factor, center=denoised_pcd.get_center())
    if visualization:
        inout.visualize_differences(pcd, ref)

    if not automatic:
        print("Action in progress: eventual mirror transformation for symmetric point clouds ...")
        # In case orientation is wrong, one getst he choice of flipping over the first two axis.
        inout.visualize_differences(pcd, ref)
        mirror_check = input("Mirror PCD? (y/n) or ENTER to continue:")
        mirror_check = mirror_check.lower()
        yes_choices = ["y", "yes"]
        no_choices = ["n", "no"]
        possible_choices = [""] + yes_choices + no_choices
        while mirror_check not in possible_choices:
            mirror_check = input("Type error! Mirror PCD? (y/n) or ENTER to continue:")
            mirror_check = mirror_check.lower()
        if mirror_check in yes_choices:
            # Compute the principal component analysis (PCA) - for original pcd:
            mean, covariance = pcd.compute_mean_and_covariance()
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            largest_eigenvector_index = np.argmax(eigenvalues)
            second_largest_eigenvector_index = np.argsort(eigenvalues)[-2]
            mirrored_points = np.array(pcd.points)
            mirrored_points[:, [largest_eigenvector_index, second_largest_eigenvector_index]] *= -1
            mirrored_point_cloud = o3d.geometry.PointCloud()
            mirrored_point_cloud.points = o3d.utility.Vector3dVector(mirrored_points)
            pcd = mirrored_point_cloud
            if visualization:
                inout.visualize_differences(pcd, ref)
            # Compute the principal component analysis (PCA) - for denoised pcd:
            mean, covariance = denoised_pcd.compute_mean_and_covariance()
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            largest_eigenvector_index = np.argmax(eigenvalues)
            second_largest_eigenvector_index = np.argsort(eigenvalues)[-2]
            mirrored_points = np.array(denoised_pcd.points)
            mirrored_points[:, [largest_eigenvector_index, second_largest_eigenvector_index]] *= -1
            mirrored_point_cloud = o3d.geometry.PointCloud()
            mirrored_point_cloud.points = o3d.utility.Vector3dVector(mirrored_points)
            denoised_pcd = mirrored_point_cloud
            if visualization:
                inout.visualize_differences(denoised_pcd, ref)

    print("Action in progress: downsize/voxelize denoised point cloud and reference ...")
    # Original downsized variables:
    # down_pcd = downsized 
    # pcd_links = downsized_links
    voxel_size_pcd = downsized_voxel_size
    # Get ref and denoised downsized variables:
    CONSTRAINT = 0.01
    # Downsize given boundary box constraint - ref:
    voxel_size_ref = round(max(ref.get_max_bound() - ref.get_min_bound()) * CONSTRAINT, 8)
    down_ref, _ = proc.downsample_and_trace_cloud(ref, voxel_size_ref)
    voxel_size_denoised_pcd = round(max(denoised_pcd.get_max_bound() - denoised_pcd.get_min_bound()) * CONSTRAINT, 8)
    down_denoised_pcd, _ = proc.downsample_and_trace_cloud(denoised_pcd, voxel_size_denoised_pcd)
    if snapshotting:
        inout.snapshot(down_ref, name = "downsized_ref")
    if visualization:
        inout.visualize(down_ref)

    print("Action in progress: fast global alignment using feature matching...")
    # Use downsized and denoised versions if the clouds are too large:
    if large:
        alignment_ref = down_ref
        alignment_pcd = down_denoised_pcd
        alignment_voxel_size = voxel_size_denoised_pcd
    else:
        alignment_pcd = denoised_pcd
        alignment_ref = ref
        alignment_voxel_size = voxel_size_denoised_pcd # If it does not work, try voxel_size_pcd
    # TODO: Non-blocking visualization (http://www.open3d.org/docs/latest/tutorial/visualization/non_blocking_visualization.html)
    ITER = 5000  # The larger the better -> but more time use
    MAX_NN = 50
    LARGE_FACTOR = 5
    max_bound = alignment_pcd.get_max_bound()
    min_bound = alignment_pcd.get_min_bound()
    max_dist = (proc.get_l2_distance(min_bound, max_bound) * 0.1) / 2
    normal_radius = alignment_voxel_size * LARGE_FACTOR if large else max_dist 
    feature_radius = alignment_voxel_size * LARGE_FACTOR if large else max_dist
    max_nn_normal = MAX_NN * LARGE_FACTOR if large else MAX_NN
    max_nn_feature = MAX_NN * LARGE_FACTOR if large else MAX_NN
    solution = [ITER, normal_radius, feature_radius, max_nn_normal, max_nn_feature]
    distances = alignment_pcd.compute_point_cloud_distance(alignment_ref)
    lowest_distance = np.min(distances)
    bounding_box_diagonal = proc.get_l2_distance(min_bound, max_bound)
    overlap_rate = bounding_box_diagonal/lowest_distance
    # Define custom tolerances
    relative_tolerance = 0.1  # 10%
    absolute_tolerance = 0.5  # 0.5 units
    # If overlapping clouds, choose thresholds according to lowest distance between closest points
    if overlap_rate >= 1:
        if np.isclose(bounding_box_diagonal, overlap_rate, rtol=relative_tolerance, atol=absolute_tolerance):
            threshold = lowest_distance
        else:
            threshold = lowest_distance/2
    else:
        threshold = (bounding_box_diagonal * overlap_rate)/2
    threshold = threshold / 2 # Change in paper
    global_result = ali.global_fast_alignment(alignment_pcd, alignment_ref, solution, threshold)
    global_trans = global_result.transformation
    vis_alignment_pcd = copy.deepcopy(alignment_pcd)
    vis_alignment_pcd.transform(global_trans)
    if visualization:
        inout.visualize_differences(vis_alignment_pcd, alignment_ref)

    
    print("Action in progress: registration refinement using ICP local alignment (P2P)...")
    icp_trans = ali.icp_P2P_registration(alignment_pcd, alignment_ref, global_trans, distance_threshold=threshold)
    eva = ali.evaluate(alignment_pcd, alignment_ref, threshold, icp_trans)
    fitness = eva.fitness
    print("Point cloud registration fitness: ", fitness)
    pcd.transform(icp_trans) # Transform actual (original) point cloud 
    if visualization:
        inout.visualize_differences(pcd, ref)
    #print("Alignment:")
    #inout.visualize_differences(pcd, ref)
    # '''

    print("Action in progress: distortion detection based on euclidean distance (P2Point)...")
    # TODO: Use downsized if large and link
    # Calculate distances between downsized digitized scan and reference
    distances = pcd.compute_point_cloud_distance(ref)
    distances = np.asarray(distances)
    # Update scores (the lowest the distance, the better --> lower score):
    update_scores(distances, inverse = False)

    print("Action in progress: validity estimated from reference distances and previous scores (P2Point)...")
    # TODO: Use downsized if large and link
    # Works if right-skewed (which is typical):
    Q1 = np.percentile(distances, 25) # First quartile
    Q3 = np.percentile(distances, 75) # Third quartile
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    outlier_ind = np.where(distances > outlier_threshold)[0]
    outliers_pcd = proc.get_cloud_by_index(original, outlier_ind) 
    non_outliers_pcd = proc.get_cloud_by_index(original, outlier_ind, invert = True) 
    NUM_OUTLIERS = len(outliers_pcd.points)
    VALIDITY = 1.0 - (NUM_OUTLIERS/len(original.points))
    if snapshotting:
        inout.snapshot(outliers_pcd, name="outliers")
    if visualization:
        inout.visualize(outliers_pcd)
    #print("Visualize outliers (vs non-outliers):")
    #inout.visualize_differences(non_outliers_pcd, outliers_pcd)
    #print("Proportion outliers equal to" + str(len(outliers_pcd.points)) + " out of " +  str(len(original.points)) + "points in original pcd.")

    print("Action in progress: completeness estimated from reference distances (P2Point)...")
    # TODO: Use downsized if large and link
    # Calculate distances between downsized digitized scan and reference
    distances = ref.compute_point_cloud_distance(pcd)
    distances = np.asarray(distances)
    Q1 = np.percentile(distances, 25) # First quartile
    Q3 = np.percentile(distances, 75) # Third quartile
    IQR = Q3 - Q1
    missing_values_threshold = Q3 + 1.5 * IQR
    missing_values_indices = np.where(distances > missing_values_threshold)[0]
    missing_values_ref_pcd = ref.select_by_index(missing_values_indices)
    non_missing_values_ref_pcd = ref.select_by_index(missing_values_indices, invert = True)
    NUM_MISSING_VALUES = len(missing_values_ref_pcd.points)
    COMPLETENESS = 1.0 - (NUM_MISSING_VALUES/NUM_RAW_POINTS_REF)
    if snapshotting:
        inout.snapshot(missing_values_ref_pcd, name="missing_values")
    if visualization:
        inout.visualize(missing_values_ref_pcd)
    #print("Visualize missing values in REF (vs non-missed values in REF):")
    #inout.visualize_differences(non_missing_values_ref_pcd, missing_values_ref_pcd)
    #print("Proportion missed values equal to" + str(len(missing_values_ref_pcd.points)) + " out of " +  str(len(original.points)) + "points in original pcd.")
    

    print("Action in progress: distortion detection based on point-to-plane euclidean distances (P2Plane)...")
    # TODO: Use downsized if large and link
    scene = o3d.t.geometry.RaycastingScene()
    mesh_ids = {}
    ref_mesh = copy.deepcopy(original_ref)
    reference = o3d.t.geometry.TriangleMesh.from_legacy(ref_mesh)
    mesh_ids[scene.add_triangles(reference)] = 'ref'
    query_points = np.array(pcd.points).astype(np.float32)
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(), axis=-1)
    rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1
    distance[is_inside] *= -1
    signed_distances = distance # Negative values mean points within a mesh.
    # Update scores (lower distance --> lower score):
    update_scores(signed_distances, inverse = False)

    # Get accuracy:
    scaled_probs = proc.minmax_scale(point_scores)
    # Use IQR rule as threshold:
    Q1 = np.percentile(scaled_probs, 25) # First quartile
    Q3 = np.percentile(scaled_probs, 75) # Third quartile
    IQR = Q3 - Q1
    noise_threshold = Q3 + 1.5 * IQR
    noise_ind = np.where(scaled_probs >= noise_threshold)[0]
    # Finding indices in noise_ind that are not in outlier_ind
    noise_ind = np.setdiff1d(noise_ind, outlier_ind)
    noise = proc.get_cloud_by_index(original, noise_ind)
    non_noise = proc.get_cloud_by_index(original, noise_ind, invert=True)
    NUM_NOISE = len(noise.points)
    ACCURACY = 1.0 - (NUM_NOISE/len(original.points))
    #print("Visualize noise (vs non-noise):")
    #inout.visualize_differences(non_noise, noise)
    #print("Proportion noise equal to" + str(len(noise.points)) + " out of " +  str(len(original.points)) + "points in original pcd.")

    # End processing time
    end_time = time.time()
    rpcqa_time = datetime.timedelta(seconds=end_time - start_time)
    print("Processing time (RPCQA):" +  str(rpcqa_time))

    if saving:
        # If save --> obtain clean:
        noise_out_ind = list(noise_ind) + list(outlier_ind)
        noise_out_ind = set(noise_out_ind)
        noise_out_ind = np.array(list(noise_out_ind))
        clean = proc.get_cloud_by_index(original, noise_out_ind, invert=True)
        inout.write_cloud("filtered_pcd.pcd", clean) # Write on current path
        #inout.visualize(clean)
    return [ACCURACY, VALIDITY, COMPLETENESS]


#____________________________________NR-PCQA____________________________________
def nrpcqa_voxelization(pcd, snapshotting = False, visualization = False, shades = 25):
    '''
        Downsize big models and assess by voxel statistics. Boundary box constraints are applied for adaptive voxel sizing.
    Args:
        pcd (type: PointCloud object) : point cloud to downsample
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
        shades (int) : bar number in plot, defaults to 25
    '''
    global downsized
    global downsized_voxel_size
    global downsized_links

    # Constrain voxel size to <constaint> of boundary box:
    CONSTRAINT = 0.01
    downsized_voxel_size = round(max(pcd.get_max_bound() - pcd.get_min_bound()) * CONSTRAINT, 8)
    # Downsize given boundary box-based constraint:
    downsized, corr_inds = proc.downsample_and_trace_cloud(pcd, downsized_voxel_size)
    # Store link between voxel and set of points it represents e.g. {voxel_ind : [actual_point_ind, ...]}:
    pcd_voxel_inds = range(len(corr_inds))
    pcd_actual_inds = [list(corr_inds[i]) for i in pcd_voxel_inds]
    downsized_links = dict(zip(pcd_voxel_inds, pcd_actual_inds))
    # Get number of points per voxel (with and without index):
    values = [len(v) for v in list(downsized_links.values())]
    # Color voxelized cloud according to number of neighbours:
    colored_down, color_range = proc.color_cloud_rainbow(downsized, values, shades = shades)
    # Update scores:
    update_scores(values, downsized = True)
    if snapshotting:
        inout.snapshot(downsized, name = "downsized")
        inout.snapshot(colored_down, name = "voxelization")
        inout.plot_values_by_color(values, color_range, x_label="Number contained points per voxel", y_label="Number of voxels", save = True, name="plot_voxelization")
    if visualization:
        inout.visualize(downsized)
        inout.visualize(colored_down)
        inout.plot_values_by_color(values, color_range, x_label="Number contained points per voxel", y_label="Number of voxels", show=True, name="plot_voxelization")
   

def nrpcqa_modelling(pcd, snapshotting = False, visualization = False):
    '''
        Turns a point cloud to a mesh by 3D-modelling voxels and clusters connected areas / finds low-connection areas to calculate metrics related to density.
    Args:
        pcd (type: PointCloud object) : point cloud to model
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
    '''
    global downsized
    global downsized_voxel_size
    global downsized_links
    global point_scores

    # Get mesh and triangles (indices of vertices):
    v_mesh = proc.point_to_mesh(pcd, downsized_voxel_size)
    v_mesh_triangles =  np.asarray(v_mesh.triangles)
    # Clustering connected triangles ...
    triangle_cluster_ids, cluster_n_triangles, cluster_area = v_mesh.cluster_connected_triangles()
    triangle_cluster_ids = np.asarray(triangle_cluster_ids)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    # Remove small clusters (small = 90% reduction in number of triangles compared to biggest cluster):
    biggest_cluster_ind = np.argmax(cluster_n_triangles)
    CONSTRAINT = int(cluster_n_triangles[biggest_cluster_ind] * 0.1)
    # big_cluster_triangles = cluster_n_triangles[triangle_cluster_ids] > constraint
    small_cluster_triangles = cluster_n_triangles[triangle_cluster_ids] <= CONSTRAINT
    small_cluster_num_triangles = np.unique(cluster_n_triangles[cluster_n_triangles <= CONSTRAINT])
    if len(small_cluster_num_triangles) != 0:
        max_small_cluster_num_triangles = np.max(small_cluster_num_triangles)
        # Associate each small cluster to a color:
        colors = proc.generate_N_unique_colors(len(small_cluster_num_triangles))
        num_color_dict = dict(zip(small_cluster_num_triangles, colors))
        color_num_dict = dict(zip(colors, small_cluster_num_triangles))
        # Fetch vertice indices for each small cluster:
        num_tr_vs_vertices = {}
        for num_tr in small_cluster_num_triangles:
            num_tr_mask = cluster_n_triangles[triangle_cluster_ids] == num_tr
            num_tr_vs_vertices[num_tr] = np.unique(np.ravel(v_mesh_triangles[num_tr_mask]))
        # Update with new colors for each small cluster:
        low_density_mesh = copy.deepcopy(v_mesh)
        small_clusters = []
        for num_tr in small_cluster_num_triangles:
            cluster_color = num_color_dict[num_tr]
            part_low_density_mesh = low_density_mesh.select_by_index(num_tr_vs_vertices[num_tr])
            part_low_density_mesh.paint_uniform_color(list(cluster_color))
            small_clusters.append(part_low_density_mesh)
        # Aggregate points to construct small cluster mesh:
        small_cluster_mesh = small_clusters[0]
        for i in range(1,len(small_clusters)):
            small_cluster_mesh += small_clusters[i]
        # Add high-density mesh as well and paint it uniformily:
        high_density_mesh = copy.deepcopy(v_mesh)
        high_density_mesh.remove_triangles_by_mask(small_cluster_triangles)
        high_density_mesh.remove_unreferenced_vertices()
        high_density_mesh.paint_uniform_color([0, 0.5, 0]) # Green
        full_mesh  = high_density_mesh + small_cluster_mesh
        # Turn mesh into point cloud:
        full_pcd = o3d.geometry.PointCloud()
        full_pcd.points = full_mesh.vertices
        full_pcd.colors = full_mesh.vertex_colors
        # Score distortion by number of triangles in the small cluster it belongs to:
        tree = o3d.geometry.KDTreeFlann(full_pcd)
        np_down = np.array(downsized.points)
        values = np.empty(point_scores.shape[0])
        for i in range(len(np_down)):
            # For each voxel, get closest point in full_pcd:
            [k, idx, _] = tree.search_knn_vector_3d(np_down[i], 1)
            # Get color of the point in full_pcd:
            point_color = list(np.asarray(full_pcd.colors)[idx[0]])
            # Get actual point cloud indices:
            actual_inds = downsized_links[i]
            if point_color == [0, 0.5, 0]:
                for ind in actual_inds:
                    values[ind] = max_small_cluster_num_triangles + 1 # Set a bigger number than the biggest small cluster, so that they get a 0 score.
            else:
                for ind in actual_inds:
                    values[ind] = color_num_dict[tuple(point_color)]
        # Update scores (less triangles --> higher score):
        update_scores(values)
        if snapshotting:
            inout.snapshot(full_mesh, name = "modelling_clusters")
        if visualization:
            inout.visualize(full_mesh)
    else:
        # One big cluster --> no score update applied.
        if snapshotting:
            inout.snapshot(v_mesh, name = "modelling")
        if visualization:
            inout.visualize(v_mesh)

def reattribute_values_and_score(values, inverse = True):
    '''
        Uses links to reattribute values to the points belonging to voxel if downsized model is used, and computes scores.
    Args:
        values (int list) : list of values beloging to voxels / all points
    Returns:
        actual_values (int list) : list of values beloging to all points
    '''
    global large
    global original
    global downsized_links

    if large:
        # Reattribute values to full point cloud:
        actual_values = np.empty(len(original.points)).astype(np.int32)
        for k,v in downsized_links.items():
            actual_values[v] = values[k]
        actual_values = list(actual_values)
    else:
        actual_values = list(values)
    # Update scores:
    update_scores(actual_values, inverse = inverse)
    return actual_values


def update_scores(values, inverse = True, downsized = False):
    '''
        Updates scores.
    Args:
        values (int list) : list of values beloging to all points
    '''
    global point_scores
    global downsized_links

    # Update scores:
    mapped_probs = proc.map_to_probabilities(values, inverse = inverse)
    if downsized:
        for k,v in downsized_links.items():
            for ind in v:
                point_scores[ind] += mapped_probs[k]
    else:
        for i in range(len(point_scores)):
            point_scores[i] += mapped_probs[i]



def nrpcqa_radius_nb(pcd, snapshotting = False, visualization = False, K = 5, P = 10, shades = 10):
    '''
        Uses radius-based neighbourhood statistics to calculate metrics related to completeness/validity.
    Args:
        pcd (type: PointCloud object) : point cloud to analyze
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
        K (int) : radius is estimated by the mean distance to p neighbours from a random set of k points, k defaults to 5
        P (int) : radius is estimated by the mean distance to p neighbours from a random small set of k points, p defaults to 10
        shades (int) : equal to bar number in plot, if plotting is allowed, defaults to 10

    Returns:
        (list) : metric list of tuples with format (METRIC_NAME, METRIC_SCORE)
    '''
    global original
    global mean_nb

    # Find mean distance to p neighbours by a small set of k points from KDTree:
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    ks = random.sample(range(0, len(pcd.points)-1), k = K) # Pick K random points
    dists = []
    for k in ks:
        # Find their n closest neighbours:
        nb, inds, _ = kdtree.search_knn_vector_3d(pcd.points[k], P)
        nb_points = [pcd.points[i] for i in inds[1:]]
        # Get all unique pairs of points:
        combs = combinations(nb_points, 2)
        # Calculate the L2 distances within pairs:
        dists += [proc.get_l2_distance(c[0], c[1]) for c in combs]
    mean_dist =  np.mean(dists)
    # Set radius to the double of the mean over all found distances:
    radius = 2 * mean_dist
    nb_within_radius = [kdtree.search_radius_vector_3d(point, radius) for point in pcd.points]
    # Color according to number of neighbours:
    values = [len(nb_list[1]) for nb_list in nb_within_radius]
    values = reattribute_values_and_score(values)
    mean_nb = int(np.mean(values)) # Save mean of neighbourhood size for LOF assessment
    if snapshotting:
        colored_pcd, color_range = proc.color_cloud_greyscale(original, values, shades = shades)
        inout.snapshot(colored_pcd, name = "radius_neighbourhood")
        inout.plot_values_by_color(values, color_range,  x_label="Number of neighbours", y_label="Frequency", save = True, name="plot_radius_neighbourhood")
    if visualization:
        colored_pcd, color_range = proc.color_cloud_greyscale(original, values, shades = shades)
        inout.visualize(colored_pcd)
        inout.plot_values_by_color(values, color_range,  x_label="Number of neighbours", y_label="Frequency", show = True, name="plot_radius_neighbourhood")


def nrpcqa_lof(pcd, snapshotting = False, visualization = False, shades = 15):
    '''
        Uses LOF outlier detection to calculate metrics related to accuracy/validity. A radius search is used to determine mean number of neighbours for the LOF algorithm, which is factored for large clouds.
    Args:
        pcd (type: PointCloud object) : point cloud to analyze
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
        shades (int) : equal to bar number in plot, if plotting is allowed, defaults to 15
    '''
    global original
    global mean_nb

    # Local Outlier Factor - get all points scored:
    clf = LocalOutlierFactor(n_neighbors = mean_nb, n_jobs = -1, contamination = "auto") # OBS! Change contamination to allow user intervention
    numpy_pcd = np.array(pcd.points)
    preds = clf.fit_predict(numpy_pcd)
    values = -clf.negative_outlier_factor_ # LOF Scores
    values = reattribute_values_and_score(values, inverse = False) # (high LOF --> higher score)
    if visualization:
        colored_pcd, color_range = proc.color_cloud_rainbow(original, values, shades = shades)
        inout.visualize(colored_pcd)
        inout.plot_values_by_color(values, color_range,  x_label="LOF Score", y_label="Frequency", save = False, name="plot_lof")


def nrpcqa_statistical(pcd, snapshotting = False, visualization = False, std_ratio = 2.0):
    '''
        Uses statistical outlier detection to calculate metrics related to accuracy/validity. A radius search is used to determine mean number of neighbours for the detection algorithm.
    Args:
        pcd (type: PointCloud object) : point cloud to analyze
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
        std_ratio (float) : standard deviation ratio used in statistical removal (lower is more aggressive), defaults to 2.0
    '''
    global large
    global downsized_links
    global point_scores
    global mean_nb
    global original

    cl, ind = proc.remove_outliers(pcd, num_nb = mean_nb, std_ratio = std_ratio)
    if large:
        # Remove all point belonging to outlier voxels from original cloud:
        actual_inds_full = [downsized_links[i] for i in ind]
        actual_inds = [j for sub in actual_inds_full for j in sub] # Flatten
        clean_pcd = original.select_by_index(actual_inds)
        noise_pcd = original.select_by_index(actual_inds, invert=True)
    else:
        clean_pcd = pcd.select_by_index(ind)
        noise_pcd = pcd.select_by_index(ind, invert=True)
    # Update scores (+1 for noise-detected points):
    true_clean_ind = actual_inds if large else ind
    index_array = np.ones(point_scores.shape[0], dtype=bool)
    index_array[true_clean_ind] = False
    point_scores[index_array] += 1
    if snapshotting:
        inout.snapshot(noise_pcd, name = "stat_noise")
    if visualization:
        inout.visualize_differences(clean_pcd, noise_pcd)
  


def quality_calculation(metrics, weights, output = "metrics"):
    '''
        Get accuracy (+ validity, completeness). Saves results to JSON. 
    Args:
        metrics (list) : list of metrics for ACCURACY/COMPLETENESS/VALIDITY respectively, or ACCURACY (if NRPCQA)
        weights (list) : list of weights for ACCURACY/COMPLETENESS/VALIDITY respectively, or None (if NRPCQA)
        output (str) : output file name
    '''
    global automatic
    json_dict = {}

    # No-reference quality assessment:
    if weights is None:
        ACCURACY = metrics[0]
        json_dict["Accuracy"] = ACCURACY
    else:
        # Reference-based assessment:
        ACCURACY = metrics[0]
        VALIDITY = metrics[1]
        COMPLETENESS = metrics[2]
        # Fill json:
        json_dict["Accuracy"] = ACCURACY
        json_dict["Validity"] = VALIDITY
        json_dict["Completeness"] = COMPLETENESS
        if not automatic:
            # Weight results and fetch overall quality metric:
            WEIGHTED_QUALITY = weight_quality(metrics, weights)
            json_dict["(Weighted) Quality"] = WEIGHTED_QUALITY 
    # Save to JSON:
    inout.write_to_json(json_dict, output)


def weight_quality(metrics, weights):
    '''
        Calculate weighted quality based on metric list and weights for reference-based assessment.
    Args:
        metrics (list) : metric scores in following order ACCURACY/VALIDITY/(optional)COMPLETENESS
        weights (list) : list of weights for  ACCURACY/VALIDITY/(optional)COMPLETENESS respectively, defaults to equally weighted
    Returns:
        quality (float) : weighted quality
    '''
    # Get the boundary values for each quality score:
    sum_weights = sum(weights)
    # If RPCQA block has been processed, consider completeness as part of quality metric:
    weighted_accuracy = metrics[0] * weights[0]
    weighted_validity =  metrics[1] * weights[1]
    weighted_completeness =  metrics[2] * weights[2]
    weighted_sum = weighted_accuracy + weighted_validity + weighted_completeness
    # Calculate weighted quality:
    quality = weighted_sum/sum_weights
    return quality


@lru_cache
def process_without_reference(input_file: str):
    '''
        Point Cloud Data Validation - without reference
    Args:
        input_file (str) : path of point cloud to analyse
    Returns:
        metric (list) : list containing accuracy of point cloud
    '''
    # Global variables to update:
    global snapshotting
    global visualization
    global original
    global large
    global point_scores

    # Start processing time:
    start_time = time.time()

    print("Action in progress: reading point cloud...") # Order matters
    # Read point cloud from path:
    original = inout.read_cloud(input_file)
    if visualization:
        inout.visualize(original)
    NUM_RAW_POINTS = len(original.points)

    # Keep boolean check for large scan (over 0.5 mil):
    if NUM_RAW_POINTS > 500_000:
        large = True

    print("Action in progress: instantiate score tracking...")
    # Instantiate point-wise distortion scores:
    point_scores = np.zeros(NUM_RAW_POINTS)

    print("Action in progress: downsize and voxelization assessment...")
    nrpcqa_voxelization(original, snapshotting = snapshotting, visualization = visualization) # OK

    print("Action in progress: density assessment based on voxel modelling...")
    nrpcqa_modelling(original, snapshotting = snapshotting, visualization = visualization) # OK

    # To make processing more time-efficient, use downsized point cloud for large clouds:
    if large:
        pcd = copy.deepcopy(downsized)
    else:
        pcd = copy.deepcopy(original)
   
    print("Action in progress: density assessment followed based on radius neighbourhood,...") 
    nrpcqa_radius_nb(pcd, snapshotting = snapshotting, visualization = visualization) # OK

    print("Action in progress: density-based assessment based on Local Outlier Factor...") 
    nrpcqa_lof(pcd, snapshotting = snapshotting, visualization = visualization) # OK

    print("Action in progress: statistical removal of noise based on distance to neighbours...") 
    nrpcqa_statistical(pcd, snapshotting = snapshotting, visualization = visualization) # OK


    # Get accuracy:
    scaled_probs = proc.minmax_scale(point_scores)
    # Use IQR rule as threshold:
    Q1 = np.percentile(scaled_probs, 25) # First quartile
    Q3 = np.percentile(scaled_probs, 75) # Third quartile
    IQR = Q3 - Q1
    noise_threshold = Q3 + 1.5 * IQR
    noise_ind = np.where(scaled_probs >= noise_threshold)[0]
    noise = proc.get_cloud_by_index(original, noise_ind)
    NUM_NOISE = len(noise.points)
    ACCURACY = 1.0 - (NUM_NOISE/NUM_RAW_POINTS)

    # End processing time
    end_time = time.time()
    nrpcqa_time = datetime.timedelta(seconds=end_time - start_time)
    print("Processing time (NRPCQA):" +  str(nrpcqa_time))

    return [ACCURACY] 


def stitch(dir_path:str, *args):
    '''
        Aggregates a directory of .ply files into single PointCloud object.
        Outputs result to same directory.
        Assumes same coordinates/scale between point clouds.
    Args:
        dir_path (str) : path of directory containing .ply files
        *args (str) : name of output file - optional
    '''
    # TODO: Multiway registration
    # Get all paths of .ply files from given directory:
    ply_files = []
    os.chdir(dir_path)
    for file in glob.glob("*.ply"):
        ply_files.append(file)
    # Check that .ply files are found in directory:
    num_plys = len(ply_files)
    if num_plys == 0:
        raise FileNotFoundError("No .ply files found in directory!")
    # Stitch :
    stitched_pcd = proc.stitch_clouds(ply_files)
    # If name specified, place in same directory with new name:
    if len(args) != 0:
        inout.write_cloud(dir_path + "/" + args[0]+ ".ply", stitched_pcd)
    # Else, place in same directory with default name:
    else:
        name = "/3ddava_Stitched_"+str(num_plys)+"_Clouds.ply"
        inout.write_cloud(dir_path + name, stitched_pcd)


def main(argv = None):
    ''' Parsing command line arguments
    Args:
        argv (list): list of arguments
    '''
    global visualization
    global snapshotting
    global automatic
    global saving

    # Top-level parser:
    parser = argparse.ArgumentParser(add_help=False)

    # No arguments:
    if argv is None:
        argv = sys.argv[1:]
        if not len(argv):
            parser.error('Insufficient arguments provided.')

    # Subparsers for each functionality (processing and stitching):
    subparsers = parser.add_subparsers(help='Help', dest='which')
    # Processing subparser:
    processing_input = subparsers.add_parser('processing', help='Processing Point Cloud Help')
    # Obligatory argument:
    processing_input.add_argument("cloud_file", help = "Input point cloud file to be analyzed (format: .ply).")
    # Optional arguments:
    processing_input.add_argument("-r", "--reference",  help = "The reference file (format: .stl) - optional.")
    processing_input.add_argument("-o", "--output",  help = "Filename of output file. No file extension needed.")
    # Add visualization / snapshotting as argument:
    processing_input.add_argument("-vis", "--visualize",  help = "Allow visualization of process steps.",  action="store_true")
    processing_input.add_argument("-snap", "--snapshot",  help = "Allow snapshotting of process steps." ,  action="store_true")
    # Add saving denoised point cloud as argument
    processing_input.add_argument("-save", "--save",  help = "Save denoised point cloud as (.pcd)." ,  action="store_true")
    # Add automatic processing as argument:
    processing_input.add_argument("-auto", "--automatic",  help = "Automatic quality assessment, no intervention." ,  action="store_true")
    # Stitching subparser:
    stitching = subparsers.add_parser('stitching', help='Stitching Files Help')
    # Obligatory argument:
    stitching.add_argument("directory_path", type=proc.is_dir, help = "Absolute path of directory containing .ply clouds.")
    # Optional arguments:
    stitching.add_argument("-o", "--output",  help = "Filename of output file. No file extension needed.")

    args = parser.parse_args(argv)
    # Parse stitching:
    if args.which == "stitching":
        try:
            print("Stitching .ply files from directory ... ")
            directory_path = args.directory_path
            if args.output:
                output_file = args.output
                stitch(directory_path, output_file)
            else:
                stitch(directory_path)
        except Exception as e:
            print(str(e) + " Action dropped: stitching clouds.  Use -h, --h or --help for more information.")
    # Parse processing:
    else:
        try:
            # Get arguments:
            input_file = args.cloud_file
            visualization = args.visualize
            snapshotting = args.snapshot
            saving = args.save
            automatic = args.automatic
             # Validate input arg format:
            if not proc.is_path(input_file):
                raise FileNotFoundError("Specified input cloud file was not found on given path.")
            if not input_file.endswith(".ply"):
                if not input_file.endswith(".pcd"):
                    raise TypeError("Input file must be in .ply or .pcd format.")
            # If reference arg given, process with reference:
            if args.reference:
                print("Processing with reference ... ")
                reference_file = args.reference
                if not proc.is_path(reference_file):
                    raise FileNotFoundError("Specified reference/CAD file was not found on given path.")
                if not reference_file.endswith(".stl"):
                    raise TypeError("Reference/CAD file must be in .stl format.")
                # Ask for weight importance if not automatic, else equal weighting is applied:
                if automatic:
                    # All metrics have same importance:
                    weight_accuracy = 1.0 
                    weight_validity = 1.0 
                    weight_completeness = 1.0
                else:
                    while True:
                        try:
                            weight_accuracy = input("Accuracy weight/importance OR unknown (ENTER): ")
                            weight_validity = input("Validity weight/importance OR unknown (ENTER): ")
                            weight_completeness = input("Completeness weight/importance OR unknown (ENTER): ")
                            # Check that it is in right format:
                            if weight_accuracy:
                                weight_accuracy = float(weight_accuracy)
                            else:
                                weight_accuracy = 1.0
                            if weight_validity:
                                weight_validity = float(weight_validity)
                            else:
                                weight_validity = 1.0
                            if weight_completeness:
                                weight_completeness = float(weight_completeness)
                            else:
                                weight_completeness = 1.0
                            break
                        except ValueError:
                            print("Invalid input. Please enter a valid float or integer.")
                weights = [weight_accuracy,  weight_validity,  weight_completeness]
                metrics = process_with_reference(input_file, reference_file)
                # If output filename is given, use as output file name:
                if args.output:
                    output_file = args.output
                    quality_calculation(metrics, weights, output = output_file)
                else:
                    quality_calculation(metrics, weights)
            else:
                print("Processing without reference ... ")
                metrics = process_without_reference(input_file)
                # If output filename is given, use as output file name:
                if args.output:
                    output_file = args.output
                    quality_calculation(metrics, None, output = output_file)
                else:
                    quality_calculation(metrics, None)
            # TODO Save the clean model if saving arg is given:
        except Exception as e:
            print(str(e) + " Action dropped: processing clouds.  Use -h, --h or --help for more information.")

if __name__ == '__main__':
    main()
