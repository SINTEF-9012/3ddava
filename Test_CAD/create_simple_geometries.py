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


# Create a sphere:
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
sphere.compute_vertex_normals()

# Create a cube:
cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
cube.compute_vertex_normals()

# Create cylinder:
radius = 1.0  # Radius of the cylinder
height = 4.0  # Height of the cylinder
cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution = 100)
cylinder.compute_vertex_normals()

# Save the meshes as STL:
o3d.io.write_triangle_mesh("cylinder.stl", cylinder)
o3d.io.write_triangle_mesh("cube.stl", cube)
o3d.io.write_triangle_mesh("sphere.stl", sphere)
