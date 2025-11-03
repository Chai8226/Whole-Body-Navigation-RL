#!/usr/bin/env python3
"""
Simple 3D visualization of shape_scan data with OBJ mesh overlay
Usage: python visualize_shape_scan.py <path_to_npz_file>
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import os

def load_shape_scan(npz_file):
    """Load shape_scan data from npz file"""
    data = np.load(npz_file, allow_pickle=True)
    return data

def load_obj_file(obj_file):
    """Load OBJ file and return vertices and faces"""
    vertices = []
    faces = []
    
    if not os.path.exists(obj_file):
        print(f"Warning: OBJ file not found: {obj_file}")
        return None, None
    
    with open(obj_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    
    if not vertices or not faces:
        return None, None
    
    return np.array(vertices), np.array(faces)

def visualize_3d(shape_scan, horizontal_angles, vertical_angles, vertices=None, faces=None):
    """Visualize shape_scan as 3D point cloud with optional OBJ mesh overlay"""
    H, V = shape_scan.shape[1], shape_scan.shape[2]
    
    # Convert shape_scan to 3D coordinates
    points = []
    colors = []
    
    for h_idx in range(H):
        for v_idx in range(V):
            r = shape_scan[0, h_idx, v_idx]
            if r > 0:  # Only plot valid points
                h_angle = horizontal_angles[h_idx]
                v_angle = vertical_angles[v_idx]
                
                # Spherical to Cartesian
                x = r * np.cos(v_angle) * np.cos(h_angle)
                y = r * np.cos(v_angle) * np.sin(h_angle)
                z = r * np.sin(v_angle)
                
                points.append([x, y, z])
                colors.append(r)
    
    points = np.array(points)
    colors = np.array(colors)
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    
    # 3D scatter plot with mesh
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot OBJ mesh first (as wireframe/surface)
    if vertices is not None and faces is not None:
        # Create mesh triangles
        mesh_triangles = []
        for face in faces:
            if len(face) >= 3:
                # For triangles
                triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
                mesh_triangles.append(triangle)
                # If quad or polygon, triangulate simply
                for i in range(3, len(face)):
                    triangle = [vertices[face[0]], vertices[face[i-1]], vertices[face[i]]]
                    mesh_triangles.append(triangle)
        
        mesh_collection = Poly3DCollection(mesh_triangles, alpha=0.3, 
                                          facecolor='cyan', edgecolor='blue', linewidths=0.5)
        ax1.add_collection3d(mesh_collection)
        
        # Plot vertices as small points
        ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                   c='blue', s=5, alpha=0.5, label='OBJ vertices')
    
    # Plot shape_scan points on top
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=colors, cmap='hot', s=3, alpha=0.8, label='Shape scan')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Shape Scan + OBJ Mesh')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Distance (m)', shrink=0.5)
    
    # Set equal aspect ratio
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                         points[:, 1].max()-points[:, 1].min(),
                         points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Top view (XY plane)
    ax2 = fig.add_subplot(132)
    if vertices is not None:
        ax2.scatter(vertices[:, 0], vertices[:, 1], c='blue', s=5, alpha=0.3, label='OBJ')
    ax2.scatter(points[:, 0], points[:, 1], c=colors, cmap='hot', s=3, alpha=0.8, label='Scan')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (XY)')
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Side view (XZ plane)
    ax3 = fig.add_subplot(133)
    if vertices is not None:
        ax3.scatter(vertices[:, 0], vertices[:, 2], c='blue', s=5, alpha=0.3, label='OBJ')
    ax3.scatter(points[:, 0], points[:, 2], c=colors, cmap='hot', s=3, alpha=0.8, label='Scan')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (XZ)')
    ax3.axis('equal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        # Try to find npz file automatically
        shape_scan_dir = os.path.join(os.path.dirname(__file__), "..", "shape_scan_data")
        if os.path.exists(shape_scan_dir):
            npz_files = [f for f in os.listdir(shape_scan_dir) if f.endswith('.npz')]
            if npz_files:
                npz_file = os.path.join(shape_scan_dir, npz_files[0])
                print(f"Using: {npz_file}")
            else:
                print("No .npz files found in shape_scan_data/")
                return
        else:
            print("Usage: python visualize_shape_scan.py <path_to_npz_file>")
            return
    else:
        npz_file = sys.argv[1]
    
    if not os.path.exists(npz_file):
        print(f"File not found: {npz_file}")
        return
    
    # Load data
    print(f"Loading {npz_file}...")
    data = load_shape_scan(npz_file)
    
    shape_scan = data['shape_scan']
    horizontal_angles = data['horizontal_angles']
    vertical_angles = data['vertical_angles']
    shape_name = str(data['shape_name'])
    
    print(f"Shape name: {shape_name}")
    print(f"Shape scan shape: {shape_scan.shape}")
    print(f"Min: {shape_scan.min():.4f}, Max: {shape_scan.max():.4f}, Mean: {shape_scan.mean():.4f}")
    print(f"Non-zero points: {np.count_nonzero(shape_scan)}/{shape_scan.size}")
    
    # Try to load corresponding OBJ file
    obj_folder = os.path.join(os.path.dirname(__file__), "..", "obj")
    obj_file = os.path.join(obj_folder, f"{shape_name}.obj")
    
    print(f"\nLooking for OBJ file: {obj_file}")
    vertices, faces = load_obj_file(obj_file)
    
    if vertices is not None:
        print(f"Loaded OBJ: {len(vertices)} vertices, {len(faces)} faces")
        print(f"OBJ bounds: X[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}], "
              f"Y[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}], "
              f"Z[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")
    else:
        print("No OBJ file loaded, showing shape_scan only")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_3d(shape_scan, horizontal_angles, vertical_angles, vertices, faces)

if __name__ == "__main__":
    main()
