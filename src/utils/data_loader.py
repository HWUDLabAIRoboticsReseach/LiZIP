import numpy as np
import os

def load_nuscenes_lidar(path):
    """
    Loads a NuScenes binary LiDAR file.
    NuScenes format: [x, y, z, intensity, ring_index]
    Returns: Numpy array of shape (N, 4) -> drops ring_index
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    scan = np.fromfile(path, dtype=np.float32)
    
    points = scan.reshape((-1, 5))
    
    return points[:, :4]

def load_kitti_data(path):
    """
    Loads a KITTI text-based LiDAR file.
    Format: x y z intensity (space separated)
    Returns: Numpy array of shape (N, 4)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Use np.loadtxt for robust text reading
    try:
        points = np.loadtxt(path, dtype=np.float32)
        
        # Ensure it has at least 3 columns (XYZ)
        if points.ndim == 1:
            points = points.reshape(1, -1)
            
        if points.shape[1] < 3:
            raise ValueError(f"KITTI data must have at least 3 columns (XYZ), found {points.shape[1]}")
            
        return points
    except Exception as e:
        raise ValueError(f"Failed to load KITTI data from {path}: {e}")

def load_ply_data(path):
    """
    Loads a PLY point cloud file.
    Returns: Numpy array of shape (N, 3) or (N, 4)
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is required to load PLY files. Please install it with 'pip install open3d'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError("PLY file is empty or invalid.")
        
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Remove NaNs and Infs
        points = points[~np.isnan(points).any(axis=1)]
        points = points[~np.isinf(points).any(axis=1)]
        
        return points
    except Exception as e:
        raise ValueError(f"Failed to load PLY data from {path}: {e}")

def load_point_cloud(path):
    """
    Generic point cloud loader. Dispatches based on file extension.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.bin':
        return load_nuscenes_lidar(path)
    elif ext == '.txt':
        return load_kitti_data(path)
    elif ext == '.ply':
        return load_ply_data(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def save_kitti_data(points, path):
    """
    Saves points to a KITTI-style text file.
    Args:
        points: (N, 3) or (N, 4) numpy array
        path: Output file path
    """
    fmt = '%.6f'
    np.savetxt(path, points, fmt=fmt, delimiter=' ')

def visualize_point_cloud(points):
    """
    Visualizes (N, 3+) numpy point cloud using Open3D.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("open3d is required for visualization. Please install it with 'pip install open3d'.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    z_vals = points[:, 2]
    colors = np.zeros((points.shape[0], 3))

    min_z, max_z = np.percentile(z_vals, 1), np.percentile(z_vals, 99)
    norm_z = np.clip((z_vals - min_z) / (max_z - min_z + 1e-6), 0, 1)
    
    colors[:, 0] = norm_z
    colors[:, 2] = 1 - norm_z
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="LiDAR Viewer")