import numpy as np

def generate_context_target_pairs(sorted_points, context_size=5):
    """
    Generates (Context, Target) pairs from a spatially sorted point cloud.
    
    Args:
        sorted_points (np.ndarray): (N, 3) or (N, 4) numpy array of sorted points.
        context_size (int): 'k', the number of previous points to use as context.
        
    Returns:
        tuple:
            - **contexts** (np.ndarray): (M, context_size * 3) Input array for the model.
            - **targets** (np.ndarray): (M, 3) Output array (labels) for the model.
    """
    num_points = len(sorted_points)
    
    num_samples = num_points - context_size
    
    slices = []
    points_xyz = sorted_points[:, :3]
    
    for i in range(context_size):
        slices.append(points_xyz[i : i + num_samples])
    
    contexts = np.hstack(slices)
    
    targets = points_xyz[context_size : context_size + num_samples]
        
    return contexts, targets