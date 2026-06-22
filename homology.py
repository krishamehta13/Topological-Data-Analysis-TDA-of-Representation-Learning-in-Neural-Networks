import numpy as np
import ripser
import gudhi
from scipy.spatial.distance import pdist, squareform

def compute_persistence(embeddings, max_dimension=1):
    """Computes persistence diagrams of the given embeddings using Ripser.
    
    Args:
        embeddings (np.ndarray): Data points of shape (n_samples, n_features).
        max_dimension (int): Maximum homology dimension to compute.
        
    Returns:
        list of np.ndarray: List of persistence diagrams for dimensions 0 to max_dimension.
    """
    # Run Ripser
    result = ripser.ripser(embeddings, maxdim=max_dimension)
    return result['dgms']

def compute_persistence_entropy(diagram):
    """Computes the persistence entropy of a persistence diagram.
    
    Persistence entropy is defined as the Shannon entropy of the normalized lifetimes of the points.
    
    Args:
        diagram (np.ndarray): Persistence diagram of shape (n_points, 2).
        
    Returns:
        float: Persistence entropy value.
    """
    if len(diagram) == 0:
        return 0.0
        
    # Replace infinity deaths with a large finite value (max finite death in the diagram, or a default)
    diag_copy = diagram.copy()
    finite_deaths = diag_copy[np.isfinite(diag_copy[:, 1]), 1]
    
    if len(finite_deaths) == 0:
        # If all deaths are infinite, assign arbitrary max value
        max_val = 1.0
    else:
        max_val = np.max(finite_deaths)
        
    diag_copy[~np.isfinite(diag_copy[:, 1]), 1] = max_val
    
    lifetimes = diag_copy[:, 1] - diag_copy[:, 0]
    # Filter out zero or negative lifetimes
    lifetimes = lifetimes[lifetimes > 1e-9]
    
    if len(lifetimes) == 0:
        return 0.0
        
    L_sum = np.sum(lifetimes)
    p = lifetimes / L_sum
    entropy = -np.sum(p * np.log(p))
    return entropy

def compute_bottleneck_distance(diag1, diag2):
    """Computes the Bottleneck distance between two persistence diagrams using Gudhi.
    
    Args:
        diag1 (np.ndarray): First diagram of shape (n_points_1, 2).
        diag2 (np.ndarray): Second diagram of shape (n_points_2, 2).
        
    Returns:
        float: Bottleneck distance.
    """
    # Gudhi expects diagrams as lists of lists or numpy arrays of birth/death.
    # If diagrams contain inf values, gudhi.bottleneck_distance might throw an error or need handling.
    # Typically, we filter out points with infinity death.
    d1 = diag1[np.isfinite(diag1[:, 1])] if len(diag1) > 0 else np.empty((0, 2))
    d2 = diag2[np.isfinite(diag2[:, 1])] if len(diag2) > 0 else np.empty((0, 2))
    
    try:
        return gudhi.bottleneck_distance(d1, d2)
    except Exception as e:
        # Fallback in case of errors
        return 0.0

def compute_wasserstein_distance(diag1, diag2, order=1):
    """Computes the Wasserstein distance between two persistence diagrams using Gudhi.
    
    Args:
        diag1 (np.ndarray): First diagram of shape (n_points_1, 2).
        diag2 (np.ndarray): Second diagram of shape (n_points_2, 2).
        order (int): Wasserstein order.
        
    Returns:
        float: Wasserstein distance.
    """
    d1 = diag1[np.isfinite(diag1[:, 1])] if len(diag1) > 0 else np.empty((0, 2))
    d2 = diag2[np.isfinite(diag2[:, 1])] if len(diag2) > 0 else np.empty((0, 2))
    
    try:
        return gudhi.wasserstein_distance(d1, d2, order=order)
    except Exception as e:
        return 0.0

def get_significant_features(diagram, threshold=0.1):
    """Returns persistent features that have a lifetime (death - birth) greater than a threshold.
    
    Args:
        diagram (np.ndarray): Persistence diagram.
        threshold (float): Lifetime threshold.
        
    Returns:
        np.ndarray: Filtered diagram points.
    """
    if len(diagram) == 0:
        return np.empty((0, 2))
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return diagram[lifetimes > threshold]
