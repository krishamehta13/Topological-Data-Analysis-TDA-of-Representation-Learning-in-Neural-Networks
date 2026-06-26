import numpy as np
import ripser
import gudhi
from scipy.spatial.distance import pdist, squareform

def compute_persistence(embeddings, max_dimension=1):
    """Evaluates the structural properties of activation spaces by tracking open neighborhoods.
    
    This function treats the embeddings as a metric space (X, d) and grows open balls
    B(x, epsilon) of radius epsilon around each point. It computes the birth and death of
    connected components (H0) and loops (H1) as the open neighborhoods overlap.
    
    Args:
        embeddings (np.ndarray): Layer activations of shape (n_samples, n_features).
        max_dimension (int): Maximum complexity dimension to compute (0 for components, 1 for loops).
        
    Returns:
        list of np.ndarray: Birth/death radius scales for structural features.
    """
    # Run Ripser to compute persistent features
    result = ripser.ripser(embeddings, maxdim=max_dimension)
    return result['dgms']

def compute_persistence_entropy(diagram):
    """Computes the topological complexity of the open ball neighborhood structure.
    
    This calculates the Shannon entropy of the normalized lifetimes (death radius - birth radius)
    of the topological features. A lower H0 entropy signifies highly compact and separated 
    connected components in the metric space.
    
    Args:
        diagram (np.ndarray): Feature scale intervals of shape (n_points, 2).
        
    Returns:
        float: Normalized structural complexity score.
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
    """Computes the representation deviation (Bottleneck distance) between two structures.
    
    This measures the topological shift in the metric space configurations (e.g., between
    untrained and trained states).
    
    Args:
        diag1 (np.ndarray): First diagram of shape (n_points_1, 2).
        diag2 (np.ndarray): Second diagram of shape (n_points_2, 2).
        
    Returns:
        float: Representation shift score.
    """
    # Gudhi expects diagrams as lists of lists or numpy arrays of birth/death.
    d1 = diag1[np.isfinite(diag1[:, 1])] if len(diag1) > 0 else np.empty((0, 2))
    d2 = diag2[np.isfinite(diag2[:, 1])] if len(diag2) > 0 else np.empty((0, 2))
    
    try:
        return gudhi.bottleneck_distance(d1, d2)
    except Exception as e:
        return 0.0

def compute_wasserstein_distance(diag1, diag2, order=1):
    """Computes the Wasserstein distance (representation alignment deviation) between two structures.
    
    Args:
        diag1 (np.ndarray): First diagram.
        diag2 (np.ndarray): Second diagram.
        order (int): Distance order.
        
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
    """Returns stable structural features that persist beyond a radius threshold.
    
    Args:
        diagram (np.ndarray): Feature scale intervals.
        threshold (float): Minimum lifespan (death - birth radius).
        
    Returns:
        np.ndarray: Filtered diagram points.
    """
    if len(diagram) == 0:
        return np.empty((0, 2))
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return diagram[lifetimes > threshold]
