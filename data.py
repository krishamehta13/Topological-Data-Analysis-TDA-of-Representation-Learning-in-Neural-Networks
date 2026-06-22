import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_circles(n_samples=600, noise=0.05, factor=0.5):
    """Generates two concentric circles in 2D."""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    return X, y

def generate_torus(n_samples=600, R=2.0, r=0.5, noise=0.05):
    """Generates points on a 3D torus.
    Classes are defined by partition along the angular coordinate.
    """
    np.random.seed(42)
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    phi = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # Parametric equations of torus
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    X = np.stack([x, y, z], axis=1)
    # Add noise
    X += np.random.normal(0, noise, X.shape)
    
    # Class label based on theta (split into two classes for classification task)
    y = (theta > np.pi).astype(int)
    return X, y

def generate_spirals(n_samples=600, noise=0.1):
    """Generates two interlocked spirals in 2D."""
    np.random.seed(42)
    n = n_samples // 2
    
    # Spiral 1
    theta1 = np.sqrt(np.random.rand(n)) * 2 * np.pi * 1.5
    r1 = 2 * theta1 + 1
    x1 = -np.cos(theta1) * r1 + np.random.randn(n) * noise
    y1 = np.sin(theta1) * r1 + np.random.randn(n) * noise
    
    # Spiral 2
    theta2 = np.sqrt(np.random.rand(n)) * 2 * np.pi * 1.5
    r2 = 2 * theta2 + 1
    x2 = np.cos(theta2) * r2 + np.random.randn(n) * noise
    y2 = -np.sin(theta2) * r2 + np.random.randn(n) * noise
    
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    # Normalize data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y

def get_dataloader(X, y, batch_size=32, shuffle=True):
    dataset = SimpleDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
