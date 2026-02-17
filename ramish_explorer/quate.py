"""
Quaternion operations for .ramish file queries.

Pure numpy implementations — no torch dependency.
Only the operations needed for inference (read path), not training.
"""

import numpy as np


def hamilton_product_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product for numpy arrays.
    Args:
        q1, q2: Arrays of shape (..., 4) representing quaternions
    Returns:
        Product quaternion of same shape
    """
    a1, b1, c1, d1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    a2, b2, c2, d2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2
    ], axis=-1)


def quaternion_conjugate_np(q: np.ndarray) -> np.ndarray:
    """
    Quaternion conjugate for numpy arrays.
    For a unit quaternion, the conjugate is the inverse rotation.
    q = a + bi + cj + dk  ->  q* = a - bi - cj - dk
    """
    result = q.copy()
    result[..., 1] = -result[..., 1]
    result[..., 2] = -result[..., 2]
    result[..., 3] = -result[..., 3]
    return result
