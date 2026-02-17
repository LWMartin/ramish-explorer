"""
Embedding quantization for .ramish files.

Validated results (Feb 2, 2026 — Chinook, MovieLens, IMDB):
- fp16: Effectively lossless (score correlation 0.999999, <0.2% rank swaps)
- int8: Viable with minor loss (score correlation 0.9993+, <2.2% rank swaps)
- DO NOT use quaternion renormalization (entity embeddings are not on S^3)

Entity embeddings are NOT constrained to unit quaternions during training.
Only relation embeddings are normalized. Renormalization destroys learned
distance structure (kNN drops to 0.1%, 28% rank swaps on MovieLens).
"""

import numpy as np
from typing import Dict, Any


SUPPORTED_DTYPES = ("fp32", "fp16", "int8")

DTYPE_TO_BITS = {"fp32": 32, "fp16": 16, "int8": 8}


class EmbeddingQuantizer:
    """
    Quantize and dequantize QuatE embeddings.

    Supports:
    - fp32: No quantization (baseline)
    - fp16: Half precision (2x compression, lossless for practical purposes)
    - int8: 8-bit with global scale (4x compression, <2.2% rank swaps)

    WARNING: Do not use quaternion renormalization. Entity embeddings
    are not constrained to S^3 during training, so renormalization
    destroys the learned distance structure.
    """

    @staticmethod
    def quantize(embeddings: np.ndarray, dtype: str = "fp16") -> Dict[str, Any]:
        """
        Quantize embeddings to specified precision.

        Args:
            embeddings: Entity or relation embeddings (n, dim) as numpy array.
            dtype: Target precision ('fp32', 'fp16', 'int8')

        Returns:
            Dict containing quantized data and metadata for reconstruction.
        """
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}. Use one of {SUPPORTED_DTYPES}")

        # Handle torch tensors gracefully (no torch import required)
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.detach().cpu().numpy()

        emb = embeddings.astype(np.float32)

        if dtype == "fp32":
            return {
                "embeddings": emb.copy(),
                "dtype": "fp32",
                "shape": list(emb.shape),
            }
        elif dtype == "fp16":
            return {
                "embeddings": emb.astype(np.float16),
                "dtype": "fp16",
                "shape": list(emb.shape),
            }
        elif dtype == "int8":
            scale = float(np.abs(emb).max())
            if scale < 1e-8:
                scale = 1.0
            quantized = np.clip(
                np.round(emb / scale * 127.0), -127, 127
            ).astype(np.int8)
            return {
                "embeddings": quantized,
                "dtype": "int8",
                "scale": scale,
                "shape": list(emb.shape),
            }

    @staticmethod
    def dequantize(data: Dict[str, Any]) -> np.ndarray:
        """
        Dequantize embeddings back to float32.

        Args:
            data: Dict from quantize() containing embeddings and metadata.

        Returns:
            Float32 numpy array.
        """
        dtype = data["dtype"]
        embeddings = data["embeddings"]

        if dtype == "fp32":
            return embeddings.astype(np.float32)
        elif dtype == "fp16":
            return embeddings.astype(np.float32)
        elif dtype == "int8":
            scale = data["scale"]
            return embeddings.astype(np.float32) / 127.0 * scale
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

    @staticmethod
    def get_size_bytes(data: Dict[str, Any]) -> int:
        """Calculate storage size in bytes."""
        size = data["embeddings"].nbytes
        if data["dtype"] == "int8":
            size += 8  # float64 scale factor
        return size

    @staticmethod
    def get_compression_ratio(data: Dict[str, Any]) -> float:
        """Calculate compression ratio vs fp32."""
        shape = data["shape"]
        fp32_size = int(np.prod(shape)) * 4
        actual_size = EmbeddingQuantizer.get_size_bytes(data)
        return fp32_size / max(actual_size, 1)

    # Legacy API (backward compat)
    @staticmethod
    def compute_size_bytes(data: Dict[str, Any]) -> int:
        """Alias for get_size_bytes."""
        return EmbeddingQuantizer.get_size_bytes(data)
