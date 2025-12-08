# Adaptive Router Core (CUDA 12.x) - GPU-Accelerated Inference Engine

High-performance C++ inference core for Adaptive Router with **CUDA 12.x GPU acceleration**, Python bindings, and C FFI API.

## Overview

This is the **CUDA-enabled** variant of `adaptive-router-core`, providing GPU-accelerated cluster assignment and model scoring for maximum throughput. It automatically falls back to CPU execution when no CUDA GPU is available.

**Key differences from CPU variant (`adaptive-router-core`):**
- Built with CUDA 12.x support for GPU acceleration
- Significantly higher throughput for batch operations
- Same API and interface as CPU variant
- Runtime CPU fallback when GPU not available

## Features

- **GPU-accelerated cluster assignment** using CUDA kernels
- **Ultra-fast batch processing** for high-throughput scenarios
- **Automatic fallback** to CPU when CUDA is unavailable
- **Zero-copy Python integration** via nanobind
- **C FFI API** for integration with other languages
- **Linux x86_64 only** (CUDA limitation)

## Requirements

### Runtime Requirements

- **Linux x86_64** (CUDA is Linux-only for this package)
- **CUDA 12.x compatible GPU** (optional - falls back to CPU)
- **CUDA Runtime 12.x** (if using GPU acceleration)
- **Python 3.11+**

### Build Requirements (for building from source)

- **NVIDIA CUDA Toolkit 12.x**
- **C++20 compiler**: GCC 11+
- **CMake** 3.24+
- **Conan** 2.x (package manager)

## Installation

### From PyPI

```bash
# Install CUDA variant
pip install adaptive-router-core-cu12

# Or with uv
uv pip install adaptive-router-core-cu12
```

### Building from Source

```bash
cd adaptive_router_core_cu12

# Install dependencies via Conan
conan install . --output-folder=build --build=missing --settings=build_type=Release

# Configure CMake with CUDA enabled
cmake --preset conan-release -DADAPTIVE_ENABLE_CUDA=ON

# Build
cmake --build build/build/Release

# Install Python package
pip install .
```

## Usage

The API is identical to the CPU variant. Simply import and use:

```python
from adaptive_core_ext import Router
import numpy as np

# Load router from profile
router = Router.from_file("profile.json")

# Single routing (GPU-accelerated)
embedding = np.random.randn(384).astype(np.float32)
response = router.route(embedding, cost_bias=0.5)

print(f"Selected: {response.selected_model}")
print(f"Cluster: {response.cluster_id}")

# Batch routing (optimized for GPU)
embeddings = np.random.randn(1000, 384).astype(np.float32)
responses = router.route_batch(embeddings, cost_bias=0.5)

print(f"Processed {len(responses)} embeddings")
```

## Performance

### GPU vs CPU Comparison (NVIDIA L40S)

| Operation | CPU Latency | GPU Latency | Speedup |
|-----------|-------------|-------------|---------|
| Single route | ~0.2ms | ~0.1ms | 2x |
| Batch (100) | ~20ms | ~1ms | **20x** |
| Batch (1000) | ~200ms | ~5ms | **40x** |

GPU acceleration provides the most significant speedup for batch operations.

## When to Use This Package

**Use `adaptive-router-core-cu12` (this package) when:**
- Running on Linux with NVIDIA GPU
- Processing high volumes of requests (batch routing)
- Deploying to cloud instances with GPU (e.g., Modal, AWS GPU instances)

**Use `adaptive-router-core` (CPU variant) when:**
- Running on macOS or Windows
- Running on Linux without GPU
- Lower throughput requirements
- Simpler deployment without CUDA dependencies

## API Reference

See the [adaptive-router-core documentation](https://github.com/Egham-7/adaptive) for the complete API reference. The API is identical between CPU and CUDA variants.

### Python API

```python
class Router:
    @staticmethod
    def from_file(path: str) -> Router: ...

    @staticmethod
    def from_json_string(json_str: str) -> Router: ...

    def route(self, embedding: np.ndarray, cost_bias: float = 0.5) -> RouteResponse: ...
    def route_batch(self, embeddings: np.ndarray, cost_bias: float = 0.5) -> list[RouteResponse]: ...
    def get_supported_models(self) -> list[str]: ...
    def get_n_clusters(self) -> int: ...
    def get_embedding_dim(self) -> int: ...

class RouteResponse:
    selected_model: str
    alternatives: list[str]
    cluster_id: int
    cluster_distance: float
```

## Troubleshooting

### "CUDA not available" warning

This is normal if no CUDA GPU is present. The router will automatically use CPU execution:

```
Warning: CUDA not available, falling back to CPU execution
```

### "CUDA version mismatch" error

Ensure your CUDA runtime matches the package:
```bash
nvidia-smi  # Check CUDA version
```

This package requires CUDA 12.x.

### Import error on non-Linux systems

This package only supports Linux x86_64. Use `adaptive-router-core` (CPU variant) for macOS/Windows.

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Related Packages

- **[adaptive-router](https://pypi.org/project/adaptive-router/)** - Main Python library
- **[adaptive-router-core](https://pypi.org/project/adaptive-router-core/)** - CPU-only C++ core

## Support

- **Issues**: [GitHub Issues](https://github.com/Egham-7/adaptive/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Egham-7/adaptive/discussions)
