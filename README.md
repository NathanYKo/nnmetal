# Neural Network Training on MNIST

Three implementations comparing CPU single-threaded, CPU multi-threaded, and GPU (Metal) performance for neural network training.

## Quick Start

### 1. Download MNIST Dataset
```bash
python3 download_mnist.py
```

### 2. Build All Implementations
```bash
make
```

### 3. Run Implementations

**CPU Single-threaded:**
```bash
./train_cpu_single
```

**CPU Multi-threaded:**
```bash
./train_cpu_multi
```

**GPU (Metal):**
```bash
./train_gpu
```

**Run all for comparison:**
```bash
make run-all
```

## What Each Implementation Does

- **train_cpu_single.mm** - Sequential CPU training (single thread)
- **train_cpu_multi.mm** - Parallel CPU training (multiple threads)
- **train_gpu.mm** - GPU-accelerated training using Metal

All implementations:
- Train a neural network (784 → 256 → 128 → 10) on MNIST
- Run for 100 epochs
- Display per-epoch timing and total training time
- Test on the test set and report accuracy

## Requirements

- macOS (for Metal GPU support)
- Xcode Command Line Tools
- Python 3 (for downloading MNIST dataset)

## Files

- `train_cpu_single.mm` - CPU single-threaded implementation
- `train_cpu_multi.mm` - CPU multi-threaded implementation
- `train_gpu.mm` - GPU Metal implementation
- `neural_network.metal` - Metal shader code
- `neural_net.h` - Neural network structure
- `mnist_loader.h` - MNIST dataset loader
- `benchmark_utils.h` - Timing utilities
- `download_mnist.py` - Script to download MNIST dataset
- `Makefile` - Build configuration

## Clean Build

```bash
make clean  # Remove compiled executables
make        # Rebuild all
```

## System Specification

**Epochs:**

100

**Learning Rate:**

0.01

**Data Size:**

60,000 training samples, 10,000 test samples (standard MNIST dataset)

**Topology:**

784 → 256 → 128 → 10

**System Information:**

Metal GPU framework (uses system default Metal device)

**Parallelization:**

- 256 threads per threadgroup
- One GPU thread per (batch_item, neuron) combination
- Batch size: 64 (multiple inputs processed in parallel)

**Other:**

- Shuffle data set every epoch
- Kernels and buffers are created once. These buffers are then written to every epoch.

