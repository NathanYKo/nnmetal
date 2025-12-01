# Makefile for Neural Network Training - Three Implementations

CXX = clang++
CXXFLAGS_CPU = -std=c++17 -O3
CXXFLAGS_GPU = -std=c++17 -O3 -framework Metal -framework Foundation -framework MetalKit

# Source files
CPU_SINGLE = train_cpu_single.mm
CPU_MULTI = train_cpu_multi.mm
GPU = train_gpu.mm
METAL_SOURCE = neural_network.metal

TARGETS = train_cpu_single train_cpu_multi train_gpu

# Default target
all: $(TARGETS)

# CPU Single-threaded
train_cpu_single: $(CPU_SINGLE)
	@echo "Compiling CPU Single-threaded implementation..."
	$(CXX) $(CXXFLAGS_CPU) $(CPU_SINGLE) -o train_cpu_single
	@echo "Build complete! Run with: ./train_cpu_single"

# CPU Multi-threaded
train_cpu_multi: $(CPU_MULTI)
	@echo "Compiling CPU Multi-threaded implementation..."
	$(CXX) $(CXXFLAGS_CPU) $(CPU_MULTI) -o train_cpu_multi
	@echo "Build complete! Run with: ./train_cpu_multi"

# GPU (Metal)
train_gpu: $(GPU) $(METAL_SOURCE)
	@echo "Compiling GPU (Metal) implementation..."
	$(CXX) $(CXXFLAGS_GPU) $(GPU) -o train_gpu
	@echo "Build complete! Run with: ./train_gpu"
	@echo "Note: Metal shader will be compiled at runtime from $(METAL_SOURCE)"

# Clean build artifacts
clean:
	rm -f $(TARGETS)

# Run all implementations (for comparison)
run-all: $(TARGETS)
	@echo "\n=== Running CPU Single-threaded ==="
	./train_cpu_single
	@echo "\n=== Running CPU Multi-threaded ==="
	./train_cpu_multi
	@echo "\n=== Running GPU (Metal) ==="
	./train_gpu

.PHONY: all clean run-all

