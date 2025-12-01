// Neural Network Training on MNIST using Metal
// GPU-accelerated training with parallelization comparison

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "benchmark_utils.h"
#include "neural_net.h"
#include "mnist_loader.h"
#include <vector>
#include <thread>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>

// CPU implementations for comparison
void cpu_forward_single(const float* input, const Layer& layer, float* output, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        const float* in = input + b * layer.input_size;
        float* out = output + b * layer.output_size;
        
        for (int out_idx = 0; out_idx < layer.output_size; out_idx++) {
            float sum = 0.0f;
            for (int i = 0; i < layer.input_size; i++) {
                sum += in[i] * layer.weights[out_idx * layer.input_size + i];
            }
            out[out_idx] = fmax(0.0f, sum + layer.bias[out_idx]);
        }
    }
}

void cpu_forward_multi(const float* input, const Layer& layer, float* output, int batch_size, int num_threads) {
    std::vector<std::thread> threads;
    int items_per_thread = batch_size / num_threads;
    
    auto worker = [&](int batch_start, int batch_end) {
        for (int b = batch_start; b < batch_end; b++) {
            const float* in = input + b * layer.input_size;
            float* out = output + b * layer.output_size;
            
            for (int out_idx = 0; out_idx < layer.output_size; out_idx++) {
                float sum = 0.0f;
                for (int i = 0; i < layer.input_size; i++) {
                    sum += in[i] * layer.weights[out_idx * layer.input_size + i];
                }
                out[out_idx] = fmax(0.0f, sum + layer.bias[out_idx]);
            }
        }
    };
    
    for (int t = 0; t < num_threads; t++) {
        int start = t * items_per_thread;
        int end = (t == num_threads - 1) ? batch_size : (t + 1) * items_per_thread;
        threads.emplace_back(worker, start, end);
    }
    
    for (auto& t : threads) t.join();
}

// Compute cross-entropy loss (for softmax output)
float compute_loss(const float* output, const float* targets, int batch_size, int num_classes) {
    float loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < num_classes; c++) {
            int idx = b * num_classes + c;
            float prob = output[idx];
            float target = targets[idx];
            // Cross-entropy: -sum(target * log(prob))
            // For one-hot encoding, target is 1.0 for true class, 0.0 for others
            if (target > 0.5f) {  // Only true class contributes
                loss -= logf(fmax(prob, 1e-10f));  // target is 1.0, so no need to multiply
            }
        }
    }
    return loss / (float)batch_size;
}

float compute_accuracy(const float* output, const float* targets, int batch_size, int num_classes) {
    int correct = 0;
    for (int b = 0; b < batch_size; b++) {
        int pred_class = 0;
        float max_val = output[b * num_classes];
        for (int c = 1; c < num_classes; c++) {
            if (output[b * num_classes + c] > max_val) {
                max_val = output[b * num_classes + c];
                pred_class = c;
            }
        }
        
        int true_class = 0;
        float max_target = targets[b * num_classes];
        for (int c = 1; c < num_classes; c++) {
            if (targets[b * num_classes + c] > max_target) {
                max_target = targets[b * num_classes + c];
                true_class = c;
            }
        }
        
        if (pred_class == true_class) {
            correct++;
        }
    }
    return (float)correct / (float)batch_size;
}

// Metal training context
struct MetalTrainingContext {
    __strong id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    std::vector<id<MTLComputePipelineState>> forwardPipelines;
    std::vector<id<MTLComputePipelineState>> forwardLinearPipelines;
    id<MTLComputePipelineState> softmaxPipeline;
    std::vector<id<MTLComputePipelineState>> backwardOutputPipelines;
    std::vector<id<MTLComputePipelineState>> backwardHiddenPipelines;
    std::vector<id<MTLComputePipelineState>> gradPipelines;
    std::vector<id<MTLComputePipelineState>> updatePipelines;
    std::vector<std::vector<id<MTLBuffer>>> buffers; // [input, weights, bias, output, deltas, weight_grads, bias_grads, targets]
    
    ~MetalTrainingContext() {
        device = nil;
        commandQueue = nil;
        forwardPipelines.clear();
        forwardLinearPipelines.clear();
        backwardOutputPipelines.clear();
        backwardHiddenPipelines.clear();
        gradPipelines.clear();
        updatePipelines.clear();
        buffers.clear();
    }
};

MetalTrainingContext* init_metal_training(const NeuralNet& net, int batch_size) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        
        if (!device) {
            NSArray<id<MTLDevice>>* allDevices = MTLCopyAllDevices();
            if (allDevices && allDevices.count > 0) {
                device = allDevices[0];
            }
            if (!device) {
                std::cerr << "Error: Failed to create Metal device" << std::endl;
                return nullptr;
            }
        }
        
        std::cout << Colors::CYAN << "Device: " << [device.name UTF8String] << Colors::RESET << std::endl;
        
        // Read Metal shader source
        std::ifstream file("neural_network.metal");
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open neural_network.metal" << std::endl;
            return nullptr;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string metalSourceStr = buffer.str();
        NSString* metalSource = [NSString stringWithUTF8String:metalSourceStr.c_str()];
        
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:metalSource options:nil error:&error];
        if (!library) {
            std::cerr << "Error: Failed to compile Metal shader: " << [error.localizedDescription UTF8String] << std::endl;
            return nullptr;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            std::cerr << "Error: Failed to create command queue" << std::endl;
            return nullptr;
        }
        
        MetalTrainingContext* ctx = new MetalTrainingContext();
        ctx->device = device;
        ctx->commandQueue = commandQueue;
        
        // Create pipeline states for each layer
        for (size_t layer_idx = 0; layer_idx < net.layers.size(); layer_idx++) {
            const auto& layer = net.layers[layer_idx];
            
            // Forward pipeline (ReLU)
            id<MTLFunction> forwardFunc = [library newFunctionWithName:@"forward_pass_batch"];
            if (!forwardFunc) {
                std::cerr << "Error: Failed to get forward_pass_batch function" << std::endl;
                delete ctx;
                return nullptr;
            }
            id<MTLComputePipelineState> forwardPipeline = [device newComputePipelineStateWithFunction:forwardFunc error:&error];
            if (!forwardPipeline) {
                std::cerr << "Error: Failed to create forward pipeline: " << [error.localizedDescription UTF8String] << std::endl;
                delete ctx;
                return nullptr;
            }
            ctx->forwardPipelines.push_back(forwardPipeline);
            
            // Forward linear pipeline (for output layer)
            id<MTLFunction> forwardLinearFunc = [library newFunctionWithName:@"forward_pass_batch_linear"];
            if (!forwardLinearFunc) {
                std::cerr << "Error: Failed to get forward_pass_batch_linear function" << std::endl;
                delete ctx;
                return nullptr;
            }
            id<MTLComputePipelineState> forwardLinearPipeline = [device newComputePipelineStateWithFunction:forwardLinearFunc error:&error];
            if (!forwardLinearPipeline) {
                std::cerr << "Error: Failed to create forward linear pipeline: " << [error.localizedDescription UTF8String] << std::endl;
                delete ctx;
                return nullptr;
            }
            ctx->forwardLinearPipelines.push_back(forwardLinearPipeline);
        }
        
        // Create softmax pipeline (only one, for output layer)
        id<MTLFunction> softmaxFunc = [library newFunctionWithName:@"softmax"];
        if (!softmaxFunc) {
            std::cerr << "Error: Failed to get softmax function" << std::endl;
            delete ctx;
            return nullptr;
        }
        id<MTLComputePipelineState> softmaxPipeline = [device newComputePipelineStateWithFunction:softmaxFunc error:&error];
        if (!softmaxPipeline) {
            std::cerr << "Error: Failed to create softmax pipeline: " << [error.localizedDescription UTF8String] << std::endl;
            delete ctx;
            return nullptr;
        }
        ctx->softmaxPipeline = softmaxPipeline;
        
        // Create backward pipelines for each layer
        for (size_t layer_idx = 0; layer_idx < net.layers.size(); layer_idx++) {
            // Backward output pipeline
            id<MTLFunction> bwdOutputFunc = [library newFunctionWithName:@"backward_pass_output"];
            id<MTLComputePipelineState> bwdOutputPipeline = [device newComputePipelineStateWithFunction:bwdOutputFunc error:&error];
            if (!bwdOutputPipeline) {
                std::cerr << "Error: Failed to create backward output pipeline" << std::endl;
                delete ctx;
                return nullptr;
            }
            ctx->backwardOutputPipelines.push_back(bwdOutputPipeline);
            
            // Backward hidden pipeline
            id<MTLFunction> bwdHiddenFunc = [library newFunctionWithName:@"backward_pass_hidden"];
            id<MTLComputePipelineState> bwdHiddenPipeline = [device newComputePipelineStateWithFunction:bwdHiddenFunc error:&error];
            if (!bwdHiddenPipeline) {
                std::cerr << "Error: Failed to create backward hidden pipeline" << std::endl;
                delete ctx;
                return nullptr;
            }
            ctx->backwardHiddenPipelines.push_back(bwdHiddenPipeline);
            
            // Gradient pipeline
            id<MTLFunction> gradFunc = [library newFunctionWithName:@"compute_weight_gradients"];
            id<MTLComputePipelineState> gradPipeline = [device newComputePipelineStateWithFunction:gradFunc error:&error];
            if (!gradPipeline) {
                std::cerr << "Error: Failed to create gradient pipeline" << std::endl;
                delete ctx;
                return nullptr;
            }
            ctx->gradPipelines.push_back(gradPipeline);
            
            // Update pipeline
            id<MTLFunction> updateFunc = [library newFunctionWithName:@"update_weights"];
            id<MTLComputePipelineState> updatePipeline = [device newComputePipelineStateWithFunction:updateFunc error:&error];
            if (!updatePipeline) {
                std::cerr << "Error: Failed to create update pipeline" << std::endl;
                delete ctx;
                return nullptr;
            }
            ctx->updatePipelines.push_back(updatePipeline);
            
            // Create buffers: [input, weights, bias, output, deltas, weight_grads, bias_grads, targets]
            std::vector<id<MTLBuffer>> bufs(8);
            
            const auto& layer = net.layers[layer_idx];
            int input_size = layer.input_size;
            int output_size = layer.output_size;
            
            bufs[0] = [device newBufferWithLength:sizeof(float) * batch_size * input_size
                                          options:MTLResourceStorageModeShared];
            bufs[1] = [device newBufferWithLength:sizeof(float) * output_size * input_size
                                          options:MTLResourceStorageModeShared];
            bufs[2] = [device newBufferWithLength:sizeof(float) * output_size
                                          options:MTLResourceStorageModeShared];
            bufs[3] = [device newBufferWithLength:sizeof(float) * batch_size * output_size
                                          options:MTLResourceStorageModeShared];
            bufs[4] = [device newBufferWithLength:sizeof(float) * batch_size * output_size
                                          options:MTLResourceStorageModeShared];
            // Direct gradient accumulation (not per-batch)
            bufs[5] = [device newBufferWithLength:sizeof(float) * output_size * input_size
                                          options:MTLResourceStorageModeShared];
            bufs[6] = [device newBufferWithLength:sizeof(float) * output_size
                                          options:MTLResourceStorageModeShared];
            bufs[7] = [device newBufferWithLength:sizeof(float) * batch_size * output_size
                                          options:MTLResourceStorageModeShared];
            
            // Initialize weights and bias
            memcpy([bufs[1] contents], layer.weights.data(), 
                   sizeof(float) * output_size * input_size);
            memcpy([bufs[2] contents], layer.bias.data(), 
                   sizeof(float) * output_size);
            
            ctx->buffers.push_back(bufs);
        }
        
        return ctx;
    }
}

// Metal training step
void metal_train_step(MetalTrainingContext* ctx, const NeuralNet& net, int batch_size,
                     const std::vector<float>& input, const std::vector<float>& targets,
                     float learning_rate, int actual_batch_size = -1) {
    // Use actual_batch_size if provided, otherwise use full batch_size
    int effective_batch_size = (actual_batch_size > 0) ? actual_batch_size : batch_size;
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        
        // Copy input to first layer
        memcpy([ctx->buffers[0][0] contents], input.data(),
               sizeof(float) * batch_size * net.layers[0].input_size);
        
        // Forward pass
        for (size_t layer_idx = 0; layer_idx < net.layers.size(); layer_idx++) {
            const auto& layer = net.layers[layer_idx];
            auto& bufs = ctx->buffers[layer_idx];
            
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            
            // Use linear for output layer, ReLU for hidden layers
            id<MTLComputePipelineState> pipeline = (layer_idx == net.layers.size() - 1)
                ? ctx->forwardLinearPipelines[layer_idx]
                : ctx->forwardPipelines[layer_idx];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:bufs[0] offset:0 atIndex:0];
            [encoder setBuffer:bufs[1] offset:0 atIndex:1];
            [encoder setBuffer:bufs[2] offset:0 atIndex:2];
            [encoder setBuffer:bufs[3] offset:0 atIndex:3];
            
            int input_size = layer.input_size;
            int output_size = layer.output_size;
            [encoder setBytes:&input_size length:sizeof(int) atIndex:4];
            [encoder setBytes:&output_size length:sizeof(int) atIndex:5];
            [encoder setBytes:&effective_batch_size length:sizeof(int) atIndex:6];
            
            MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
            MTLSize gridSize = MTLSizeMake(effective_batch_size * output_size, 1, 1);
            NSUInteger threadGroupCount = (gridSize.width + threadGroupSize.width - 1) / threadGroupSize.width;
            
            [encoder dispatchThreadgroups:MTLSizeMake(threadGroupCount, 1, 1) threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
            
            // Copy output for next layer (use GPU blit to preserve order in command buffer)
            if (layer_idx < net.layers.size() - 1) {
                id<MTLBlitCommandEncoder> copyEncoder = [commandBuffer blitCommandEncoder];
                [copyEncoder copyFromBuffer:bufs[3]
                                sourceOffset:0
                                    toBuffer:ctx->buffers[layer_idx + 1][0]
                           destinationOffset:0
                                        size:sizeof(float) * effective_batch_size * output_size];
                [copyEncoder endEncoding];
            }
        }
        
        // Apply softmax to output layer
        int last_idx = net.layers.size() - 1;
        const auto& last_layer = net.layers[last_idx];
        auto& last_bufs = ctx->buffers[last_idx];
        
        id<MTLComputeCommandEncoder> softmaxEncoder = [commandBuffer computeCommandEncoder];
        [softmaxEncoder setComputePipelineState:ctx->softmaxPipeline];
        [softmaxEncoder setBuffer:last_bufs[3] offset:0 atIndex:0];
        int output_size = last_layer.output_size;
        [softmaxEncoder setBytes:&output_size length:sizeof(int) atIndex:1];
        [softmaxEncoder setBytes:&effective_batch_size length:sizeof(int) atIndex:2];
        
        MTLSize softmaxGridSize = MTLSizeMake(effective_batch_size, 1, 1);
        NSUInteger softmaxThreadGroupCount = (softmaxGridSize.width + 255) / 256;
        [softmaxEncoder dispatchThreadgroups:MTLSizeMake(softmaxThreadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [softmaxEncoder endEncoding];
        
        // Backward pass - output layer
        
        // Copy targets
        memcpy([last_bufs[7] contents], targets.data(),
               sizeof(float) * effective_batch_size * last_layer.output_size);
        
        // Compute deltas
        id<MTLComputeCommandEncoder> bwdEncoder = [commandBuffer computeCommandEncoder];
        [bwdEncoder setComputePipelineState:ctx->backwardOutputPipelines[last_idx]];
        [bwdEncoder setBuffer:last_bufs[3] offset:0 atIndex:0];
        [bwdEncoder setBuffer:last_bufs[7] offset:0 atIndex:1];
        [bwdEncoder setBuffer:last_bufs[4] offset:0 atIndex:2];
        [bwdEncoder setBytes:&output_size length:sizeof(int) atIndex:3];
        [bwdEncoder setBytes:&effective_batch_size length:sizeof(int) atIndex:4];
        
        MTLSize bwdGridSize = MTLSizeMake(effective_batch_size * output_size, 1, 1);
        NSUInteger bwdThreadGroupCount = (bwdGridSize.width + 255) / 256;
        [bwdEncoder dispatchThreadgroups:MTLSizeMake(bwdThreadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [bwdEncoder endEncoding];
        
        // Zero gradients using Metal's fillBuffer (thread-safe)
        int input_size = last_layer.input_size;
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        [blitEncoder fillBuffer:last_bufs[5] range:NSMakeRange(0, sizeof(float) * input_size * output_size) value:0];
        [blitEncoder fillBuffer:last_bufs[6] range:NSMakeRange(0, sizeof(float) * output_size) value:0];
        [blitEncoder endEncoding];
        
        // Compute gradients (direct accumulation)
        // Need to dispatch at least max(input_size * output_size, output_size) threads
        // to ensure both weight and bias gradients are computed
        id<MTLComputeCommandEncoder> gradEncoder = [commandBuffer computeCommandEncoder];
        [gradEncoder setComputePipelineState:ctx->gradPipelines[last_idx]];
        [gradEncoder setBuffer:last_bufs[4] offset:0 atIndex:0];
        [gradEncoder setBuffer:last_bufs[0] offset:0 atIndex:1];
        [gradEncoder setBuffer:last_bufs[5] offset:0 atIndex:2];  // weight grads (direct)
        [gradEncoder setBuffer:last_bufs[6] offset:0 atIndex:3];  // bias grads (direct)
        [gradEncoder setBytes:&input_size length:sizeof(int) atIndex:4];
        [gradEncoder setBytes:&output_size length:sizeof(int) atIndex:5];
        [gradEncoder setBytes:&effective_batch_size length:sizeof(int) atIndex:6];
        // Dispatch enough threads for both weights and bias
        int total_weights = input_size * output_size;
        int total_threads = (total_weights > output_size) ? total_weights : output_size;
        MTLSize gradGridSize = MTLSizeMake(total_threads, 1, 1);
        NSUInteger gradThreadGroupCount = (gradGridSize.width + 255) / 256;
        [gradEncoder dispatchThreadgroups:MTLSizeMake(gradThreadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [gradEncoder endEncoding];
        
        // Update weights (using direct gradients)
        id<MTLComputeCommandEncoder> updateEncoder = [commandBuffer computeCommandEncoder];
        [updateEncoder setComputePipelineState:ctx->updatePipelines[last_idx]];
        [updateEncoder setBuffer:last_bufs[1] offset:0 atIndex:0];
        [updateEncoder setBuffer:last_bufs[5] offset:0 atIndex:1];  // weight grads
        [updateEncoder setBuffer:last_bufs[2] offset:0 atIndex:2];
        [updateEncoder setBuffer:last_bufs[6] offset:0 atIndex:3];  // bias grads
        [updateEncoder setBytes:&input_size length:sizeof(int) atIndex:4];
        [updateEncoder setBytes:&output_size length:sizeof(int) atIndex:5];
        [updateEncoder setBytes:&learning_rate length:sizeof(float) atIndex:6];
        [updateEncoder setBytes:&effective_batch_size length:sizeof(int) atIndex:7];
        
        MTLSize updateGridSize = MTLSizeMake(input_size * output_size, 1, 1);
        NSUInteger updateThreadGroupCount = (updateGridSize.width + 255) / 256;
        [updateEncoder dispatchThreadgroups:MTLSizeMake(updateThreadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [updateEncoder endEncoding];
        
        // Backward through hidden layers
        for (int layer_idx = net.layers.size() - 2; layer_idx >= 0; layer_idx--) {
            const auto& layer = net.layers[layer_idx];
            const auto& next_layer = net.layers[layer_idx + 1];
            auto& bufs = ctx->buffers[layer_idx];
            auto& next_bufs = ctx->buffers[layer_idx + 1];
            
            // Compute deltas
            id<MTLComputeCommandEncoder> hiddenEncoder = [commandBuffer computeCommandEncoder];
            [hiddenEncoder setComputePipelineState:ctx->backwardHiddenPipelines[layer_idx]];
            [hiddenEncoder setBuffer:bufs[3] offset:0 atIndex:0];
            [hiddenEncoder setBuffer:next_bufs[4] offset:0 atIndex:1];
            [hiddenEncoder setBuffer:next_bufs[1] offset:0 atIndex:2];
            [hiddenEncoder setBuffer:bufs[4] offset:0 atIndex:3];
            int layer_output_size = layer.output_size;
            int next_output_size = next_layer.output_size;
            [hiddenEncoder setBytes:&layer_output_size length:sizeof(int) atIndex:4];
            [hiddenEncoder setBytes:&next_output_size length:sizeof(int) atIndex:5];
            [hiddenEncoder setBytes:&effective_batch_size length:sizeof(int) atIndex:6];
            
            MTLSize hiddenGridSize = MTLSizeMake(effective_batch_size * layer_output_size, 1, 1);
            NSUInteger hiddenThreadGroupCount = (hiddenGridSize.width + 255) / 256;
            [hiddenEncoder dispatchThreadgroups:MTLSizeMake(hiddenThreadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [hiddenEncoder endEncoding];
            
            // Zero gradients using Metal's fillBuffer (thread-safe)
            int layer_input_size = layer.input_size;
            id<MTLBlitCommandEncoder> layerBlitEncoder = [commandBuffer blitCommandEncoder];
            [layerBlitEncoder fillBuffer:bufs[5] range:NSMakeRange(0, sizeof(float) * layer_input_size * layer_output_size) value:0];
            [layerBlitEncoder fillBuffer:bufs[6] range:NSMakeRange(0, sizeof(float) * layer_output_size) value:0];
            [layerBlitEncoder endEncoding];
            
            // Compute gradients (direct accumulation)
            // Need to dispatch at least max(input_size * output_size, output_size) threads
            id<MTLComputeCommandEncoder> layerGradEncoder = [commandBuffer computeCommandEncoder];
            [layerGradEncoder setComputePipelineState:ctx->gradPipelines[layer_idx]];
            [layerGradEncoder setBuffer:bufs[4] offset:0 atIndex:0];
            [layerGradEncoder setBuffer:bufs[0] offset:0 atIndex:1];
            [layerGradEncoder setBuffer:bufs[5] offset:0 atIndex:2];  // weight grads (direct)
            [layerGradEncoder setBuffer:bufs[6] offset:0 atIndex:3];  // bias grads (direct)
            [layerGradEncoder setBytes:&layer_input_size length:sizeof(int) atIndex:4];
            [layerGradEncoder setBytes:&layer_output_size length:sizeof(int) atIndex:5];
            [layerGradEncoder setBytes:&effective_batch_size length:sizeof(int) atIndex:6];
            // Dispatch enough threads for both weights and bias
            int layer_total_weights = layer_input_size * layer_output_size;
            int layer_total_threads = (layer_total_weights > layer_output_size) ? layer_total_weights : layer_output_size;
            MTLSize layerGradGridSize = MTLSizeMake(layer_total_threads, 1, 1);
            NSUInteger layerGradThreadGroupCount = (layerGradGridSize.width + 255) / 256;
            [layerGradEncoder dispatchThreadgroups:MTLSizeMake(layerGradThreadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [layerGradEncoder endEncoding];
            
            // Update weights (using direct gradients)
            id<MTLComputeCommandEncoder> layerUpdateEncoder = [commandBuffer computeCommandEncoder];
            [layerUpdateEncoder setComputePipelineState:ctx->updatePipelines[layer_idx]];
            [layerUpdateEncoder setBuffer:bufs[1] offset:0 atIndex:0];
            [layerUpdateEncoder setBuffer:bufs[5] offset:0 atIndex:1];  // weight grads
            [layerUpdateEncoder setBuffer:bufs[2] offset:0 atIndex:2];
            [layerUpdateEncoder setBuffer:bufs[6] offset:0 atIndex:3];  // bias grads
            [layerUpdateEncoder setBytes:&layer_input_size length:sizeof(int) atIndex:4];
            [layerUpdateEncoder setBytes:&layer_output_size length:sizeof(int) atIndex:5];
            [layerUpdateEncoder setBytes:&learning_rate length:sizeof(float) atIndex:6];
            [layerUpdateEncoder setBytes:&effective_batch_size length:sizeof(int) atIndex:7];
            
            MTLSize layerUpdateGridSize = MTLSizeMake(layer_input_size * layer_output_size, 1, 1);
            NSUInteger layerUpdateThreadGroupCount = (layerUpdateGridSize.width + 255) / 256;
            [layerUpdateEncoder dispatchThreadgroups:MTLSizeMake(layerUpdateThreadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [layerUpdateEncoder endEncoding];
        }
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

int main() {
    std::cout << Colors::MAGENTA << "\n=== Neural Network Training on MNIST (Metal) ===" << Colors::RESET << std::endl;
    std::cout << "GPU-accelerated training with parallelization comparison\n" << std::endl;
    
    // Configuration
    const int batch_size = 64;
    const int epochs = 25;  // More epochs for better accuracy
    const float initial_learning_rate = 0.01f;  // Proper learning rate for softmax + cross-entropy
    
    // MNIST: 28x28 = 784 input pixels, 10 output classes
    // Larger network for better accuracy (more capacity)
    NeuralNet net({{784, 256}, {256, 128}, {128, 10}});
    // Note: net is mutable so we can sync weights back from GPU after training
    
    std::cout << Colors::CYAN << "Configuration:" << Colors::RESET << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Initial learning rate: " << initial_learning_rate << " (with decay)" << std::endl;
    std::cout << "  Network: " << net.layers[0].input_size;
    for (const auto& layer : net.layers) {
        std::cout << " → " << layer.output_size;
    }
    std::cout << std::endl << std::endl;
    
    // Load MNIST dataset
    std::cout << Colors::YELLOW << "Loading MNIST dataset..." << Colors::RESET << std::endl;
    std::vector<std::vector<float>> train_images, train_labels_onehot;
    std::vector<std::vector<float>> test_images, test_labels_onehot;
    
    if (!load_mnist_dataset(train_images, train_labels_onehot, test_images, test_labels_onehot, false)) {
        std::cerr << "Failed to load MNIST dataset" << std::endl;
        return 1;
    }
    
    std::cout << Colors::GREEN << "Dataset loaded successfully!" << Colors::RESET << std::endl << std::endl;
    
    // Initialize Metal
    MetalTrainingContext* ctx = init_metal_training(net, batch_size);
    if (!ctx) {
        std::cerr << "Failed to initialize Metal" << std::endl;
        return 1;
    }
    
    // Benchmark CPU vs GPU forward pass
    std::cout << Colors::YELLOW << "Benchmarking forward pass performance..." << Colors::RESET << std::endl;
    
    std::vector<float> test_input(batch_size * 784);
    for (int i = 0; i < batch_size; i++) {
        std::copy(train_images[i % train_images.size()].begin(), 
                 train_images[i % train_images.size()].end(),
                 test_input.begin() + i * 784);
    }
    
    // CPU single-threaded
    std::vector<std::vector<float>> cpu_single_outputs(net.layers.size());
    for (size_t i = 0; i < net.layers.size(); i++) {
        cpu_single_outputs[i].resize(batch_size * net.layers[i].output_size);
    }
    
    auto cpu_single_start = std::chrono::high_resolution_clock::now();
    const float* curr_input = test_input.data();
    for (size_t l = 0; l < net.layers.size(); l++) {
        cpu_forward_single(curr_input, net.layers[l], cpu_single_outputs[l].data(), batch_size);
        curr_input = cpu_single_outputs[l].data();
    }
    auto cpu_single_end = std::chrono::high_resolution_clock::now();
    double cpu_single_time = std::chrono::duration<double, std::milli>(cpu_single_end - cpu_single_start).count();
    
    // CPU multi-threaded
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::vector<float>> cpu_multi_outputs(net.layers.size());
    for (size_t i = 0; i < net.layers.size(); i++) {
        cpu_multi_outputs[i].resize(batch_size * net.layers[i].output_size);
    }
    
    auto cpu_multi_start = std::chrono::high_resolution_clock::now();
    curr_input = test_input.data();
    for (size_t l = 0; l < net.layers.size(); l++) {
        cpu_forward_multi(curr_input, net.layers[l], cpu_multi_outputs[l].data(), batch_size, num_threads);
        curr_input = cpu_multi_outputs[l].data();
    }
    auto cpu_multi_end = std::chrono::high_resolution_clock::now();
    double cpu_multi_time = std::chrono::duration<double, std::milli>(cpu_multi_end - cpu_multi_start).count();
    
    // GPU (Metal) forward pass
    memcpy([ctx->buffers[0][0] contents], test_input.data(), sizeof(float) * batch_size * 784);
    
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];
        for (size_t l = 0; l < net.layers.size(); l++) {
            const auto& layer = net.layers[l];
            auto& bufs = ctx->buffers[l];
            
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            id<MTLComputePipelineState> pipeline = (l == net.layers.size() - 1)
                ? ctx->forwardLinearPipelines[l]
                : ctx->forwardPipelines[l];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:bufs[0] offset:0 atIndex:0];
            [encoder setBuffer:bufs[1] offset:0 atIndex:1];
            [encoder setBuffer:bufs[2] offset:0 atIndex:2];
            [encoder setBuffer:bufs[3] offset:0 atIndex:3];
            
            int input_size = layer.input_size;
            int output_size = layer.output_size;
            [encoder setBytes:&input_size length:sizeof(int) atIndex:4];
            [encoder setBytes:&output_size length:sizeof(int) atIndex:5];
            [encoder setBytes:&batch_size length:sizeof(int) atIndex:6];
            
            MTLSize gridSize = MTLSizeMake(batch_size * output_size, 1, 1);
            NSUInteger threadGroupCount = (gridSize.width + 255) / 256;
            [encoder dispatchThreadgroups:MTLSizeMake(threadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            
            if (l < net.layers.size() - 1) {
                id<MTLBlitCommandEncoder> copyEncoder = [cmdBuf blitCommandEncoder];
                [copyEncoder copyFromBuffer:bufs[3]
                                sourceOffset:0
                                    toBuffer:ctx->buffers[l+1][0]
                           destinationOffset:0
                                        size:sizeof(float) * batch_size * output_size];
                [copyEncoder endEncoding];
            }
        }
        
        // Apply softmax to output layer
        int last_layer_idx = net.layers.size() - 1;
        const auto& last_layer = net.layers[last_layer_idx];
        auto& last_bufs = ctx->buffers[last_layer_idx];
        
        id<MTLComputeCommandEncoder> softmaxEncoder = [cmdBuf computeCommandEncoder];
        [softmaxEncoder setComputePipelineState:ctx->softmaxPipeline];
        [softmaxEncoder setBuffer:last_bufs[3] offset:0 atIndex:0];
        int output_size = last_layer.output_size;
        [softmaxEncoder setBytes:&output_size length:sizeof(int) atIndex:1];
        [softmaxEncoder setBytes:&batch_size length:sizeof(int) atIndex:2];
        
        MTLSize softmaxGridSize = MTLSizeMake(batch_size, 1, 1);
        NSUInteger softmaxThreadGroupCount = (softmaxGridSize.width + 255) / 256;
        [softmaxEncoder dispatchThreadgroups:MTLSizeMake(softmaxThreadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [softmaxEncoder endEncoding];
        
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];
        for (size_t l = 0; l < net.layers.size(); l++) {
            const auto& layer = net.layers[l];
            auto& bufs = ctx->buffers[l];
            
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            id<MTLComputePipelineState> pipeline = (l == net.layers.size() - 1)
                ? ctx->forwardLinearPipelines[l]
                : ctx->forwardPipelines[l];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:bufs[0] offset:0 atIndex:0];
            [encoder setBuffer:bufs[1] offset:0 atIndex:1];
            [encoder setBuffer:bufs[2] offset:0 atIndex:2];
            [encoder setBuffer:bufs[3] offset:0 atIndex:3];
            
            int input_size = layer.input_size;
            int output_size = layer.output_size;
            [encoder setBytes:&input_size length:sizeof(int) atIndex:4];
            [encoder setBytes:&output_size length:sizeof(int) atIndex:5];
            [encoder setBytes:&batch_size length:sizeof(int) atIndex:6];
            
            MTLSize gridSize = MTLSizeMake(batch_size * output_size, 1, 1);
            NSUInteger threadGroupCount = (gridSize.width + 255) / 256;
            [encoder dispatchThreadgroups:MTLSizeMake(threadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            
            if (l < net.layers.size() - 1) {
                id<MTLBlitCommandEncoder> copyEncoder = [cmdBuf blitCommandEncoder];
                [copyEncoder copyFromBuffer:bufs[3]
                                sourceOffset:0
                                    toBuffer:ctx->buffers[l+1][0]
                           destinationOffset:0
                                        size:sizeof(float) * batch_size * output_size];
                [copyEncoder endEncoding];
            }
        }
        
        // Apply softmax to output layer
        int last_layer_idx = net.layers.size() - 1;
        const auto& last_layer = net.layers[last_layer_idx];
        auto& last_bufs = ctx->buffers[last_layer_idx];
        
        id<MTLComputeCommandEncoder> softmaxEncoder = [cmdBuf computeCommandEncoder];
        [softmaxEncoder setComputePipelineState:ctx->softmaxPipeline];
        [softmaxEncoder setBuffer:last_bufs[3] offset:0 atIndex:0];
        int output_size = last_layer.output_size;
        [softmaxEncoder setBytes:&output_size length:sizeof(int) atIndex:1];
        [softmaxEncoder setBytes:&batch_size length:sizeof(int) atIndex:2];
        
        MTLSize softmaxGridSize = MTLSizeMake(batch_size, 1, 1);
        NSUInteger softmaxThreadGroupCount = (softmaxGridSize.width + 255) / 256;
        [softmaxEncoder dispatchThreadgroups:MTLSizeMake(softmaxThreadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [softmaxEncoder endEncoding];
        
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "\n" << Colors::CYAN << "=== Forward Pass Performance ===" << Colors::RESET << std::endl;
    std::cout << "CPU Single: " << std::fixed << std::setprecision(2) << cpu_single_time << " ms" << std::endl;
    std::cout << "CPU Multi (" << num_threads << " threads): " << cpu_multi_time << " ms" << std::endl;
    std::cout << "GPU (Metal): " << gpu_time << " ms" << std::endl;
    std::cout << "\n⚡ Speedup:" << std::endl;
    std::cout << "  GPU vs CPU Single: " << std::setprecision(2) << (cpu_single_time / gpu_time) << "x" << std::endl;
    std::cout << "  GPU vs CPU Multi: " << (cpu_multi_time / gpu_time) << "x" << std::endl;
    std::cout << "  CPU Multi vs Single: " << (cpu_single_time / cpu_multi_time) << "x" << std::endl;
    
    // Training loop with proper shuffling
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Create indices for shuffling
    std::vector<size_t> train_indices(train_images.size());
    for (size_t i = 0; i < train_indices.size(); i++) {
        train_indices[i] = i;
    }
    
    std::cout << "\n" << Colors::YELLOW << "Training..." << Colors::RESET << std::endl;
    
    // Store initial weights for comparison
    std::vector<std::vector<float>> initial_weights(net.layers.size());
    for (size_t i = 0; i < net.layers.size(); i++) {
        initial_weights[i].resize(10);  // Store first 10 weights of each layer
        memcpy(initial_weights[i].data(), [ctx->buffers[i][1] contents], sizeof(float) * 10);
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Learning rate decay: reduce by factor of 0.9 every 5 epochs
        float learning_rate = initial_learning_rate * powf(0.9f, epoch / 5.0f);
        
        // Shuffle training data each epoch
        std::shuffle(train_indices.begin(), train_indices.end(), gen);
        
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        int num_batches = 0;
        
        // Train on all data, properly shuffled
        int total_batches = (train_images.size() + batch_size - 1) / batch_size;
        for (int batch = 0; batch < total_batches; batch++) {
            int batch_start = batch * batch_size;
            int batch_end = std::min(batch_start + batch_size, (int)train_images.size());
            int current_batch_size = batch_end - batch_start;
            
            std::vector<float> batch_input(batch_size * 784, 0.0f);
            std::vector<float> batch_targets(batch_size * 10, 0.0f);
            
            for (int b = 0; b < current_batch_size; b++) {
                size_t idx = train_indices[batch_start + b];
                std::copy(train_images[idx].begin(), train_images[idx].end(),
                         batch_input.begin() + b * 784);
                std::copy(train_labels_onehot[idx].begin(), train_labels_onehot[idx].end(),
                         batch_targets.begin() + b * 10);
            }
            
            // Training step (pass actual batch size to avoid training on padding)
            metal_train_step(ctx, net, batch_size, batch_input, batch_targets, learning_rate, current_batch_size);
            
            // Delta analysis diagnostic
            if (epoch == 0 && batch == 0) {
                std::cout << "\n=== Delta Analysis ===" << std::endl;
                
                for (size_t i = 0; i < net.layers.size(); i++) {
                    std::vector<float> deltas(net.layers[i].output_size);
                    memcpy(deltas.data(), [ctx->buffers[i][4] contents], 
                           sizeof(float) * net.layers[i].output_size);
                    
                    float sum = 0.0f;
                    int zeros = 0;
                    for (auto d : deltas) {
                        sum += fabs(d);
                        if (fabs(d) < 1e-10f) zeros++;
                    }
                    
                    std::cout << "Layer " << i << " (" << net.layers[i].output_size << " neurons):" << std::endl;
                    std::cout << "  Avg |delta|: " << (sum / deltas.size()) << std::endl;
                    std::cout << "  Zero deltas: " << zeros << " / " << deltas.size() << std::endl;
                    
                    // Also check first few actual delta values
                    std::cout << "  First 5 deltas: ";
                    for (int j = 0; j < std::min(5, (int)deltas.size()); j++) {
                        std::cout << deltas[j] << " ";
                    }
                    std::cout << std::endl;
                    
                    // Check weight gradients for Layer 0
                    if (i == 0) {
                        std::vector<float> weight_grads(10);
                        memcpy(weight_grads.data(), [ctx->buffers[i][5] contents], sizeof(float) * 10);
                        std::cout << "  First 10 weight gradients: ";
                        for (int j = 0; j < 10; j++) {
                            std::cout << std::fixed << std::setprecision(6) << weight_grads[j] << " ";
                        }
                        std::cout << std::endl;
                        
                        // Check input buffer (should have actual input data)
                        std::vector<float> input_sample(10);
                        memcpy(input_sample.data(), [ctx->buffers[i][0] contents], sizeof(float) * 10);
                        std::cout << "  First 10 input values: ";
                        for (int j = 0; j < 10; j++) {
                            std::cout << std::fixed << std::setprecision(4) << input_sample[j] << " ";
                        }
                        std::cout << std::endl;
                    }
                }
            }
            
            // Compute loss and accuracy
            std::vector<float> output(batch_size * 10);
            memcpy(output.data(), [ctx->buffers.back()[3] contents],
                   sizeof(float) * batch_size * 10);
            
            // Debug output for first batch of first epoch
            if (batch == 0 && epoch == 0) {
                std::cout << "\n" << Colors::CYAN << "=== Debug: First Batch ===" << Colors::RESET << std::endl;
                std::cout << "First sample softmax output: ";
                for (int i = 0; i < 10; i++) {
                    std::cout << std::fixed << std::setprecision(4) << output[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "First sample target: ";
                for (int i = 0; i < 10; i++) {
                    std::cout << batch_targets[i] << " ";
                }
                std::cout << std::endl;
                
                // Check if softmax sums to 1
                float sum = 0.0f;
                for (int i = 0; i < 10; i++) {
                    sum += output[i];
                }
                std::cout << "Softmax sum: " << sum << std::endl;
                
                // Check gradients after first batch
                int last_layer_idx = net.layers.size() - 1;
                auto& last_bufs = ctx->buffers[last_layer_idx];
                
                std::vector<float> weight_grads(10);
                memcpy(weight_grads.data(), [last_bufs[5] contents], sizeof(float) * 10);
                std::cout << "First 10 weight gradients (output layer): ";
                for (int i = 0; i < 10; i++) {
                    std::cout << std::fixed << std::setprecision(6) << weight_grads[i] << " ";
                }
                std::cout << std::endl;
                
                std::vector<float> bias_grads(10);
                memcpy(bias_grads.data(), [last_bufs[6] contents], sizeof(float) * 10);
                std::cout << "Bias gradients (output layer): ";
                for (int i = 0; i < 10; i++) {
                    std::cout << std::fixed << std::setprecision(6) << bias_grads[i] << " ";
                }
                std::cout << std::endl;
            }
            
            // Check weight changes after first epoch
            if (batch == 0 && epoch == 1) {
                std::cout << "\n" << Colors::CYAN << "=== Debug: After First Epoch ===" << Colors::RESET << std::endl;
                for (size_t layer_idx = 0; layer_idx < net.layers.size(); layer_idx++) {
                    std::vector<float> updated_weights(10);
                    memcpy(updated_weights.data(), [ctx->buffers[layer_idx][1] contents], sizeof(float) * 10);
                    
                    std::cout << "Layer " << layer_idx << " weight changes (first 10): ";
                    for (int i = 0; i < 10; i++) {
                        float change = updated_weights[i] - initial_weights[layer_idx][i];
                        std::cout << std::fixed << std::setprecision(6) << change << " ";
                    }
                    std::cout << std::endl;
                }
            }
            
            float loss = compute_loss(output.data(), batch_targets.data(), current_batch_size, 10);
            float acc = compute_accuracy(output.data(), batch_targets.data(), current_batch_size, 10);
            
            epoch_loss += loss;
            epoch_acc += acc;
            num_batches++;
        }
        
        epoch_loss /= num_batches;
        epoch_acc /= num_batches;
        
        std::cout << "Epoch " << epoch << ": Loss = " << std::fixed << std::setprecision(4) 
                  << epoch_loss << ", Accuracy = " << std::setprecision(2) 
                  << (epoch_acc * 100.0f) << "%, LR = " << std::setprecision(4) << learning_rate << std::endl;
    }
    
    // Test on test set
    std::cout << std::endl << Colors::YELLOW << "Testing on test set..." << Colors::RESET << std::endl;
    float test_loss = 0.0f;
    float test_acc = 0.0f;
    int test_batches = 0;
    
    for (size_t i = 0; i < test_images.size(); i += batch_size) {
        int current_batch_size = std::min(batch_size, (int)(test_images.size() - i));
        
        std::vector<float> batch_input(batch_size * 784, 0.0f);
        std::vector<float> batch_targets(batch_size * 10, 0.0f);
        
        for (int b = 0; b < current_batch_size; b++) {
            std::copy(test_images[i + b].begin(), test_images[i + b].end(),
                     batch_input.begin() + b * 784);
            std::copy(test_labels_onehot[i + b].begin(), test_labels_onehot[i + b].end(),
                     batch_targets.begin() + b * 10);
        }
        
        // Forward pass only (no training)
        memcpy([ctx->buffers[0][0] contents], batch_input.data(), sizeof(float) * batch_size * 784);
        
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];
            for (size_t l = 0; l < net.layers.size(); l++) {
                const auto& layer = net.layers[l];
                auto& bufs = ctx->buffers[l];
                
                id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                id<MTLComputePipelineState> pipeline = (l == net.layers.size() - 1)
                    ? ctx->forwardLinearPipelines[l]
                    : ctx->forwardPipelines[l];
                
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:bufs[0] offset:0 atIndex:0];
                [encoder setBuffer:bufs[1] offset:0 atIndex:1];
                [encoder setBuffer:bufs[2] offset:0 atIndex:2];
                [encoder setBuffer:bufs[3] offset:0 atIndex:3];
                
                int input_size = layer.input_size;
                int output_size = layer.output_size;
                [encoder setBytes:&input_size length:sizeof(int) atIndex:4];
                [encoder setBytes:&output_size length:sizeof(int) atIndex:5];
                [encoder setBytes:&batch_size length:sizeof(int) atIndex:6];
                
                MTLSize gridSize = MTLSizeMake(batch_size * output_size, 1, 1);
                NSUInteger threadGroupCount = (gridSize.width + 255) / 256;
                [encoder dispatchThreadgroups:MTLSizeMake(threadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
                
                if (l < net.layers.size() - 1) {
                    id<MTLBlitCommandEncoder> copyEncoder = [cmdBuf blitCommandEncoder];
                    [copyEncoder copyFromBuffer:bufs[3]
                                    sourceOffset:0
                                        toBuffer:ctx->buffers[l+1][0]
                               destinationOffset:0
                                            size:sizeof(float) * batch_size * output_size];
                    [copyEncoder endEncoding];
                }
            }
            
            // Apply softmax to output layer
            int last_layer_idx = net.layers.size() - 1;
            const auto& last_layer = net.layers[last_layer_idx];
            auto& last_bufs = ctx->buffers[last_layer_idx];
            
            id<MTLComputeCommandEncoder> softmaxEncoder = [cmdBuf computeCommandEncoder];
            [softmaxEncoder setComputePipelineState:ctx->softmaxPipeline];
            [softmaxEncoder setBuffer:last_bufs[3] offset:0 atIndex:0];
            int output_size = last_layer.output_size;
            [softmaxEncoder setBytes:&output_size length:sizeof(int) atIndex:1];
            [softmaxEncoder setBytes:&batch_size length:sizeof(int) atIndex:2];
            
            MTLSize softmaxGridSize = MTLSizeMake(batch_size, 1, 1);
            NSUInteger softmaxThreadGroupCount = (softmaxGridSize.width + 255) / 256;
            [softmaxEncoder dispatchThreadgroups:MTLSizeMake(softmaxThreadGroupCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [softmaxEncoder endEncoding];
            
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
        }
        
        std::vector<float> output(batch_size * 10);
        memcpy(output.data(), [ctx->buffers.back()[3] contents],
               sizeof(float) * batch_size * 10);
        
        float loss = compute_loss(output.data(), batch_targets.data(), current_batch_size, 10);
        float acc = compute_accuracy(output.data(), batch_targets.data(), current_batch_size, 10);
        
        test_loss += loss;
        test_acc += acc;
        test_batches++;
    }
    
    test_loss /= test_batches;
    test_acc /= test_batches;
    
    std::cout << Colors::GREEN << "\nTest Results:" << Colors::RESET << std::endl;
    std::cout << "  Loss: " << std::fixed << std::setprecision(4) << test_loss << std::endl;
    std::cout << "  Accuracy: " << std::setprecision(2) << (test_acc * 100.0f) << "%" << std::endl;
    
    // Critical Fix #1: Copy trained weights back from GPU to CPU
    std::cout << Colors::YELLOW << "\nSynchronizing weights from GPU to CPU..." << Colors::RESET << std::endl;
    for (size_t i = 0; i < net.layers.size(); i++) {
        auto& bufs = ctx->buffers[i];
        memcpy(net.layers[i].weights.data(), 
               [bufs[1] contents],
               sizeof(float) * net.layers[i].weights.size());
        memcpy(net.layers[i].bias.data(),
               [bufs[2] contents], 
               sizeof(float) * net.layers[i].bias.size());
    }
    std::cout << Colors::GREEN << "Weights synchronized!" << Colors::RESET << std::endl;
    
    std::cout << Colors::GREEN << "\nTraining complete!" << Colors::RESET << std::endl;
    
    delete ctx;
    return 0;
}

