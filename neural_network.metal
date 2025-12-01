#include <metal_stdlib>
using namespace metal;

// Batch processing kernel - processes entire batch in parallel
kernel void forward_pass_batch(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& input_size [[buffer(4)]],
    constant int& output_size [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    uint global_id [[thread_position_in_grid]]
) {
    int total_work_items = batch_size * output_size;
    
    if (global_id < total_work_items) {
        int batch_idx = global_id / output_size;
        int output_idx = global_id % output_size;
        
        float sum = 0.0f;
        int input_offset = batch_idx * input_size;
        
        for (int i = 0; i < input_size; i++) {
            sum += input[input_offset + i] * weights[output_idx * input_size + i];
        }
        
        output[batch_idx * output_size + output_idx] = fmax(0.0f, sum + bias[output_idx]);
    }
}

// Forward pass without activation (for output layer in classification)
kernel void forward_pass_batch_linear(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& input_size [[buffer(4)]],
    constant int& output_size [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    uint global_id [[thread_position_in_grid]]
) {
    int total_work_items = batch_size * output_size;
    
    if (global_id < total_work_items) {
        int batch_idx = global_id / output_size;
        int output_idx = global_id % output_size;
        
        float sum = 0.0f;
        int input_offset = batch_idx * input_size;
        
        for (int i = 0; i < input_size; i++) {
            sum += input[input_offset + i] * weights[output_idx * input_size + i];
        }
        
        // Linear output (no activation) - will apply softmax separately
        output[batch_idx * output_size + output_idx] = sum + bias[output_idx];
    }
}

// Apply softmax to output layer (for classification)
kernel void softmax(
    device float* output [[buffer(0)]],
    constant int& output_size [[buffer(1)]],
    constant int& batch_size [[buffer(2)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id < batch_size) {
        int batch_idx = global_id;
        int offset = batch_idx * output_size;
        
        // Find max for numerical stability
        float max_val = output[offset];
        for (int i = 1; i < output_size; i++) {
            max_val = fmax(max_val, output[offset + i]);
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++) {
            float val = output[offset + i] - max_val;
            // Only clip very negative values (they become exp(-50) ≈ 0 anyway)
            // Don't clip positive values - they're safe after subtracting max
            val = fmax(-50.0f, val);
            float exp_val = exp(val);
            output[offset + i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        sum = fmax(sum, 1e-10f);
        for (int i = 0; i < output_size; i++) {
            output[offset + i] /= sum;
            // Ensure probabilities are in valid range (avoid exactly 0 or 1 for numerical stability)
            output[offset + i] = fmax(1e-10f, fmin(1.0f - 1e-10f, output[offset + i]));
        }
    }
}

// Backward pass for output layer - computes deltas
// For softmax + cross-entropy, the gradient simplifies to (softmax_out - target)
kernel void backward_pass_output(
    device const float* output [[buffer(0)]],  // softmax output
    device const float* targets [[buffer(1)]],
    device float* deltas [[buffer(2)]],
    constant int& output_size [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    uint global_id [[thread_position_in_grid]]
) {
    int total_work_items = batch_size * output_size;
    
    if (global_id < total_work_items) {
        int batch_idx = global_id / output_size;
        int output_idx = global_id % output_size;
        
        int out_idx = batch_idx * output_size + output_idx;
        float out_val = output[out_idx];  // softmax probability
        float target_val = targets[out_idx];  // one-hot target
        
        // For softmax + cross-entropy: gradient = (softmax_out - target)
        float delta = (out_val - target_val);
        deltas[out_idx] = delta;
    }
}

// Compute weight gradients from deltas (direct accumulation)
kernel void compute_weight_gradients(
    device const float* deltas [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* weight_gradients [[buffer(2)]],  // [output_size * input_size] - directly accumulated
    device float* bias_gradients [[buffer(3)]],     // [output_size] - directly accumulated
    constant int& input_size [[buffer(4)]],
    constant int& output_size [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    uint global_id [[thread_position_in_grid]]
) {
    int total_weights = output_size * input_size;
    
    if (global_id < total_weights) {
        int output_idx = global_id / input_size;
        int input_idx = global_id % input_size;
        
        // Accumulate gradients across all batch items
        float grad_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            int delta_idx = b * output_size + output_idx;
            int input_offset = b * input_size + input_idx;
            grad_sum += deltas[delta_idx] * input[input_offset];
        }
        weight_gradients[global_id] = grad_sum;
    }
    
    // Accumulate bias gradients across batch
    if (global_id < output_size) {
        float bias_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            bias_sum += deltas[b * output_size + global_id];
        }
        bias_gradients[global_id] = bias_sum;
    }
}


// Backward pass for hidden layers - computes deltas
kernel void backward_pass_hidden(
    device const float* output [[buffer(0)]],
    device const float* next_delta [[buffer(1)]],
    device const float* next_weights [[buffer(2)]],
    device float* deltas [[buffer(3)]],
    constant int& output_size [[buffer(4)]],
    constant int& next_output_size [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    uint global_id [[thread_position_in_grid]]
) {
    int total_work_items = batch_size * output_size;
    
    if (global_id < total_work_items) {
        int batch_idx = global_id / output_size;
        int output_idx = global_id % output_size;
        
        int out_idx = batch_idx * output_size + output_idx;
        float out_val = output[out_idx];
        
        float delta = 0.0f;
        for (int j = 0; j < next_output_size; j++) {
            int next_delta_idx = batch_idx * next_output_size + j;
            delta += next_delta[next_delta_idx] * next_weights[j * output_size + output_idx];
        }
        delta *= (out_val > 0.0f ? 1.0f : 0.0f);
        
        deltas[out_idx] = delta;
    }
}

// Update weights using SGD with gradient clipping
kernel void update_weights(
    device float* weights [[buffer(0)]],
    device const float* weight_gradients [[buffer(1)]],
    device float* bias [[buffer(2)]],
    device const float* bias_gradients [[buffer(3)]],
    constant int& input_size [[buffer(4)]],
    constant int& output_size [[buffer(5)]],
    constant float& learning_rate [[buffer(6)]],
    constant int& batch_size [[buffer(7)]],
    uint global_id [[thread_position_in_grid]]
) {
    int total_weights = output_size * input_size;
    
    if (global_id < total_weights) {
        // Average gradient over batch
        float grad = weight_gradients[global_id] / (float)batch_size;
        // Clip to prevent explosion
        grad = fmax(-5.0f, fmin(5.0f, grad));
        // Check for NaN/Inf
        if (!isnan(grad) && !isinf(grad)) {
            weights[global_id] -= learning_rate * grad;
            // Clip weights to prevent explosion
            weights[global_id] = fmax(-100.0f, fmin(100.0f, weights[global_id]));
        }
    }
    
    if (global_id < output_size) {
        // Average gradient over batch
        float grad = bias_gradients[global_id] / (float)batch_size;
        // Clip to prevent explosion
        grad = fmax(-5.0f, fmin(5.0f, grad));
        // Check for NaN/Inf
        if (!isnan(grad) && !isinf(grad)) {
            bias[global_id] -= learning_rate * grad;
            // Clip bias to prevent explosion
            bias[global_id] = fmax(-100.0f, fmin(100.0f, bias[global_id]));
        }
    }
}

