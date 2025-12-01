#include "benchmark_utils.h"
#include "neural_net.h"
#include "mnist_loader.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>

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

void cpu_softmax(float* output, int batch_size, int output_size) {
    for (int b = 0; b < batch_size; b++) {
        int offset = b * output_size;
        float max_val = output[offset];
        for (int i = 1; i < output_size; i++) {
            max_val = fmax(max_val, output[offset + i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++) {
            float val = fmax(-50.0f, output[offset + i] - max_val);
            float exp_val = expf(val);
            output[offset + i] = exp_val;
            sum += exp_val;
        }
        sum = fmax(sum, 1e-10f);
        for (int i = 0; i < output_size; i++) {
            output[offset + i] = fmax(1e-10f, fmin(1.0f - 1e-10f, output[offset + i] / sum));
        }
    }
}

void cpu_backward_single(NeuralNet& net, 
                        const float* input,
                        const std::vector<std::vector<float>>& layer_outputs,
                        const float* targets, int batch_size, float learning_rate) {
    int last_layer_idx = net.layers.size() - 1;
    Layer& output_layer = net.layers[last_layer_idx];
    
    std::vector<float> deltas(batch_size * output_layer.output_size);
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_layer.output_size; o++) {
            int idx = b * output_layer.output_size + o;
            deltas[idx] = layer_outputs[last_layer_idx][idx] - targets[idx];
        }
    }
    
    std::vector<float> weight_grads(output_layer.input_size * output_layer.output_size, 0.0f);
    std::vector<float> bias_grads(output_layer.output_size, 0.0f);
    
    for (int o = 0; o < output_layer.output_size; o++) {
        for (int i = 0; i < output_layer.input_size; i++) {
            float grad = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                int delta_idx = b * output_layer.output_size + o;
                int input_idx = b * output_layer.input_size + i;
                grad += deltas[delta_idx] * layer_outputs[last_layer_idx - 1][input_idx];
            }
            weight_grads[o * output_layer.input_size + i] = grad / batch_size;
        }
        
        float bias_grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            bias_grad += deltas[b * output_layer.output_size + o];
        }
        bias_grads[o] = bias_grad / batch_size;
    }
    
    for (int o = 0; o < output_layer.output_size; o++) {
        for (int i = 0; i < output_layer.input_size; i++) {
            float grad = fmax(-5.0f, fmin(5.0f, weight_grads[o * output_layer.input_size + i]));
            output_layer.weights[o * output_layer.input_size + i] -= learning_rate * grad;
        }
        float grad = fmax(-5.0f, fmin(5.0f, bias_grads[o]));
        output_layer.bias[o] -= learning_rate * grad;
    }
    
    std::vector<float> next_deltas = deltas;
    for (int layer_idx = last_layer_idx - 1; layer_idx >= 0; layer_idx--) {
        Layer& layer = net.layers[layer_idx];
        Layer& next_layer = net.layers[layer_idx + 1];
        
        std::vector<float> layer_deltas(batch_size * layer.output_size, 0.0f);
        for (int b = 0; b < batch_size; b++) {
            for (int o = 0; o < layer.output_size; o++) {
                float delta = 0.0f;
                for (int no = 0; no < next_layer.output_size; no++) {
                    int next_delta_idx = b * next_layer.output_size + no;
                    delta += next_deltas[next_delta_idx] * next_layer.weights[no * layer.output_size + o];
                }
                int out_idx = b * layer.output_size + o;
                delta *= (layer_outputs[layer_idx][out_idx] > 0.0f ? 1.0f : 0.0f);
                layer_deltas[out_idx] = delta;
            }
        }
        
        std::vector<float> weight_grads(layer.input_size * layer.output_size, 0.0f);
        std::vector<float> bias_grads(layer.output_size, 0.0f);
        
        for (int o = 0; o < layer.output_size; o++) {
            for (int i = 0; i < layer.input_size; i++) {
                float grad = 0.0f;
                for (int b = 0; b < batch_size; b++) {
                    int delta_idx = b * layer.output_size + o;
                    int input_idx = b * layer.input_size + i;
                    const float* prev_output = (layer_idx == 0) ? input : layer_outputs[layer_idx - 1].data();
                    grad += layer_deltas[delta_idx] * prev_output[input_idx];
                }
                weight_grads[o * layer.input_size + i] = grad / batch_size;
            }
            
            float bias_grad = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                bias_grad += layer_deltas[b * layer.output_size + o];
            }
            bias_grads[o] = bias_grad / batch_size;
        }
        
        for (int o = 0; o < layer.output_size; o++) {
            for (int i = 0; i < layer.input_size; i++) {
                float grad = fmax(-5.0f, fmin(5.0f, weight_grads[o * layer.input_size + i]));
                layer.weights[o * layer.input_size + i] -= learning_rate * grad;
            }
            float grad = fmax(-5.0f, fmin(5.0f, bias_grads[o]));
            layer.bias[o] -= learning_rate * grad;
        }
        
        next_deltas = layer_deltas;
    }
}

float compute_loss(const float* output, const float* targets, int batch_size, int num_classes) {
    float loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < num_classes; c++) {
            int idx = b * num_classes + c;
            float prob = output[idx];
            float target = targets[idx];
            if (target > 0.5f) {
                loss -= logf(fmax(prob, 1e-10f));
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

int main() {
    std::cout << Colors::MAGENTA << "\n=== Neural Network Training on MNIST (CPU Single-threaded) ===" << Colors::RESET << std::endl;
    
    const int batch_size = 64;
    const int epochs = 5;
    const float initial_learning_rate = 0.01f;
    
    NeuralNet net({{784, 256}, {256, 128}, {128, 10}});
    
    std::cout << Colors::CYAN << "Configuration:" << Colors::RESET << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Initial learning rate: " << initial_learning_rate << " (with decay)" << std::endl;
    std::cout << "  Network: " << net.layers[0].input_size;
    for (const auto& layer : net.layers) {
        std::cout << " → " << layer.output_size;
    }
    std::cout << std::endl << std::endl;
    
    std::cout << Colors::YELLOW << "Loading MNIST dataset..." << Colors::RESET << std::endl;
    std::vector<std::vector<float>> train_images, train_labels_onehot;
    std::vector<std::vector<float>> test_images, test_labels_onehot;
    
    if (!load_mnist_dataset(train_images, train_labels_onehot, test_images, test_labels_onehot, false)) {
        std::cerr << "Failed to load MNIST dataset" << std::endl;
        return 1;
    }
    
    std::cout << Colors::GREEN << "Dataset loaded successfully!" << Colors::RESET << std::endl << std::endl;
    
    std::cout << Colors::YELLOW << "=== Benchmarking on Full Training Dataset ===" << Colors::RESET << std::endl;
    std::cout << "Training samples: " << train_images.size() << std::endl;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<size_t> train_indices(train_images.size());
    for (size_t i = 0; i < train_indices.size(); i++) {
        train_indices[i] = i;
    }
    
    std::cout << "\n" << Colors::YELLOW << "Training..." << Colors::RESET << std::endl;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        float learning_rate = initial_learning_rate * powf(0.9f, epoch / 5.0f);
        std::shuffle(train_indices.begin(), train_indices.end(), gen);
        
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        int num_batches = 0;
        
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
            
            std::vector<std::vector<float>> layer_outputs(net.layers.size());
            const float* curr_input = batch_input.data();
            for (size_t l = 0; l < net.layers.size(); l++) {
                layer_outputs[l].resize(batch_size * net.layers[l].output_size);
                cpu_forward_single(curr_input, net.layers[l], layer_outputs[l].data(), batch_size);
                curr_input = layer_outputs[l].data();
            }
            
            cpu_softmax(layer_outputs.back().data(), batch_size, 10);
            
            cpu_backward_single(net, batch_input.data(), layer_outputs, 
                               batch_targets.data(), batch_size, learning_rate);
            
            float loss = compute_loss(layer_outputs.back().data(), batch_targets.data(), current_batch_size, 10);
            float acc = compute_accuracy(layer_outputs.back().data(), batch_targets.data(), current_batch_size, 10);
            
            epoch_loss += loss;
            epoch_acc += acc;
            num_batches++;
        }
        
        epoch_loss /= num_batches;
        epoch_acc /= num_batches;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_time = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();
        
        std::cout << "Epoch " << epoch << ": Loss = " << std::fixed << std::setprecision(4) 
                  << epoch_loss << ", Accuracy = " << std::setprecision(2) 
                  << (epoch_acc * 100.0f) << "%, LR = " << std::setprecision(4) << learning_rate
                  << ", Time = " << std::setprecision(2) << (epoch_time / 1000.0) << "s" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    std::cout << "\n" << Colors::CYAN << "=== Performance Summary ===" << Colors::RESET << std::endl;
    std::cout << Colors::GREEN << "Total training time: " << std::fixed << std::setprecision(2) 
              << (total_time / 1000.0) << " seconds (" << total_time << " ms)" << Colors::RESET << std::endl;
    std::cout << "Average time per epoch: " << std::setprecision(2) 
              << (total_time / epochs / 1000.0) << " seconds" << std::endl;
    std::cout << "Training samples processed: " << train_images.size() * epochs << std::endl;
    std::cout << "Throughput: " << std::setprecision(2) 
              << (train_images.size() * epochs / (total_time / 1000.0)) << " samples/second" << std::endl;
    
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
        
        std::vector<std::vector<float>> layer_outputs(net.layers.size());
        const float* curr_input = batch_input.data();
        for (size_t l = 0; l < net.layers.size(); l++) {
            layer_outputs[l].resize(batch_size * net.layers[l].output_size);
            cpu_forward_single(curr_input, net.layers[l], layer_outputs[l].data(), batch_size);
            curr_input = layer_outputs[l].data();
        }
        cpu_softmax(layer_outputs.back().data(), batch_size, 10);
        
        float loss = compute_loss(layer_outputs.back().data(), batch_targets.data(), current_batch_size, 10);
        float acc = compute_accuracy(layer_outputs.back().data(), batch_targets.data(), current_batch_size, 10);
        
        test_loss += loss;
        test_acc += acc;
        test_batches++;
    }
    
    test_loss /= test_batches;
    test_acc /= test_batches;
    
    std::cout << Colors::GREEN << "\nTest Results:" << Colors::RESET << std::endl;
    std::cout << "  Loss: " << std::fixed << std::setprecision(4) << test_loss << std::endl;
    std::cout << "  Accuracy: " << std::setprecision(2) << (test_acc * 100.0f) << "%" << std::endl;
    
    std::cout << Colors::GREEN << "\nTraining complete!" << Colors::RESET << std::endl;
    
    return 0;
}

