#include "benchmark_utils.h"
#include "neural_net.h"
#include "mnist_loader.h"
#include <vector>
#include <thread>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>

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

void cpu_backward_multi(NeuralNet& net,
                       const float* input,
                       const std::vector<std::vector<float>>& layer_outputs,
                       const float* targets, int batch_size, float learning_rate, int num_threads) {
    int last_layer_idx = net.layers.size() - 1;
    Layer& output_layer = net.layers[last_layer_idx];
    
    std::vector<float> deltas(batch_size * output_layer.output_size);
    int items_per_thread = batch_size / num_threads;
    std::vector<std::thread> threads;
    
    auto delta_worker = [&](int batch_start, int batch_end) {
        for (int b = batch_start; b < batch_end; b++) {
            for (int o = 0; o < output_layer.output_size; o++) {
                int idx = b * output_layer.output_size + o;
                deltas[idx] = layer_outputs[last_layer_idx][idx] - targets[idx];
            }
        }
    };
    
    for (int t = 0; t < num_threads; t++) {
        int start = t * items_per_thread;
        int end = (t == num_threads - 1) ? batch_size : (t + 1) * items_per_thread;
        threads.emplace_back(delta_worker, start, end);
    }
    for (auto& t : threads) t.join();
    threads.clear();
    
    std::vector<float> weight_grads(output_layer.input_size * output_layer.output_size, 0.0f);
    std::vector<float> bias_grads(output_layer.output_size, 0.0f);
    
    auto grad_worker = [&](int output_start, int output_end) {
        for (int o = output_start; o < output_end; o++) {
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
    };
    
    int outputs_per_thread = output_layer.output_size / num_threads;
    for (int t = 0; t < num_threads; t++) {
        int start = t * outputs_per_thread;
        int end = (t == num_threads - 1) ? output_layer.output_size : (t + 1) * outputs_per_thread;
        threads.emplace_back(grad_worker, start, end);
    }
    for (auto& t : threads) t.join();
    threads.clear();
    
    auto update_worker = [&](int output_start, int output_end) {
        for (int o = output_start; o < output_end; o++) {
            for (int i = 0; i < output_layer.input_size; i++) {
                float grad = fmax(-5.0f, fmin(5.0f, weight_grads[o * output_layer.input_size + i]));
                output_layer.weights[o * output_layer.input_size + i] -= learning_rate * grad;
            }
            float grad = fmax(-5.0f, fmin(5.0f, bias_grads[o]));
            output_layer.bias[o] -= learning_rate * grad;
        }
    };
    
    for (int t = 0; t < num_threads; t++) {
        int start = t * outputs_per_thread;
        int end = (t == num_threads - 1) ? output_layer.output_size : (t + 1) * outputs_per_thread;
        threads.emplace_back(update_worker, start, end);
    }
    for (auto& t : threads) t.join();
    threads.clear();
    
    std::vector<float> next_deltas = deltas;
    for (int layer_idx = last_layer_idx - 1; layer_idx >= 0; layer_idx--) {
        Layer& layer = net.layers[layer_idx];
        Layer& next_layer = net.layers[layer_idx + 1];
        
        std::vector<float> layer_deltas(batch_size * layer.output_size, 0.0f);
        auto hidden_delta_worker = [&](int batch_start, int batch_end) {
            for (int b = batch_start; b < batch_end; b++) {
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
        };
        
        items_per_thread = batch_size / num_threads;
        for (int t = 0; t < num_threads; t++) {
            int start = t * items_per_thread;
            int end = (t == num_threads - 1) ? batch_size : (t + 1) * items_per_thread;
            threads.emplace_back(hidden_delta_worker, start, end);
        }
        for (auto& t : threads) t.join();
        threads.clear();
        
        std::vector<float> weight_grads(layer.input_size * layer.output_size, 0.0f);
        std::vector<float> bias_grads(layer.output_size, 0.0f);
        
        outputs_per_thread = layer.output_size / num_threads;
        auto hidden_grad_worker = [&](int output_start, int output_end) {
            for (int o = output_start; o < output_end; o++) {
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
        };
        
        for (int t = 0; t < num_threads; t++) {
            int start = t * outputs_per_thread;
            int end = (t == num_threads - 1) ? layer.output_size : (t + 1) * outputs_per_thread;
            threads.emplace_back(hidden_grad_worker, start, end);
        }
        for (auto& t : threads) t.join();
        threads.clear();
        
        auto hidden_update_worker = [&](int output_start, int output_end) {
            for (int o = output_start; o < output_end; o++) {
                for (int i = 0; i < layer.input_size; i++) {
                    float grad = fmax(-5.0f, fmin(5.0f, weight_grads[o * layer.input_size + i]));
                    layer.weights[o * layer.input_size + i] -= learning_rate * grad;
                }
                float grad = fmax(-5.0f, fmin(5.0f, bias_grads[o]));
                layer.bias[o] -= learning_rate * grad;
            }
        };
        
        for (int t = 0; t < num_threads; t++) {
            int start = t * outputs_per_thread;
            int end = (t == num_threads - 1) ? layer.output_size : (t + 1) * outputs_per_thread;
            threads.emplace_back(hidden_update_worker, start, end);
        }
        for (auto& t : threads) t.join();
        threads.clear();
        
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
    std::cout << Colors::MAGENTA << "\n=== Neural Network Training on MNIST (CPU Multi-threaded) ===" << Colors::RESET << std::endl;
    
    const int batch_size = 64;
    const int epochs = 100;
    const float initial_learning_rate = 0.01f;
    int num_threads = std::thread::hardware_concurrency();
    
    NeuralNet net({{784, 256}, {256, 128}, {128, 10}});
    
    std::cout << Colors::CYAN << "Configuration:" << Colors::RESET << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  Learning rate: " << initial_learning_rate << std::endl;
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
    
    // Split dataset: use first 50,000 for training, last 5,000 for testing
    const int train_size = 50000;
    const int test_size = 5000;
    
    if (train_images.size() < train_size + test_size) {
        std::cerr << "Error: Not enough training samples. Need " << (train_size + test_size) 
                  << " but only have " << train_images.size() << std::endl;
        return 1;
    }
    
    // Use last 5,000 from training set as test set
    test_images.clear();
    test_labels_onehot.clear();
    test_images.insert(test_images.end(), train_images.begin() + train_size, train_images.end());
    test_labels_onehot.insert(test_labels_onehot.end(), train_labels_onehot.begin() + train_size, train_labels_onehot.end());
    
    // Resize training set to first 55,000
    train_images.resize(train_size);
    train_labels_onehot.resize(train_size);
    
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
        float learning_rate = initial_learning_rate;
        std::shuffle(train_indices.begin(), train_indices.end(), gen);
        
        int total_batches = (train_images.size() + batch_size - 1) / batch_size;
        
        // Process batches in parallel - each thread handles one batch
        int batches_per_thread = std::max(1, total_batches / num_threads);
        std::vector<std::thread> batch_threads;
        std::mutex net_mutex;
        
        auto batch_worker = [&](int batch_start_idx, int batch_end_idx) {
            NeuralNet local_net = net;  // Local copy for this thread
            
            for (int batch = batch_start_idx; batch < batch_end_idx; batch++) {
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
                
                std::vector<std::vector<float>> layer_outputs(local_net.layers.size());
                const float* curr_input = batch_input.data();
                for (size_t l = 0; l < local_net.layers.size(); l++) {
                    layer_outputs[l].resize(batch_size * local_net.layers[l].output_size);
                    cpu_forward_multi(curr_input, local_net.layers[l], layer_outputs[l].data(), batch_size, 1);
                    curr_input = layer_outputs[l].data();
                }
                
                cpu_softmax(layer_outputs.back().data(), batch_size, 10);
                
                cpu_backward_multi(local_net, batch_input.data(), layer_outputs,
                                  batch_targets.data(), batch_size, learning_rate, 1);
            }
            
            // Update shared network with synchronized access
            std::lock_guard<std::mutex> lock(net_mutex);
            for (size_t l = 0; l < net.layers.size(); l++) {
                for (size_t i = 0; i < net.layers[l].weights.size(); i++) {
                    net.layers[l].weights[i] = local_net.layers[l].weights[i];
                }
                for (size_t i = 0; i < net.layers[l].bias.size(); i++) {
                    net.layers[l].bias[i] = local_net.layers[l].bias[i];
                }
            }
        };
        
        for (int t = 0; t < num_threads; t++) {
            int start = t * batches_per_thread;
            int end = (t == num_threads - 1) ? total_batches : (t + 1) * batches_per_thread;
            if (start < total_batches) {
                batch_threads.emplace_back(batch_worker, start, end);
            }
        }
        
        for (auto& t : batch_threads) {
            t.join();
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_time = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();
        
        std::cout << "Epoch " << epoch << ": Time = " << std::setprecision(2) << (epoch_time / 1000.0) << "s" << std::endl;
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
            cpu_forward_multi(curr_input, net.layers[l], layer_outputs[l].data(), batch_size, num_threads);
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

