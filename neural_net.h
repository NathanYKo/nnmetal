#pragma once

#include <vector>
#include <cmath>
#include <random>

struct Layer {
    int input_size;
    int output_size;
    std::vector<float> weights;
    std::vector<float> bias;
    
    void init(int in_size, int out_size) {
        input_size = in_size;
        output_size = out_size;
        weights.resize(input_size * output_size);
        bias.resize(output_size);
        
        float scale = sqrtf(2.0f / (float)input_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] = dist(gen) * scale;
        }
        for (size_t i = 0; i < bias.size(); i++) {
            bias[i] = 0.0f;
        }
    }
    
    long long ops(int batch_size) const {
        return 2LL * batch_size * input_size * output_size;
    }
};

class NeuralNet {
public:
    std::vector<Layer> layers;
    
    NeuralNet(const std::vector<std::pair<int, int>>& layer_sizes) {
        for (const auto& [in_size, out_size] : layer_sizes) {
            Layer layer;
            layer.init(in_size, out_size);
            layers.push_back(layer);
        }
    }
    
    long long total_ops(int batch_size) const {
        long long total = 0;
        for (const auto& layer : layers) {
            total += layer.ops(batch_size);
        }
        return total;
    }
    
    long long total_neurons(int batch_size) const {
        long long total = 0;
        for (const auto& layer : layers) {
            total += (long long)batch_size * layer.output_size;
        }
        return total;
    }
};

