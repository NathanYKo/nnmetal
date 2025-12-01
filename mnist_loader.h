// MNIST Dataset Loader
// Downloads and parses MNIST IDX format files

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <algorithm>

// Reverse byte order for endianness conversion
inline uint32_t reverse_bytes(uint32_t val) {
    return ((val & 0x000000FF) << 24) |
           ((val & 0x0000FF00) << 8) |
           ((val & 0x00FF0000) >> 8) |
           ((val & 0xFF000000) >> 24);
}

// Download MNIST files if they don't exist
bool download_mnist_file(const std::string& url, const std::string& filename) {
    std::ifstream test_file(filename);
    if (test_file.good()) {
        std::cout << "File " << filename << " already exists, skipping download." << std::endl;
        return true;
    }
    
    std::string command = "curl -L -o " + filename + " " + url;
    std::cout << "Downloading " << filename << "..." << std::endl;
    int result = system(command.c_str());
    return result == 0;
}

// Load MNIST images (IDX format)
bool load_mnist_images(const std::string& filename, std::vector<std::vector<float>>& images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return false;
    }
    
    // Read magic number
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    magic = reverse_bytes(magic);
    if (magic != 2051) {
        std::cerr << "Error: Invalid magic number for MNIST images" << std::endl;
        return false;
    }
    
    // Read number of images
    uint32_t num_images;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = reverse_bytes(num_images);
    
    // Read dimensions
    uint32_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    rows = reverse_bytes(rows);
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    cols = reverse_bytes(cols);
    
    std::cout << "Loading " << num_images << " images of size " << rows << "x" << cols << std::endl;
    
    images.clear();
    images.resize(num_images);
    
    // Read image data
    for (uint32_t i = 0; i < num_images; i++) {
        images[i].resize(rows * cols);
        for (uint32_t j = 0; j < rows * cols; j++) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            // Normalize to [0, 1]
            images[i][j] = static_cast<float>(pixel) / 255.0f;
        }
    }
    
    return true;
}

// Load MNIST labels (IDX format)
bool load_mnist_labels(const std::string& filename, std::vector<int>& labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return false;
    }
    
    // Read magic number
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    magic = reverse_bytes(magic);
    if (magic != 2049) {
        std::cerr << "Error: Invalid magic number for MNIST labels" << std::endl;
        return false;
    }
    
    // Read number of labels
    uint32_t num_labels;
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = reverse_bytes(num_labels);
    
    labels.clear();
    labels.resize(num_labels);
    
    // Read label data
    for (uint32_t i = 0; i < num_labels; i++) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }
    
    return true;
}

// Convert labels to one-hot encoding
void labels_to_onehot(const std::vector<int>& labels, std::vector<std::vector<float>>& onehot, int num_classes = 10) {
    onehot.clear();
    onehot.resize(labels.size());
    
    for (size_t i = 0; i < labels.size(); i++) {
        onehot[i].resize(num_classes, 0.0f);
        if (labels[i] >= 0 && labels[i] < num_classes) {
            onehot[i][labels[i]] = 1.0f;
        }
    }
}

// Load MNIST dataset
bool load_mnist_dataset(std::vector<std::vector<float>>& train_images,
                       std::vector<std::vector<float>>& train_labels_onehot,
                       std::vector<std::vector<float>>& test_images,
                       std::vector<std::vector<float>>& test_labels_onehot,
                       bool download = true) {
    
    // Use a reliable source for MNIST
    const std::string base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    const std::string train_images_file = "train-images-idx3-ubyte";
    const std::string train_labels_file = "train-labels-idx1-ubyte";
    const std::string test_images_file = "t10k-images-idx3-ubyte";
    const std::string test_labels_file = "t10k-labels-idx1-ubyte";
    
    // Check if uncompressed files already exist
    std::ifstream test1(train_images_file, std::ios::binary);
    std::ifstream test2(train_labels_file, std::ios::binary);
    std::ifstream test3(test_images_file, std::ios::binary);
    std::ifstream test4(test_labels_file, std::ios::binary);
    
    bool file1_ok = test1.is_open() && test1.good();
    bool file2_ok = test2.is_open() && test2.good();
    bool file3_ok = test3.is_open() && test3.good();
    bool file4_ok = test4.is_open() && test4.good();
    
    test1.close();
    test2.close();
    test3.close();
    test4.close();
    
    if (!file1_ok || !file2_ok || !file3_ok || !file4_ok) {
        std::cerr << "Error: MNIST files not found:" << std::endl;
        if (!file1_ok) std::cerr << "  Missing: " << train_images_file << std::endl;
        if (!file2_ok) std::cerr << "  Missing: " << train_labels_file << std::endl;
        if (!file3_ok) std::cerr << "  Missing: " << test_images_file << std::endl;
        if (!file4_ok) std::cerr << "  Missing: " << test_labels_file << std::endl;
        std::cerr << "Please run: python3 download_mnist.py" << std::endl;
        return false;
    }
    
    std::cout << "MNIST files found, loading..." << std::endl;
    
    // Load training data
    std::vector<int> train_labels;
    if (!load_mnist_images(train_images_file, train_images)) {
        return false;
    }
    if (!load_mnist_labels(train_labels_file, train_labels)) {
        return false;
    }
    labels_to_onehot(train_labels, train_labels_onehot);
    
    // Load test data
    std::vector<int> test_labels;
    if (!load_mnist_images(test_images_file, test_images)) {
        return false;
    }
    if (!load_mnist_labels(test_labels_file, test_labels)) {
        return false;
    }
    labels_to_onehot(test_labels, test_labels_onehot);
    
    std::cout << "Loaded " << train_images.size() << " training examples" << std::endl;
    std::cout << "Loaded " << test_images.size() << " test examples" << std::endl;
    
    return true;
}

