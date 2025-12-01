// Benchmark utilities inspired by tinygrad's clean, simple approach
#pragma once

#include <chrono>
#include <string>
#include <iomanip>
#include <iostream>

// Simple timing utility - like tinygrad's approach
class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_time).count() / 1000.0;
    }
    double elapsed_us() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_time).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// Benchmark runner - inspired by tinygrad's benchmarking practices
template<typename Func>
double benchmark(const std::string& name, Func&& func, int warmup=3, int iterations=10) {
    // Warmup runs (tinygrad does this)
    for (int i = 0; i < warmup; i++) {
        func();
    }
    
    // Actual benchmark
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        func();
    }
    double elapsed = timer.elapsed_us() / iterations;
    
    std::cout << name << ": " << std::fixed << std::setprecision(2) 
              << (elapsed / 1000.0) << " ms" << std::endl;
    return elapsed;
}

// Calculate TFLOPS from operations and time
inline double calculate_tflops(long long ops, double time_us) {
    double time_sec = time_us / 1e6;
    return (ops / time_sec) / 1e12;
}

// ANSI colors for clean output
namespace Colors {
    constexpr const char* RESET = "\033[0m";
    constexpr const char* RED = "\033[31m";
    constexpr const char* GREEN = "\033[32m";
    constexpr const char* YELLOW = "\033[33m";
    constexpr const char* BLUE = "\033[34m";
    constexpr const char* MAGENTA = "\033[35m";
    constexpr const char* CYAN = "\033[36m";
}

