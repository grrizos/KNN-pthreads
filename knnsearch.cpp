#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>      
#include <ctime>       
#include <cblas.h>     
#include <queue>
#include <chrono>

struct DistanceIndex {
    float distance;
    int index;
    bool operator<(const DistanceIndex& other) const {
        return distance < other.distance; // για max-heap
    }
};

void knnsearch(const float* C, const float* Q, int n_C, int n_Q, int d, int k, int* indices, float* distances) {
    // Υπολογισμός των norms των Q
    std::vector<float> Q_norms(n_Q, 0.0f);
    for (int j = 0; j < n_Q; ++j) {
        const float* q_ptr = Q + j * d;
        for (int dim = 0; dim < d; ++dim) {
            Q_norms[j] += q_ptr[dim] * q_ptr[dim];
        }
    }

    // Block-based επεξεργασία
    const int block_size = 1024;
    const int num_blocks = (n_C + block_size - 1) / block_size;

    // Ένα heap για κάθε query
    std::vector<std::priority_queue<DistanceIndex>> heaps(n_Q);

    for (int block = 0; block < num_blocks; ++block) {
        int start = block * block_size;
        int end = std::min((block + 1) * block_size, n_C);
        int current_block_size = end - start;
        const float* C_block = C + start * d;

        // Norms του C block
        std::vector<float> C_norms(current_block_size, 0.0f);
        for (int i = 0; i < current_block_size; ++i) {
            const float* c_ptr = C_block + i * d;
            for (int dim = 0; dim < d; ++dim) {
                C_norms[i] += c_ptr[dim] * c_ptr[dim];
            }
        }

        // Υπολογισμός C_block * Q^T
        std::vector<float> products(current_block_size * n_Q, 0.0f);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    current_block_size, n_Q, d,
                    1.0f, C_block, d,
                          Q, d,
                    0.0f, products.data(), n_Q);

        // Ενημέρωση heaps
        for (int q_idx = 0; q_idx < n_Q; ++q_idx) {
            for (int c_idx = 0; c_idx < current_block_size; ++c_idx) {
                float dist2 = C_norms[c_idx] + Q_norms[q_idx] - 2 * products[c_idx * n_Q + q_idx];
                float distance = std::sqrt(std::max(dist2, 0.0f)); // για αποφυγή αρνητικών λόγω σφαλμάτων (todo)
                int global_idx = start + c_idx;

                auto& heap = heaps[q_idx];
                if ((int)heap.size() < k) {
                    heap.push({distance, global_idx});
                } else if (distance < heap.top().distance) {
                    heap.pop();
                    heap.push({distance, global_idx});
                }
            }
        }
    }

    // Εξαγωγή αποτελεσμάτων
    for (int j = 0; j < n_Q; ++j) {
        std::vector<DistanceIndex> results;
        while (!heaps[j].empty()) {
            results.push_back(heaps[j].top());
            heaps[j].pop();
        }
        std::reverse(results.begin(), results.end());

        for (int i = 0; i < k; ++i) {
            if (i < (int)results.size()) {
                indices[j * k + i] = results[i].index;
                distances[j * k + i] = results[i].distance;
            } else {
                indices[j * k + i] = -1;
                distances[j * k + i] = INFINITY;
            }
        }
    }
}

int main() {
    // Παράμετροι
    const int n_C = 100000; 
    const int n_Q = 100000;   
    const int d = 20;      
    const int k = 5;      

    std::vector<float> C(n_C * d);
    std::vector<float> Q(n_Q * d);


    std::mt19937 rng(static_cast<unsigned>(std::time(nullptr))); // random number generator
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);   


    for (auto& val : C) val = dist(rng);
    for (auto& val : Q) val = dist(rng);

    // Χώρος για τα αποτελέσματα
    std::vector<int> indices(n_Q * k);
    std::vector<float> distances(n_Q * k);

    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    knnsearch(C.data(), Q.data(), n_C, n_Q, d, k, indices.data(), distances.data());
    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time
    std::chrono::duration<float> duration = end - start;
    std::cout << "Vanilla knnsearch (no threads) execution time: " 
            << duration.count() << " seconds." << std::endl;

    return 0;
}