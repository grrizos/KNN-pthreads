#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <ctime>
#include <cblas.h>
#include <queue>
#include <chrono>
#include <pthread.h>

struct DistanceIndex {
    float distance;
    int index;
    bool operator<(const DistanceIndex& other) const {
        return distance < other.distance;
    }
};

struct ThreadArgs {
    const float* C;
    const float* Q;
    const float* Q_norms;
    int n_C;
    int n_Q;
    int d;
    int k;
    int q_start;
    int q_end;
    int* indices;
    float* distances;
};

void* knnThread(void* args_ptr) {
    ThreadArgs* args = static_cast<ThreadArgs*>(args_ptr);
    const float* C = args->C;
    const float* Q = args->Q;
    const float* Q_norms = args->Q_norms;
    int n_C = args->n_C;
    int n_Q = args->n_Q;
    int d = args->d;
    int k = args->k;
    int q_start = args->q_start;
    int q_end = args->q_end;
    int* indices = args->indices;
    float* distances = args->distances;

    const int block_size = 1024;
    const int num_blocks = (n_C + block_size - 1) / block_size;

    for (int q_idx = q_start; q_idx < q_end; ++q_idx) {
        std::priority_queue<DistanceIndex> heap;

        for (int block = 0; block < num_blocks; ++block) {
            int start = block * block_size;
            int end = std::min((block + 1) * block_size, n_C);
            int cur_block_size = end - start;
            const float* C_block = C + start * d;

            // Compute C norms
            std::vector<float> C_norms(cur_block_size, 0.0f);
            for (int i = 0; i < cur_block_size; ++i) {
                const float* c_ptr = C_block + i * d;
                for (int dim = 0; dim < d; ++dim) {
                    C_norms[i] += c_ptr[dim] * c_ptr[dim];
                }
            }

            // Compute dot products
            std::vector<float> products(cur_block_size, 0.0f);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        cur_block_size, 1, d,
                        1.0f, C_block, d,
                              Q + q_idx * d, d,
                        0.0f, products.data(), 1);

            // Calculate distances and maintain heap
            for (int c_idx = 0; c_idx < cur_block_size; ++c_idx) {
                float dist2 = C_norms[c_idx] + Q_norms[q_idx] - 2 * products[c_idx];
                float dist = std::sqrt(std::max(dist2, 0.0f));
                int global_idx = start + c_idx;

                if ((int)heap.size() < k) {
                    heap.push({dist, global_idx});
                } else if (dist < heap.top().distance) {
                    heap.pop();
                    heap.push({dist, global_idx});
                }
            }
        }

        std::vector<DistanceIndex> results;
        while (!heap.empty()) {
            results.push_back(heap.top());
            heap.pop();
        }
        std::reverse(results.begin(), results.end());
        for (int i = 0; i < k; ++i) {
            if (i < (int)results.size()) {
                indices[q_idx * k + i] = results[i].index;
                distances[q_idx * k + i] = results[i].distance;
            } else {
                indices[q_idx * k + i] = -1;
                distances[q_idx * k + i] = INFINITY;
            }
        }
    }

    return nullptr;
}

int main() {
    const int n_C = 100000;
    const int n_Q = 100000;
    const int d = 20;
    const int k = 5;
    const int num_threads = 1;

    std::vector<float> C(n_C * d);
    std::vector<float> Q(n_Q * d);
    std::vector<int> indices(n_Q * k);
    std::vector<float> distances(n_Q * k);

    std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& val : C) val = dist(rng);
    for (auto& val : Q) val = dist(rng);

    std::vector<float> Q_norms(n_Q, 0.0f);
    for (int j = 0; j < n_Q; ++j) {
        const float* q_ptr = Q.data() + j * d;
        for (int dim = 0; dim < d; ++dim) {
            Q_norms[j] += q_ptr[dim] * q_ptr[dim];
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    pthread_t threads[num_threads];
    ThreadArgs args[num_threads];

    int chunk = (n_Q + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        args[i] = {
            C.data(), Q.data(), Q_norms.data(),
            n_C, n_Q, d, k,
            i * chunk, std::min((i + 1) * chunk, n_Q),
            indices.data(), distances.data()
        };
        pthread_create(&threads[i], nullptr, knnThread, &args[i]);
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Multithreaded knnsearch execution time: " 
              << duration.count() << " seconds." << std::endl;

    return 0;
}
