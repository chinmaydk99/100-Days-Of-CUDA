#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm> // For std::fill

#define CUDA_CHECK(call)                                                          \
{                                                                                \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess)                                                      \
    {                                                                            \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,       \
                cudaGetErrorString(err));                                        \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
}

#define UNVISITED -1

__global__ void bfs_kernel(const int* row_ptr, const int* col_idx, int* level,
                           int* frontier_in, int* frontier_out, int* out_frontier_size,
                           int current_level, int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= num_vertices) return;

    if (frontier_in[v] == current_level) {
        int start_edge = row_ptr[v];
        int end_edge = row_ptr[v + 1];

        for (int edge = start_edge; edge < end_edge; ++edge) {
            int neighbor = col_idx[edge];
            int previous_value = atomicCAS(&level[neighbor], UNVISITED, current_level + 1);
            if (previous_value == UNVISITED) {
                frontier_out[neighbor] = current_level + 1;
                atomicAdd(out_frontier_size, 1);
            }
        }
    }
}


void bfs_gpu(int num_vertices, int num_edges, const std::vector<int>& h_row_ptr,
             const std::vector<int>& h_col_idx, std::vector<int>& h_level, int source_vertex)
{
    if (source_vertex < 0 || source_vertex >= num_vertices) {
        std::cerr << "Error: Source vertex " << source_vertex << " is out of bounds." << std::endl;
        return;
    }

    size_t vertices_size = num_vertices * sizeof(int);
    size_t row_ptr_size = (num_vertices + 1) * sizeof(int);
    size_t col_idx_size = num_edges * sizeof(int);

    int *d_row_ptr, *d_col_idx, *d_level, *d_frontier1, *d_frontier2, *d_frontier_size_atomic;
    int *d_frontier_in, *d_frontier_out; // Pointers to swap between frontier1 and frontier2

    CUDA_CHECK(cudaMalloc(&d_row_ptr, row_ptr_size));
    CUDA_CHECK(cudaMalloc(&d_col_idx, col_idx_size));
    CUDA_CHECK(cudaMalloc(&d_level, vertices_size));
    CUDA_CHECK(cudaMalloc(&d_frontier1, vertices_size));
    CUDA_CHECK(cudaMalloc(&d_frontier2, vertices_size));
    CUDA_CHECK(cudaMalloc(&d_frontier_size_atomic, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), row_ptr_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx.data(), col_idx_size, cudaMemcpyHostToDevice));

    // Initialize level and frontiers on host first for clarity
    std::vector<int> h_frontier_init(num_vertices, UNVISITED);
    std::fill(h_level.begin(), h_level.end(), UNVISITED);

    h_level[source_vertex] = 0;
    h_frontier_init[source_vertex] = 0; // Use level 0 in frontier array
    int h_frontier_size = 1;

    CUDA_CHECK(cudaMemcpy(d_level, h_level.data(), vertices_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier1, h_frontier_init.data(), vertices_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_frontier2, UNVISITED, vertices_size)); // Clear the 'out' frontier buffer
    CUDA_CHECK(cudaMemcpy(d_frontier_size_atomic, &h_frontier_size, sizeof(int), cudaMemcpyHostToDevice));


    d_frontier_in = d_frontier1;
    d_frontier_out = d_frontier2;

    int current_level = 0;
    const int threads_per_block = 256;
    dim3 grid_dim((num_vertices + threads_per_block - 1) / threads_per_block);
    dim3 block_dim(threads_per_block);

    std::cout << "Starting BFS on GPU from source vertex " << source_vertex << "..." << std::endl;

    while (h_frontier_size > 0) {
        int h_out_frontier_size = 0;
        CUDA_CHECK(cudaMemcpy(d_frontier_size_atomic, &h_out_frontier_size, sizeof(int), cudaMemcpyHostToDevice)); // Reset atomic counter

        bfs_kernel<<<grid_dim, block_dim>>>(d_row_ptr, d_col_idx, d_level,
                                            d_frontier_in, d_frontier_out,
                                            d_frontier_size_atomic, current_level, num_vertices);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_frontier_size_atomic, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "  Level " << current_level << ": Frontier size = " << h_frontier_size << std::endl;

        if (h_frontier_size == 0) {
            break;
        }

        // Swap frontiers
        int* temp_ptr = d_frontier_in;
        d_frontier_in = d_frontier_out;
        d_frontier_out = temp_ptr;
        // Note: d_frontier_out (which was the old d_frontier_in) now holds stale level info,
        // but it doesn't matter as it's only read based on the *new* d_frontier_in in the next iteration.
        // We don't strictly need to clear it, but could use cudaMemset if desired.

        current_level++;
    }

    std::cout << "GPU BFS complete." << std::endl;

    CUDA_CHECK(cudaMemcpy(h_level.data(), d_level, vertices_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_level));
    CUDA_CHECK(cudaFree(d_frontier1));
    CUDA_CHECK(cudaFree(d_frontier2));
    CUDA_CHECK(cudaFree(d_frontier_size_atomic));
}

// Helper function for CPU BFS verification
void bfs_cpu(int num_vertices, const std::vector<int>& h_row_ptr,
             const std::vector<int>& h_col_idx, std::vector<int>& h_level_cpu, int source_vertex)
{
    std::fill(h_level_cpu.begin(), h_level_cpu.end(), UNVISITED);
    if (source_vertex < 0 || source_vertex >= num_vertices) return;

    std::queue<int> q;
    q.push(source_vertex);
    h_level_cpu[source_vertex] = 0;
    int current_level = 0;

    while (!q.empty()) {
        int level_size = q.size();
        for (int i = 0; i < level_size; ++i) {
            int u = q.front();
            q.pop();

            int start_edge = h_row_ptr[u];
            int end_edge = h_row_ptr[u + 1];

            for (int edge = start_edge; edge < end_edge; ++edge) {
                int v = h_col_idx[edge];
                if (h_level_cpu[v] == UNVISITED) {
                    h_level_cpu[v] = current_level + 1;
                    q.push(v);
                }
            }
        }
        current_level++;
    }
}

// Helper to create a sample graph (e.g., random graph) in CSR format
void create_random_graph_csr(int num_vertices, int avg_degree,
                             std::vector<int>& row_ptr, std::vector<int>& col_idx)
{
    row_ptr.assign(num_vertices + 1, 0);
    std::vector<std::vector<int>> adj_list(num_vertices);
    long long num_edges = 0;

    for (int i = 0; i < num_vertices; ++i) {
        int degree = std::max(1, static_cast<int>(round(avg_degree + (rand() / (RAND_MAX + 1.0) - 0.5) * avg_degree)));
        for (int j = 0; j < degree; ++j) {
            int neighbor = rand() % num_vertices;
            // Avoid self-loops for simplicity, check for duplicates
            if (i != neighbor) {
                 bool found = false;
                 for(int existing_neighbor : adj_list[i]){
                     if(existing_neighbor == neighbor) { found = true; break; }
                 }
                 if (!found) {
                    adj_list[i].push_back(neighbor);
                    // Assuming undirected graph for simplicity, add back edge
                    // Check for duplicates on the other side too
                     bool found_back = false;
                     for(int existing_neighbor : adj_list[neighbor]){
                        if(existing_neighbor == i) { found_back = true; break; }
                     }
                     if(!found_back) {
                        adj_list[neighbor].push_back(i);
                     }
                 }
            }
        }
    }

    // Build CSR from adjacency list
    col_idx.clear();
    for (int i = 0; i < num_vertices; ++i) {
        std::sort(adj_list[i].begin(), adj_list[i].end()); // Optional: sort neighbors
        row_ptr[i + 1] = row_ptr[i] + adj_list[i].size();
        col_idx.insert(col_idx.end(), adj_list[i].begin(), adj_list[i].end());
    }
    num_edges = col_idx.size();
    std::cout << "Generated random graph: " << num_vertices << " vertices, " << num_edges << " edges." << std::endl;
}


int main() {
    int num_vertices = 1 << 14; // 16k vertices
    int avg_degree = 16;
    int source_vertex = 0;

    std::vector<int> h_row_ptr;
    std::vector<int> h_col_idx;
    create_random_graph_csr(num_vertices, avg_degree, h_row_ptr, h_col_idx);
    int num_edges = h_col_idx.size();

    std::vector<int> h_level_gpu(num_vertices);
    std::vector<int> h_level_cpu(num_vertices);

    // --- GPU Calculation ---
    bfs_gpu(num_vertices, num_edges, h_row_ptr, h_col_idx, h_level_gpu, source_vertex);

    // --- CPU Calculation ---
    std::cout << "Calculating CPU reference..." << std::endl;
    bfs_cpu(num_vertices, h_row_ptr, h_col_idx, h_level_cpu, source_vertex);
    std::cout << "CPU calculation finished." << std::endl;

    // --- Verification ---
    std::cout << "Verifying results..." << std::endl;
    bool passed = true;
    int errors = 0;
    int max_print_errors = 10;
    for (int i = 0; i < num_vertices; ++i) {
        if (h_level_gpu[i] != h_level_cpu[i]) {
            if (errors < max_print_errors) {
                std::cerr << "Verification FAILED at vertex " << i << ": "
                          << "GPU Level=" << h_level_gpu[i] << ", CPU Level=" << h_level_cpu[i] << std::endl;
            }
            passed = false;
            errors++;
        }
    }
    if (errors > max_print_errors) {
         std::cerr << "... (" << (errors - max_print_errors) << " more verification failures not shown)" << std::endl;
    }

    if (passed) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED! (" << errors << " mismatches)" << std::endl;
    }

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}