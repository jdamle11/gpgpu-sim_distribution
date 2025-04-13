#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

// DSM Shared Memory
__shared__  int product_results[2];
__shared__  int barrier_counter;
__shared__  int barrier_sense;

// Manual DSM barrier
__device__ void my_cluster_barrier(int num_blocks, int* counter, int* sense) {
    __shared__ int dummy;
    __syncthreads();

    if (threadIdx.x == 0) {
        int val = atomicAdd(counter, 1);
        if (val == num_blocks - 1) {
            *counter = 0;
            *sense = 1;
        } else {
            while (*sense == 0) {
                // spin wait (not ideal, but no __nanosleep)
            }
        }
    }
    __syncthreads();
}

__global__ void dsm_product_kernel_noatomics(int* output) {
    int tid = threadIdx.x;
    int block_rank = blockIdx.x;

    __shared__ int partial[32]; // assuming max 32 threads
    if (tid < 32) partial[tid] = 1;

    // Each thread multiplies a number (1–5 or 6–10 depending on block)
    int base = block_rank * 5;
    for (int i = tid; i < 5; i += blockDim.x) {
        partial[tid] *= (base + i + 1);
    }

    __syncthreads();

    // Thread 0 reduces partial products
    __shared__ int local_product;
    if (tid == 0) {
        local_product = 1;
        for (int i = 0; i < blockDim.x; i++) {
            local_product *= partial[i];
        }
        product_results[block_rank] = local_product;
    }

    // Custom DSM barrier
    my_cluster_barrier(2, &barrier_counter, &barrier_sense);

    if (block_rank == 0 && tid == 0) {
        int final_sum = product_results[0] + product_results[1];
        *output = final_sum;
    }
}

int main() {
    int* d_output;
    CHECK(cudaMalloc(&d_output, sizeof(int)));

    // Cluster configuration
    CHECK(cudaFuncSetAttribute(dsm_product_kernel_noatomics,
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CHECK(cudaFuncSetAttribute(dsm_product_kernel_noatomics,
        cudaFuncAttributeClusterDimMustBeSet, 1));
    CHECK(cudaFuncSetAttribute(dsm_product_kernel_noatomics,
        cudaFuncAttributeRequiredClusterWidth, 2));

    dsm_product_kernel_noatomics<<<2, 32>>>(d_output);
    CHECK(cudaDeviceSynchronize());

    int h_output;
    CHECK(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Sum of the two products = %d\n", h_output);  // Expect 30360

    CHECK(cudaFree(d_output));
    return 0;
}
